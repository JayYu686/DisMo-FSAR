#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Train a video classification model."""
import numpy as np
import pprint
import torch
import torch.nn.functional as F
import math
import os
import glob
import json
try:
    import oss2 as oss
except ImportError:
    oss = None
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import models.utils.losses as losses
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
import utils.bucket as bu
from utils.meters import TrainMeter, ValMeter

from models.base.builder import build_model
from datasets.base.builder import build_loader, shuffle_dataset

from datasets.utils.mixup import Mixup

logger = logging.get_logger(__name__)


def _use_batched_episodic_tasks(cfg):
    return bool(getattr(cfg.TRAIN, "BATCH_EPISODIC_TASKS", False))


def _flatten_task_tensor(value):
    if not torch.is_tensor(value):
        raise TypeError("Expected tensor value for episodic task flattening.")
    return value.reshape(-1)


def _get_task_query_labels(task_dict):
    return _flatten_task_tensor(task_dict["target_labels"]).long()


def _get_task_query_count(task_dict):
    return int(_flatten_task_tensor(task_dict["target_labels"]).numel())


def _get_grad_accum_steps(cfg):
    legacy_steps = int(getattr(cfg.TRAIN, "BATCH_SIZE_PER_TASK", 1))
    grad_accum_steps = int(getattr(cfg.TRAIN, "GRAD_ACCUM_STEPS", legacy_steps))
    if grad_accum_steps <= 0:
        raise ValueError(
            "TRAIN.GRAD_ACCUM_STEPS must be a positive integer, "
            f"but got {grad_accum_steps}"
        )
    return grad_accum_steps


def _get_accumulation_style(cfg):
    style = str(getattr(cfg.TRAIN, "ACCUMULATION_STYLE", "standard")).lower()
    if style not in {"standard", "d2st_legacy"}:
        raise ValueError(
            "TRAIN.ACCUMULATION_STYLE must be 'standard' or 'd2st_legacy', "
            f"but got {style}"
        )
    return style


def _get_optimizer_step_interval(cfg):
    if _get_accumulation_style(cfg) == "d2st_legacy":
        step_interval = int(getattr(cfg.TRAIN, "BATCH_SIZE_PER_TASK", 1))
        if step_interval <= 0:
            raise ValueError(
                "TRAIN.BATCH_SIZE_PER_TASK must be a positive integer for "
                f"d2st_legacy accumulation, but got {step_interval}"
            )
        return step_interval
    return _get_grad_accum_steps(cfg)


def _prepare_task_dict(task_dict, cfg):
    """Move episodic batch tensors to device with optional task-batch preservation."""
    use_gpu = misc.get_num_gpus(cfg) > 0
    keep_task_batch = _use_batched_episodic_tasks(cfg)
    for k in list(task_dict.keys()):
        value = task_dict[k]
        if torch.is_tensor(value):
            # Legacy few-shot heads consume one task at a time. CAST mainline keeps
            # the leading task-batch dimension so multiple episodic tasks can be
            # processed per rank in a single step.
            if not keep_task_batch and value.dim() > 0:
                value = value[0]
            if use_gpu:
                value = value.cuda(non_blocking=True)
            task_dict[k] = value
    return task_dict


def _extract_eval_metric(eval_stats, metric_key):
    """Safely fetch a metric value from eval return dict."""
    if not isinstance(eval_stats, dict):
        return None
    metric_val = eval_stats.get(metric_key, None)
    if metric_val is None:
        return None
    return float(metric_val)


def _is_better_metric(new_val, best_val, mode):
    """Compare metrics with support for min/max mode."""
    if new_val is None:
        return False
    if best_val is None:
        return True
    if mode == "max":
        return new_val > best_val
    return new_val < best_val


def _format_metric_for_filename(metric_name, metric_value):
    if metric_name is None or metric_value is None:
        return None
    safe_name = str(metric_name).replace("/", "_").replace(" ", "")
    return "{}{:.2f}".format(safe_name, float(metric_value))


def _promote_to_best_checkpoint(saved_checkpoint_path, cfg, eval_round=None, metric_name=None, metric_value=None):
    """Keep only one best checkpoint file in checkpoint dir, with round and metric in filename."""
    checkpoint_dir = cu.get_checkpoint_dir(cfg.OUTPUT_DIR)
    best_name = getattr(cfg.TRAIN, "BEST_CHECKPOINT_NAME", "checkpoint_best.pyth")
    stem, ext = os.path.splitext(best_name)
    if ext == "":
        ext = ".pyth"

    suffix_parts = []
    if eval_round is not None:
        suffix_parts.append("round{:03d}".format(int(eval_round)))
    metric_part = _format_metric_for_filename(metric_name, metric_value)
    if metric_part is not None:
        suffix_parts.append(metric_part)

    dynamic_best_name = "{}_{}{}".format(stem, "_".join(suffix_parts), ext) if suffix_parts else "{}{}".format(stem, ext)
    best_path = os.path.join(checkpoint_dir, dynamic_best_name)

    # Replace previous best with the latest promoted checkpoint.
    os.replace(saved_checkpoint_path, best_path)

    # Remove any epoch-style checkpoints and old best checkpoints so only current best is retained.
    for stale in glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pyth")):
        try:
            os.remove(stale)
        except OSError:
            pass
    for stale in glob.glob(os.path.join(checkpoint_dir, "{}*{}".format(stem, ext))):
        if os.path.abspath(stale) == os.path.abspath(best_path):
            continue
        try:
            os.remove(stale)
        except OSError:
            pass

    return best_path


def _cleanup_epoch_checkpoints(cfg):
    """Delete stale epoch checkpoints when best-only policy is enabled."""
    checkpoint_dir = cu.get_checkpoint_dir(cfg.OUTPUT_DIR)
    if not os.path.isdir(checkpoint_dir):
        return

    removed = 0
    for stale in glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pyth")):
        try:
            os.remove(stale)
            removed += 1
        except OSError:
            pass

    if removed > 0:
        logger.info("[BEST_CKPT] Removed %d stale epoch checkpoints.", removed)


def _recover_best_metric_from_log(cfg, metric_key, mode):
    """Recover best metric from existing training log for resume scenarios."""
    log_path = os.path.join(cfg.OUTPUT_DIR, cfg.TRAIN.LOG_FILE)
    if not os.path.exists(log_path):
        return None

    best_metric = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            json_start = line.find("{")
            if json_start < 0:
                continue
            try:
                stats = json.loads(line[json_start:])
            except Exception:
                continue
            # Best-checkpoint promotion is based on non-EMA validation stats.
            # Keep resume behavior consistent with in-training selection.
            if stats.get("_type") != "val_epoch":
                continue
            if metric_key not in stats:
                continue
            try:
                metric_val = float(stats[metric_key])
            except Exception:
                continue
            if _is_better_metric(metric_val, best_metric, mode):
                best_metric = metric_val

    return best_metric


def _compute_few_shot_loss(task_dict, model_dict, cfg):
    """Compute few-shot loss with optional auxiliary terms."""
    accumulation_style = _get_accumulation_style(cfg)
    target_labels = _get_task_query_labels(task_dict)
    main_loss = F.cross_entropy(model_dict["logits"], target_labels)

    if (
        hasattr(cfg.TRAIN, "USE_CLASSIFICATION")
        and cfg.TRAIN.USE_CLASSIFICATION
        and model_dict.get("class_logits") is not None
    ):
        real_labels = torch.cat(
            [
                _flatten_task_tensor(task_dict["real_support_labels"]),
                _flatten_task_tensor(task_dict["real_target_labels"]),
            ],
            dim=0,
        )
        if hasattr(cfg.TRAIN, "USE_LOCAL") and cfg.TRAIN.USE_LOCAL:
            real_labels = real_labels.unsqueeze(1).repeat(1, cfg.DATA.NUM_INPUT_FRAMES).reshape(-1)
        cls_loss = F.cross_entropy(model_dict["class_logits"], real_labels.long())
        cls_weight = float(getattr(cfg.TRAIN, "USE_CLASSIFICATION_VALUE", 1.0))
        if hasattr(cfg, "SEGREL") and hasattr(cfg.SEGREL, "CLASSIFICATION_WEIGHT"):
            cls_weight = float(cfg.SEGREL.CLASSIFICATION_WEIGHT)
        if hasattr(cfg, "DARM") and hasattr(cfg.DARM, "CLASSIFICATION_WEIGHT"):
            cls_weight = float(cfg.DARM.CLASSIFICATION_WEIGHT)
        if hasattr(cfg, "RSM") and hasattr(cfg.RSM, "CLASSIFICATION_WEIGHT"):
            cls_weight = float(cfg.RSM.CLASSIFICATION_WEIGHT)
        if hasattr(cfg, "SEMCAL") and hasattr(cfg.SEMCAL, "CLASSIFICATION_WEIGHT"):
            cls_weight = float(cfg.SEMCAL.CLASSIFICATION_WEIGHT)
        main_loss = main_loss + cls_weight * cls_loss

    aux_losses = model_dict.get("aux_losses", {})
    if isinstance(aux_losses, dict):
        for aux_loss in aux_losses.values():
            if isinstance(aux_loss, torch.Tensor):
                main_loss = main_loss + aux_loss

    if accumulation_style == "d2st_legacy":
        batch_size = int(getattr(cfg.TRAIN, "BATCH_SIZE", 1))
        if batch_size <= 0:
            raise ValueError(
                "TRAIN.BATCH_SIZE must be a positive integer for d2st_legacy "
                f"accumulation, but got {batch_size}"
            )
        return main_loss / batch_size

    return main_loss / _get_grad_accum_steps(cfg)


def train_epoch(
    train_loader, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg, writer=None, val_meter=None, val_loader=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    norm_train = False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)) and module.training:
            norm_train = True
    logger.info(f"Norm training: {norm_train}")
    train_meter.iter_tic()
    train_loader_len = len(train_loader)
    if train_loader_len <= 0:
        raise ValueError("Train loader is empty. Please check TRAIN.DATASET/ANNO_DIR/data splits.")

    save_best_only = bool(getattr(cfg.TRAIN, "SAVE_BEST_ONLY", False))
    best_metric_key = getattr(cfg.TRAIN, "BEST_METRIC", "top1_acc")
    best_metric_mode = str(getattr(cfg.TRAIN, "BEST_MODE", "max")).lower()
    if best_metric_mode not in ["min", "max"]:
        raise ValueError("TRAIN.BEST_MODE must be 'min' or 'max'")
    best_metric_value = None
    if save_best_only:
        if val_loader is None:
            raise ValueError(
                "TRAIN.SAVE_BEST_ONLY=True requires validation loader. "
                "Please enable eval and provide TEST dataset/batch settings."
            )
        recovered_best = _recover_best_metric_from_log(cfg, best_metric_key, best_metric_mode)
        if recovered_best is not None:
            best_metric_value = recovered_best
            logger.info(
                "[BEST_CKPT] Recovered historical best {}={:.6f} from log.".format(
                    best_metric_key, recovered_best
                )
            )
        _cleanup_epoch_checkpoints(cfg)

    steps_iter = int(getattr(cfg.SOLVER, "STEPS_ITER", train_loader_len))
    if steps_iter <= 0:
        steps_iter = train_loader_len
    data_size = steps_iter
    # Keep meter epoch/iter display aligned with synthetic epoch length.
    train_meter.epoch_iters = steps_iter
    train_meter.MAX_EPOCH = cfg.OPTIMIZER.MAX_EPOCH * steps_iter
    amp_enabled = bool(getattr(cfg.TRAIN, "AMP_ENABLE", False)) and misc.get_num_gpus(cfg) > 0
    scaler = GradScaler(enabled=amp_enabled)
    optimizer_step_interval = _get_optimizer_step_interval(cfg)

    if int(cfg.TRAIN.VAL_FRE_ITER) <= 0:
        raise ValueError(
            "TRAIN.VAL_FRE_ITER must be a positive integer, "
            "but got {}".format(cfg.TRAIN.VAL_FRE_ITER)
        )

    total_train_tasks_global = int(cfg.TRAIN.NUM_TRAIN_TASKS)
    if total_train_tasks_global <= 0:
        raise ValueError(
            "TRAIN.NUM_TRAIN_TASKS must be a positive integer, "
            "but got {}".format(cfg.TRAIN.NUM_TRAIN_TASKS)
        )
    world_size = max(1, du.get_world_size())
    # NUM_TRAIN_TASKS is configured as global tasks. Convert to per-rank tasks.
    total_train_tasks = int(math.ceil(float(total_train_tasks_global) / float(world_size)))
    if du.is_master_proc():
        logger.info(
            "Training task budget: global=%d, world_size=%d, per_rank=%d, steps_iter=%d",
            total_train_tasks_global,
            world_size,
            total_train_tasks,
            steps_iter,
        )

    # Iterate over per-rank task budget, restarting the loader when exhausted.
    loader_epoch = 0
    task_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    for cur_iter in range(total_train_tasks):
        try:
            task_dict = next(task_iter)
        except StopIteration:
            loader_epoch += 1
            shuffle_dataset(train_loader, loader_epoch)
            task_iter = iter(train_loader)
            task_dict = next(task_iter)
        
        '''['support_set', 'support_labels', 'target_set', 'target_labels', 'real_target_labels', 'batch_class_list', "real_support_labels"]'''
        # Save a checkpoint.
        cur_epoch = cur_iter // steps_iter
        iter_in_epoch = cur_iter % steps_iter
        # if (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0 and cur_iter>=200:   # 
        if (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0:   # 
            
            if cfg.OSS.ENABLE:
                model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
                model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
            else:
                model_bucket = None
            cur_epoch_save = cur_iter//cfg.TRAIN.VAL_FRE_ITER
            if save_best_only:
                val_meter.set_model_ema_enabled(False)
                eval_stats = eval_epoch(
                    val_loader, model, val_meter, cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1, cfg, writer
                )
                if model_ema is not None:
                    val_meter.set_model_ema_enabled(True)
                    eval_epoch(val_loader, model_ema.module, val_meter, cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1, cfg, writer)
                model.train()

                metric_value = _extract_eval_metric(eval_stats, best_metric_key)
                should_save = False
                if metric_value is None:
                    # Fallback: keep first checkpoint if metric is unavailable.
                    should_save = best_metric_value is None
                else:
                    should_save = _is_better_metric(metric_value, best_metric_value, best_metric_mode)

                if should_save:
                    saved_path = cu.save_checkpoint(
                        cfg.OUTPUT_DIR,
                        model,
                        model_ema,
                        optimizer,
                        cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1,
                        cfg,
                        model_bucket,
                    )
                    # Keep best metric state consistent on all ranks.
                    if metric_value is not None:
                        best_metric_value = metric_value

                    # save_checkpoint only returns a path on master rank.
                    if saved_path is not None:
                        best_path = _promote_to_best_checkpoint(
                            saved_path,
                            cfg,
                            eval_round=cur_epoch_save + 1,
                            metric_name=best_metric_key,
                            metric_value=metric_value,
                        )
                        if metric_value is not None:
                            logger.info(
                                "[BEST_CKPT] Updated best {}={:.6f}, saved {}".format(
                                    best_metric_key, metric_value, best_path
                                )
                            )
                        else:
                            logger.info(
                                "[BEST_CKPT] Metric '{}' unavailable, saved fallback checkpoint {}".format(
                                    best_metric_key, best_path
                                )
                            )
                else:
                    logger.info(
                        "[BEST_CKPT] Keep existing best. current {}={}, best={}".format(
                            best_metric_key,
                            "None" if metric_value is None else "{:.6f}".format(metric_value),
                            "None" if best_metric_value is None else "{:.6f}".format(best_metric_value),
                        )
                    )
                if misc.get_num_gpus(cfg) > 1:
                    du.synchronize()
            else:
                cu.save_checkpoint(
                    cfg.OUTPUT_DIR,
                    model,
                    model_ema,
                    optimizer,
                    cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1,
                    cfg,
                    model_bucket,
                )

                val_meter.set_model_ema_enabled(False)
                eval_epoch(val_loader, model, val_meter, cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1, cfg, writer)
                if model_ema is not None:
                    val_meter.set_model_ema_enabled(True)
                    eval_epoch(val_loader, model_ema.module, val_meter, cur_epoch_save + cfg.TRAIN.NUM_FOLDS - 1, cfg, writer)
                model.train()

        task_dict = _prepare_task_dict(task_dict, cfg)
            

        if mixup_fn is not None:
            inputs, labels["supervised_mixup"] = mixup_fn(inputs, labels["supervised"])


        # Update the learning rate.
        # Use intra-epoch progress to avoid double-counting global steps in LR schedule.
        lr = optim.get_epoch_lr(cur_epoch + cfg.TRAIN.NUM_FOLDS * float(iter_in_epoch) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

        else:
           
            with autocast(enabled=amp_enabled):
                model_dict = model(task_dict)

        target_logits = model_dict['logits']

        with autocast(enabled=amp_enabled):
            loss = _compute_few_shot_loss(task_dict, model_dict, cfg)
       
        # check Nan Loss.
        if math.isnan(loss):
            # logger.info(f"logits: {model_dict}")
            if amp_enabled:
                scaler.scale(loss).backward(retain_graph=False)
            else:
                loss.backward(retain_graph=False)
            optimizer.zero_grad()
            continue
        if amp_enabled:
            scaler.scale(loss).backward(retain_graph=False)
        else:
            loss.backward(retain_graph=False)

        # optimize
        should_step = ((cur_iter + 1) % optimizer_step_interval == 0) or ((cur_iter + 1) == total_train_tasks)
        if should_step:
            if hasattr(cfg.TRAIN,"CLIP_GRAD_NORM") and cfg.TRAIN.CLIP_GRAD_NORM:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.TRAIN.CLIP_GRAD_NORM)
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # self.scheduler.step()

        if hasattr(cfg, "MULTI_MODAL") and\
            cfg.PRETRAIN.PROTOTYPE.ENABLE and\
            cur_epoch < cfg.PRETRAIN.PROTOTYPE.FREEZE_EPOCHS:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        # Update the parameters.
        # optimizer.step()
        if model_ema is not None:
            model_ema.update(model)

        if cfg.DETECTION.ENABLE or cfg.PRETRAIN.ENABLE:
            if misc.get_num_gpus(cfg) > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                None, None, loss, lr, inputs["video"].shape[0] if isinstance(inputs, dict) else inputs.shape[0]
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + iter_in_epoch,
                )
            if cfg.PRETRAIN.ENABLE:
                train_meter.update_custom_stats(loss_in_parts)

        else:
            top1_err, top5_err = None, None
            if isinstance(task_dict['target_labels'], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 5))
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                if misc.get_num_gpus(cfg) > 1:
                    loss = du.all_reduce([loss])[0].item()
                    for k, v in loss_in_parts.items():
                        loss_in_parts[k] = du.all_reduce([v])[0].item()
                else:
                    loss = loss.item()
                    for k, v in loss_in_parts.items():
                        loss_in_parts[k] = v.item()
                train_meter.update_custom_stats(loss_in_parts)
                train_meter.update_custom_stats(top1_err_all)
                train_meter.update_custom_stats(top5_err_all)
            else:
                # Compute the errors.
                preds = target_logits
                flat_labels = _get_task_query_labels(task_dict)
                num_topks_correct = metrics.topks_correct(preds, flat_labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if misc.get_num_gpus(cfg) > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                _get_task_query_count(task_dict)
                * max(misc.get_num_gpus(cfg), 1),
            )
            
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_acc": 100.0 - top1_err,
                        "Train/Top5_acc": 100.0 - top5_err,
                    },
                    global_step=data_size * cur_epoch + iter_in_epoch,
                )

        train_meter.log_iter_stats(cur_epoch, iter_in_epoch)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch+cfg.TRAIN.NUM_FOLDS-1)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    amp_enabled = bool(getattr(cfg.TRAIN, "AMP_ENABLE", False)) and misc.get_num_gpus(cfg) > 0

    for cur_iter, task_dict in enumerate(val_loader):
        if cur_iter >= cfg.TRAIN.NUM_TEST_TASKS:
            break
        task_dict = _prepare_task_dict(task_dict, cfg)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if misc.get_num_gpus(cfg):
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if misc.get_num_gpus(cfg) > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        elif cfg.PRETRAIN.ENABLE and (cfg.PRETRAIN.GENERATOR == 'PCMGenerator'):
            preds, logits = model(inputs)
            if "move_x" in preds.keys():
                preds["move_joint"] = preds["move_x"]
            elif "move_y" in preds.keys():
                preds["move_joint"] = preds["move_y"]
            num_topks_correct = metrics.topks_correct(preds["move_joint"], labels["self-supervised"]["move_joint"].reshape(preds["move_joint"].shape[0]), (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds["move_joint"].shape[0]) * 100.0 for x in num_topks_correct
            ]
            if misc.get_num_gpus(cfg) > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            val_meter.iter_toc()
            val_meter.update_stats(
                top1_err,
                top5_err,
                preds["move_joint"].shape[0]
                * max(
                    misc.get_num_gpus(cfg), 1
                ),
            )
            val_meter.update_predictions(preds, labels)
        else:
            # preds, logits = model(inputs)
            with autocast(enabled=amp_enabled):
                model_dict = model(task_dict)

            # loss, loss_in_parts, weight = losses.calculate_loss(cfg, preds, logits, labels, cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size)
            target_logits = model_dict['logits']
            flat_labels = _get_task_query_labels(task_dict)
            loss = F.cross_entropy(model_dict["logits"], flat_labels)


            top1_err, top5_err = None, None
            if isinstance(task_dict['target_labels'], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 5))
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                val_meter.update_custom_stats(top1_err_all)
                val_meter.update_custom_stats(top5_err_all)
            else:
                # Compute the errors.
                labels = flat_labels
                preds = target_logits
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if misc.get_num_gpus(cfg) > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )
            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                _get_task_query_count(task_dict)
                * max(misc.get_num_gpus(cfg), 1),
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_acc": 100.0 - top1_err, "Val/Top5_acc": 100.0 - top5_err},
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Capture summary metrics before meter reset.
    val_top1_err = None
    val_top5_err = None
    val_top1_acc = None
    val_top5_acc = None
    if hasattr(val_meter, "num_samples") and val_meter.num_samples > 0:
        val_top1_err = val_meter.num_top1_mis / val_meter.num_samples
        val_top5_err = val_meter.num_top5_mis / val_meter.num_samples
        val_top1_acc = 100.0 - val_top1_err
        val_top5_acc = 100.0 - val_top5_err

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if misc.get_num_gpus(cfg):
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()
    return {
        "top1_acc": val_top1_acc,
        "top5_acc": val_top5_acc,
        "top1_err": val_top1_err,
        "top5_err": val_top5_err,
    }

def train_few_shot(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)

    optim_cfg_changes = optim.sync_solver_optimizer_cfg(cfg)
    if optim_cfg_changes:
        logger.info("Synchronized OPTIMIZER from SOLVER: %s", optim_cfg_changes)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model, model_ema = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    if cfg.OSS.ENABLE:
        model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
    else:
        model_bucket = None

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, model_ema, optimizer, model_bucket)

    # Create the video train and val loaders.
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None  # val

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg) if val_loader is not None else None

    if cfg.AUGMENTATION.MIXUP.ENABLE or cfg.AUGMENTATION.CUTMIX.ENABLE:
        logger.info("Enabling mixup/cutmix.")
        mixup_fn = Mixup(cfg)
        cfg.TRAIN.LOSS_FUNC = "soft_target"
    else:
        logger.info("Mixup/cutmix disabled.")
        mixup_fn = None

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        misc.get_num_gpus(cfg)
    ):
        pass
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    assert (cfg.SOLVER.MAX_EPOCH-start_epoch)%cfg.TRAIN.NUM_FOLDS == 0, "Total training epochs should be divisible by cfg.TRAIN.NUM_FOLDS."

    cur_epoch = 0
    shuffle_dataset(train_loader, cur_epoch)
    
    train_epoch(
        train_loader, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg, writer, val_meter, val_loader
    )
    # torch.cuda.empty_cache()
    if writer is not None:
        writer.close()
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TRAIN.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )
