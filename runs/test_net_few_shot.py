#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Train a video classification model."""
import csv
import json
import numpy as np
import pprint
import torch
import torch.nn.functional as F
import math
import os
import time
try:
    import oss2 as oss
except ImportError:
    oss = None
import torch.nn as nn

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


def _prepare_task_dict(task_dict, cfg):
    use_gpu = misc.get_num_gpus(cfg) > 0
    keep_task_batch = _use_batched_episodic_tasks(cfg)
    for k in list(task_dict.keys()):
        value = task_dict[k]
        if torch.is_tensor(value):
            if not keep_task_batch and value.dim() > 0:
                value = value[0]
            if use_gpu:
                value = value.cuda(non_blocking=True)
            task_dict[k] = value
    return task_dict


def _flatten_task_tensor(value):
    if not torch.is_tensor(value):
        raise TypeError("Expected tensor value for episodic task flattening.")
    return value.reshape(-1)


def _get_task_query_labels(task_dict):
    return _flatten_task_tensor(task_dict["target_labels"]).long()


def _get_task_query_count(task_dict):
    return int(_flatten_task_tensor(task_dict["target_labels"]).numel())


def _get_confusion_targets(task_dict, pred_local):
    batch_class_list = task_dict["batch_class_list"]
    real_target_labels = task_dict["real_target_labels"]

    if batch_class_list.dim() == 1:
        batch_class_list = batch_class_list.unsqueeze(0)
    if real_target_labels.dim() == 1:
        real_target_labels = real_target_labels.unsqueeze(0)

    num_tasks, num_queries = real_target_labels.shape
    class_lookup = batch_class_list.unsqueeze(1).expand(-1, num_queries, -1).reshape(-1, batch_class_list.shape[-1]).long()
    true_real = real_target_labels.reshape(-1).long()
    pred_real = class_lookup.gather(1, pred_local.view(-1, 1).long()).squeeze(1)
    return true_real, pred_real


def _get_eval_class_names(cfg):
    class_names = []
    if hasattr(cfg, "TEST") and hasattr(cfg.TEST, "CLASS_NAME") and cfg.TEST.CLASS_NAME:
        class_names = list(cfg.TEST.CLASS_NAME)
    elif hasattr(cfg, "TRAIN") and hasattr(cfg.TRAIN, "CLASS_NAME") and cfg.TRAIN.CLASS_NAME:
        class_names = list(cfg.TRAIN.CLASS_NAME)
    return class_names


def _save_confusion_artifacts(cfg, confusion_matrix, class_names):
    if confusion_matrix is None or not class_names or not du.is_master_proc():
        return

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    prefix = os.path.splitext(cfg.TEST.LOG_FILE)[0]
    matrix = confusion_matrix.cpu().numpy().astype(np.int64)

    row_sums = matrix.sum(axis=1)
    diag = np.diag(matrix)
    per_class_acc = np.divide(
        diag,
        row_sums,
        out=np.zeros_like(diag, dtype=np.float64),
        where=row_sums > 0,
    )

    hardest_classes = []
    for idx, name in enumerate(class_names):
        hardest_classes.append(
            {
                "class_index": idx,
                "class_name": name,
                "correct": int(diag[idx]),
                "total": int(row_sums[idx]),
                "accuracy": float(per_class_acc[idx]),
            }
        )
    hardest_classes.sort(key=lambda x: (x["accuracy"], x["total"], x["class_index"]))

    top_confusion_pairs = []
    for idx, name in enumerate(class_names):
        row = matrix[idx].copy()
        row[idx] = 0
        pred_idx = int(row.argmax()) if row.size else idx
        count = int(row[pred_idx]) if row.size else 0
        if count <= 0:
            continue
        top_confusion_pairs.append(
            {
                "true_index": idx,
                "true_class": name,
                "pred_index": pred_idx,
                "pred_class": class_names[pred_idx],
                "count": count,
                "row_total": int(row_sums[idx]),
                "confusion_rate": float(count / row_sums[idx]) if row_sums[idx] > 0 else 0.0,
            }
        )
    top_confusion_pairs.sort(key=lambda x: (x["count"], x["confusion_rate"]), reverse=True)

    confusion_csv = os.path.join(cfg.OUTPUT_DIR, f"{prefix}_confusion_matrix.csv")
    hardest_json = os.path.join(cfg.OUTPUT_DIR, f"{prefix}_hardest_classes.json")
    confusions_json = os.path.join(cfg.OUTPUT_DIR, f"{prefix}_top_confusion_pairs.json")

    with open(confusion_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)
        for idx, name in enumerate(class_names):
            writer.writerow([name] + matrix[idx].tolist())

    with open(hardest_json, "w") as f:
        json.dump(hardest_classes, f, indent=2, ensure_ascii=False)

    with open(confusions_json, "w") as f:
        json.dump(top_confusion_pairs, f, indent=2, ensure_ascii=False)

    logger.info("Saved confusion matrix to %s", confusion_csv)
    logger.info("Saved hardest classes to %s", hardest_json)
    logger.info("Saved top confusion pairs to %s", confusions_json)

    for item in hardest_classes:
        logger.info("class: %s, acc: %s", item["class_index"], item["accuracy"])



@torch.no_grad()
def test_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
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
    class_names = _get_eval_class_names(cfg)
    confusion_matrix = None
    if class_names:
        device = torch.device("cuda") if misc.get_num_gpus(cfg) else torch.device("cpu")
        confusion_matrix = torch.zeros(
            (len(class_names), len(class_names)), dtype=torch.float32, device=device
        )

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
            
            model_dict = model(task_dict)
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
                _top_max_k_vals, top_max_k_inds = torch.topk(
                                                        preds, 1, dim=1, largest=True, sorted=True
                                                    )
                if (
                    confusion_matrix is not None
                    and "real_target_labels" in task_dict
                    and "batch_class_list" in task_dict
                ):
                    pred_local = top_max_k_inds.squeeze(-1).long()
                    true_real, pred_real = _get_confusion_targets(task_dict, pred_local)
                    valid = (
                        (true_real >= 0)
                        & (true_real < confusion_matrix.shape[0])
                        & (pred_real >= 0)
                        & (pred_real < confusion_matrix.shape[1])
                    )
                    if valid.any():
                        ones = torch.ones_like(true_real[valid], dtype=confusion_matrix.dtype)
                        confusion_matrix.index_put_(
                            (true_real[valid], pred_real[valid]),
                            ones,
                            accumulate=True,
                        )


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

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    if confusion_matrix is not None and misc.get_num_gpus(cfg) > 1:
        du.all_reduce([confusion_matrix], average=False)
    _save_confusion_artifacts(cfg, confusion_matrix, class_names)
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

def test_few_shot(cfg):
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
    logging.setup_logging(cfg, cfg.TEST.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("TEST with config:")
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

    cu.load_test_checkpoint(cfg, model, model_ema, model_bucket)

    # Create the video train and val loaders.
    val_loader = build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None  # val

    # Create meters.
    if cfg.DETECTION.ENABLE:
        # train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        # train_meter = TrainMeter(len(train_loader), cfg)
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
        # writer = tb.TensorboardWriter(cfg)
        pass
    else:
        writer = None

    cur_epoch = 0
    test_epoch(
        val_loader, model, val_meter, cur_epoch,  cfg, writer
    )

    if writer is not None:
        writer.close()
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )
