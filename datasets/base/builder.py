#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Builder for the dataloader."""

import itertools
import multiprocessing as mp
import numpy as np
import torch
import utils.misc as misc
import utils.logging as logging
from utils.sampler import MultiFoldDistributedSampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from utils.val_dist_sampler import MultiSegValDistributedSampler
from datasets.utils.collate_functions import COLLATE_FN_REGISTRY


from utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
logger = logging.get_logger(__name__)


def _resolve_num_workers(requested_workers):
    """Downgrade workers when the runtime cannot create multiprocessing locks."""
    if requested_workers <= 0:
        return 0
    try:
        test_lock = mp.get_context("spawn").Lock()
        del test_lock
        return requested_workers
    except Exception as exc:
        logger.warning(
            "Multiprocessing workers unavailable (%s). Falling back to num_workers=0.",
            exc,
        )
        return 0


def _resolve_batch_size(global_batch_size, num_gpus, split_name, cfg_key):
    """
    Compute per-process batch size and keep it >= 1.
    """
    if global_batch_size <= 0:
        raise ValueError(
            f"{cfg_key} should be a positive integer, but got {global_batch_size}"
        )

    denom = max(1, int(num_gpus))
    per_process_batch = int(global_batch_size / denom)
    if per_process_batch <= 0:
        logger.warning(
            "%s (%d) is smaller than NUM_GPUS (%d) for split '%s'. "
            "Using per-process batch_size=1. "
            "Consider increasing %s or reducing NUM_GPUS.",
            cfg_key,
            global_batch_size,
            num_gpus,
            split_name,
            cfg_key,
        )
        return 1
    return per_process_batch

def get_sampler(cfg, dataset, split, shuffle):
    """
        Returns the sampler object for the dataset.
        Args:
            dataset (Dataset): constructed dataset. 
            split   (str):     which split is the dataset for.
            shuffle (bool):    whether or not to shuffle the dataset.
        Returns:
            sampler (Sampler): dataset sampler. 
    """
    if misc.get_num_gpus(cfg) > 1:
        if split == "train" and cfg.TRAIN.NUM_FOLDS > 1:
            return MultiFoldDistributedSampler(
                dataset, cfg.TRAIN.NUM_FOLDS
            )
        elif cfg.USE_MULTISEG_VAL_DIST and cfg.TRAIN.ENABLE is False:
            return MultiSegValDistributedSampler(dataset, shuffle=False)
        else:
            return DistributedSampler(
                dataset,
                shuffle=shuffle
            )
    else:
        return None

def build_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (Configs): global config object. details in utils/config.py
        split (str): the split of the data loader. Options include `train`,
            `val`, `test`, and `submission`.
    Returns:
        loader object. 
    """
    assert split in ["train", "val", "test", "submission"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = _resolve_batch_size(
            cfg.TRAIN.BATCH_SIZE, cfg.NUM_GPUS, split_name="train", cfg_key="TRAIN.BATCH_SIZE"
        )
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = _resolve_batch_size(
            cfg.TEST.BATCH_SIZE, cfg.NUM_GPUS, split_name="val", cfg_key="TEST.BATCH_SIZE"
        )
        shuffle = False
        drop_last = False
    elif split in ["test", "submission"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = _resolve_batch_size(
            cfg.TEST.BATCH_SIZE, cfg.NUM_GPUS, split_name=split, cfg_key="TEST.BATCH_SIZE"
        )
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    # Create a sampler for multi-process training
    sampler = get_sampler(cfg, dataset, split, shuffle)
    # Create a loader
    if hasattr(cfg.DATA_LOADER, "COLLATE_FN") and cfg.DATA_LOADER.COLLATE_FN is not None:
        collate_fn = COLLATE_FN_REGISTRY.get(cfg.DATA_LOADER.COLLATE_FN)(cfg)
    else:
        collate_fn = None
    resolved_workers = _resolve_num_workers(cfg.DATA_LOADER.NUM_WORKERS)
    try:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=resolved_workers,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=collate_fn
        )
    except PermissionError as exc:
        if resolved_workers > 0:
            logger.warning(
                "DataLoader worker init failed (%s). Falling back to num_workers=0.", exc
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(False if sampler else shuffle),
                sampler=sampler,
                num_workers=0,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=drop_last,
                collate_fn=collate_fn
            )
        else:
            raise
    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the sampler for the dataset.
    Args:
        loader      (loader):   data loader to perform shuffle.
        cur_epoch   (int):      number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler, MultiFoldDistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, (DistributedSampler, MultiFoldDistributedSampler)):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)

def build_dataset(dataset_name, cfg, split):
    """
    Builds a dataset according to the "dataset_name".
    Args:
        dataset_name (str):     the name of the dataset to be constructed.
        cfg          (Config):  global config object. 
        split        (str):     the split of the data loader.
    Returns:
        Dataset      (Dataset):    a dataset object constructed for the specified dataset_name.
    """
    name = dataset_name.capitalize()
    return DATASET_REGISTRY.get(name)(cfg, split)
