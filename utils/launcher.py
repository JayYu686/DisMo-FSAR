#!/usr/bin/env python3

""" Task launcher. """

import os
import socket
from urllib.parse import urlparse

import torch

from utils.misc import get_num_gpus


_DEFAULT_LOCAL_INIT_METHOD = "tcp://localhost:9999"


def _reconcile_num_gpus(cfg):
    """Cap requested NUM_GPUS to currently visible CUDA devices."""
    requested = int(cfg.NUM_GPUS)
    if requested < 0:
        raise ValueError("NUM_GPUS must be >= 0")

    if cfg.PAI:
        # PAI setup is handled by environment variables in `run`.
        return

    available = torch.cuda.device_count()
    if requested > 0 and available == 0:
        print(
            "[Launcher] NUM_GPUS={} but no CUDA device is visible. "
            "Falling back to CPU mode (NUM_GPUS=0).".format(requested)
        )
        cfg.NUM_GPUS = 0
        return
    if requested > available:
        print(
            "[Launcher] Requested NUM_GPUS={}, but only {} CUDA device(s) are visible. "
            "Falling back to {}.".format(requested, available, available)
        )
        cfg.NUM_GPUS = available


def _can_bind_local_port(host, port):
    """Return True when the requested local TCP address is currently free."""
    try:
        addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    for family, socktype, proto, _, sockaddr in addrinfos:
        sock = socket.socket(family, socktype, proto)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            return True
        except OSError:
            continue
        finally:
            sock.close()
    return False


def _find_free_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _resolve_init_method(init_method):
    """
    Keep explicit rendezvous settings untouched.
    Only when the default localhost:9999 endpoint is occupied do we switch to
    another free local port automatically.
    """
    if not isinstance(init_method, str) or not init_method.startswith("tcp://"):
        return init_method

    parsed = urlparse(init_method)
    host = parsed.hostname
    port = parsed.port
    if host is None or port is None:
        return init_method

    normalized_host = host.lower()
    is_default_local = (
        port == 9999 and normalized_host in {"localhost", "127.0.0.1"}
    )
    if not is_default_local:
        return init_method

    bind_host = "127.0.0.1" if normalized_host == "localhost" else host
    if _can_bind_local_port(bind_host, port):
        return init_method

    fallback_port = _find_free_local_port()
    resolved = f"tcp://{host}:{fallback_port}"
    print(
        "[Launcher] Default init_method {} is busy. Falling back to {}.".format(
            init_method, resolved
        )
    )
    return resolved


def launch_task(cfg, init_method, func):
    """
    Launches the task "func" on one or multiple devices.
    Args:
        cfg (Config): global config object.
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): task to run.
    """
    _reconcile_num_gpus(cfg)
    torch.cuda.empty_cache()
    init_method = _resolve_init_method(init_method)
    if get_num_gpus(cfg) > 1:
        if cfg.PAI:
            # if using the PAI cluster, get info from the environment
            cfg.SHARD_ID = int(os.environ['RANK'])
            if "VISIBLE_DEVICE_LIST" in os.environ:
                cfg.NUM_GPUS = len(os.environ["VISIBLE_DEVICE_LIST"].split(","))
            else:
                cfg.NUM_GPUS = torch.cuda.device_count()
            cfg.NUM_SHARDS = int(os.environ['WORLD_SIZE'])

        torch.multiprocessing.spawn(
            run,
            nprocs=cfg.NUM_GPUS,
            args=(func, init_method, cfg),
            daemon=False,
        )
    else:
        func(cfg=cfg)


def run(
    local_rank, func, init_method, cfg
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
        cfg (Config): global config object.
    """

    num_proc    = cfg.NUM_GPUS      # number of nodes per machine
    shard_id    = cfg.SHARD_ID
    num_shards  = cfg.NUM_SHARDS    # number of machines
    backend     = cfg.DIST_BACKEND  # distribued backends ('nccl', 'gloo' or 'mpi')

    world_size  = num_proc * num_shards
    rank        = shard_id * num_proc + local_rank
    cfg.LOCAL_RANK = local_rank
    cfg.RANK = rank

    # dump machine info
    print("num_proc (NUM_GPU): {}".format(num_proc))
    print("shard_id (os.environ['RANK']): {}".format(shard_id))
    print("num_shards (os.environ['WORLD_SIZE']): {}".format(num_shards))
    print("rank: {}".format(rank))
    print("local_rank (GPU_ID): {}".format(local_rank))

    try:
        if cfg.PAI == False:
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )
        else:
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
            )
    except Exception as e:
        raise e

    if "VISIBLE_DEVICE_LIST" in os.environ:
        torch.cuda.set_device(int(os.environ["VISIBLE_DEVICE_LIST"]))
    else:
        torch.cuda.set_device(f'cuda:{local_rank}')
    func(cfg)
