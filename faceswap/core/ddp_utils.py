import os
import socket
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from faceswap.shared.logger import get_logger

_logger = get_logger("ddp_utils")


class WeightedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, weights=None, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        self._weights = weights

    def __iter__(self):
        indices = list(super().__iter__())
        if self._weights is not None and len(self._weights) == len(self.dataset):
            ws = [self._weights[i] for i in range(len(self.dataset))]
            total = sum(ws)
            if total > 0:
                ws = [w / total for w in ws]
            indices = list(torch.multinomial(
                torch.tensor(ws, dtype=torch.float64),
                len(indices), replacement=True).numpy())
        return iter(indices)


def is_ddp_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 1


def setup_process_group(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(find_free_port()))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    _logger.info(f"[DDP] Process group initialized: rank={rank}, world_size={world_size}, backend={backend}")


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wrap_model_ddp(model: torch.nn.Module, rank: int, **kwargs) -> torch.nn.Module:
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank, **kwargs
    )


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def ddp_spawn(
    train_fn: Callable,
    world_size: int,
    *args,
    **kwargs,
) -> None:
    _logger.info(f"[DDP] Spawning {world_size} processes")
    mp.spawn(
        _ddp_entry,
        args=(train_fn, world_size, args, kwargs),
        nprocs=world_size,
        join=True,
    )


def _ddp_entry(rank: int, train_fn: Callable, world_size: int, args, kwargs) -> None:
    setup_process_group(rank, world_size)
    try:
        train_fn(rank, world_size, *args, **kwargs)
    finally:
        cleanup_process_group()
