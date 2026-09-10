import torch
import numpy as np

from faceswap.shared.logger import get_logger

_logger = get_logger("hardware_utils")


def get_gpu_memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory / 1024**3
    return 0.0


def get_available_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        return 8.0


def probe_batch_size(
    net: torch.nn.Module,
    input_size: int,
    device: torch.device,
    in_channels: int = 3,
    max_cap: int = 256,
    safety_gb: float = 1.5,
) -> int:
    if device.type != "cuda":
        return 32
    total_gb = get_gpu_memory_gb(device)
    net = net.to(device)
    was_training = net.training
    net.eval()
    opt = torch.optim.SGD(net.parameters(), lr=1e-8)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        bs = 2
        x = torch.randn(bs, in_channels, input_size, input_size, device=device)
        pred = net(x)
        if isinstance(pred, dict):
            for v in pred.values():
                if isinstance(v, torch.Tensor):
                    pred = v
                    break
        loss = pred.float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize(device)
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        per_sample_gb = peak_gb / bs
    except torch.cuda.OutOfMemoryError:
        per_sample_gb = 1.0
        _logger.warning("batch=2 OOM during probe, falling back to batch=1")
    finally:
        if was_training:
            net.train()
        torch.cuda.empty_cache()
        del opt

    if per_sample_gb <= 0:
        return 32
    usable_gb = max(total_gb - safety_gb, 0.5)
    suggested = int(usable_gb / per_sample_gb)
    result = max(1, min(max_cap, suggested))
    _logger.info(f"GPU {total_gb:.1f}GB, per_sample={per_sample_gb:.4f}GB, suggested batch={result}")
    return result


def cache_budget_bytes(
    avail_gb: float,
    workers: int = 1,
    ratio: float = 0.05,
    cap_gb: float = 1.0,
) -> int:
    budget_gb = min(avail_gb * ratio, cap_gb)
    return int(budget_gb * 1024**3 / max(workers, 1))


def image_cache_size(
    avail_gb: float,
    bytes_per_image: int = 110_000,
    max_cap: int = 4096,
    min_cap: int = 32,
    workers: int = 1,
) -> int:
    budget_bytes = cache_budget_bytes(avail_gb)
    per_worker = budget_bytes // max(workers, 1)
    return max(min_cap, min(max_cap, per_worker // max(bytes_per_image, 1)))


def assert_no_oom(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA out of memory. 训练已停止。\n"
            "请减小 batch_size 或降低分辨率后重试。"
        )
