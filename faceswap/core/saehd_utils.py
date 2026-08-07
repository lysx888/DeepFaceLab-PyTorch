import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from faceswap.shared.logger import get_logger

_logger = get_logger("saehd_utils")


def apply_torch_compile(model, enable: bool = True) -> None:
    if not enable:
        return
    try:
        major, minor = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
        if major < 2:
            _logger.info("torch.compile skipped: PyTorch < 2.0")
            return
    except Exception:
        _logger.info("torch.compile skipped: cannot determine PyTorch version")
        return
    modules_dict = getattr(model, '_modules_dict', {})
    if not modules_dict:
        _logger.info("torch.compile skipped: no sub-modules found")
        return
    compiled_count = 0
    for name, module in list(modules_dict.items()):
        if not isinstance(module, nn.Module):
            continue
        try:
            compiled_mod = torch.compile(module, backend="aot_eager", dynamic=False)
            setattr(model, name, compiled_mod)
            modules_dict[name] = compiled_mod
            compiled_count += 1
        except Exception as e:
            _logger.warning(f"torch.compile failed for {name} ({e}), keeping eager")
    if compiled_count > 0:
        _logger.info(f"torch.compile enabled: {compiled_count} sub-modules (aot_eager)")


def compute_effective_gan_power(iter_count: int, target_iter: int,
                                gan_power: float, ramp_start_ratio: float,
                                pretrain: bool = False) -> float:
    if gan_power == 0.0 or pretrain:
        return 0.0
    if target_iter <= 0:
        return gan_power
    ramp_start = int(target_iter * ramp_start_ratio)
    if iter_count < ramp_start:
        return 0.0
    if iter_count >= target_iter:
        return gan_power
    progress = (iter_count - ramp_start) / max(1, target_iter - ramp_start)
    smooth = 1.0 / (1.0 + math.exp(-(progress - 0.5) * 10.0))
    return gan_power * smooth


def adaptive_dilate_mask(mask: torch.Tensor, sigma: float = 2.0,
                         radius: int = 3) -> torch.Tensor:
    blurred = gaussian_blur(mask, sigma)
    padded = F.pad(blurred, [radius, radius, radius, radius], mode='reflect')
    dilated = F.max_pool2d(padded, kernel_size=2 * radius + 1, stride=1, padding=0)
    return dilated.clamp(0, 1)


_gaussian_kernel_cache: dict[tuple, torch.Tensor] = {}


def gaussian_blur(image: torch.Tensor, sigma) -> torch.Tensor:
    if isinstance(sigma, int) or (isinstance(sigma, float) and sigma == int(sigma) and sigma > 2):
        kernel_size = int(sigma)
        kernel_size = max(3, kernel_size + (1 - kernel_size % 2))
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    else:
        sigma = float(sigma)
        if sigma <= 0:
            return image
        kernel_size = round(sigma * 6.0)
        kernel_size = max(3, kernel_size + (1 - kernel_size % 2))

    C = image.shape[1]
    cache_key = (kernel_size, round(sigma * 100), C)
    kernel_2d_cpu = _gaussian_kernel_cache.get(cache_key)
    if kernel_2d_cpu is None:
        x = torch.arange(kernel_size, dtype=torch.float32, device='cpu') - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        kernel_2d_cpu = kernel_2d.expand(C, 1, kernel_size, kernel_size).contiguous()
        _gaussian_kernel_cache[cache_key] = kernel_2d_cpu
    kernel_2d = kernel_2d_cpu.to(image.device)

    pad = kernel_size // 2
    out = F.conv2d(image, kernel_2d, padding=pad, groups=C)
    return out


def depth_to_space(x: torch.Tensor, block_size: int = 2) -> torch.Tensor:
    return F.pixel_shuffle(x, block_size)


def pixel_norm(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)


def total_variation_mse(image: torch.Tensor) -> torch.Tensor:
    diff_h = image[:, :, 1:, :] - image[:, :, :-1, :]
    diff_w = image[:, :, :, 1:] - image[:, :, :, :-1]
    return (diff_h ** 2).mean() + (diff_w ** 2).mean()


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def reshape_4d(x: torch.Tensor, h: int, w: int, c: int) -> torch.Tensor:
    return x.reshape(x.shape[0], c, h, w)
