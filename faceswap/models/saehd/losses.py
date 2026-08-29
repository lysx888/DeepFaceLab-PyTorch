import collections
import numpy as np
import torch
import torch.nn.functional as F


_gaussian_kernel_cache: collections.OrderedDict = collections.OrderedDict()
_GAUSSIAN_CACHE_MAX = 8


def _get_gaussian_kernel(size: int, sigma: float,
                         dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    x = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    kernel = k.unsqueeze(0) * k.unsqueeze(1)
    return kernel


def gaussian_blur(x: torch.Tensor, radius: float) -> torch.Tensor:
    kernel_size = max(3, int(2 * 2 * radius))
    if kernel_size % 2 == 0:
        kernel_size += 1

    key = (kernel_size, radius, str(x.device), str(x.dtype))
    if key in _gaussian_kernel_cache:
        kernel = _gaussian_kernel_cache[key]
    else:
        kernel = _get_gaussian_kernel(kernel_size, radius, x.dtype, x.device)
        if len(_gaussian_kernel_cache) >= _GAUSSIAN_CACHE_MAX:
            _gaussian_kernel_cache.popitem(last=False)
        _gaussian_kernel_cache[key] = kernel

    pad = kernel_size // 2
    ch = x.shape[1]
    weight = kernel.view(1, 1, kernel_size, kernel_size).expand(ch, 1, kernel_size, kernel_size)
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.conv2d(x, weight, groups=ch)


_dssim_kernel_cache: collections.OrderedDict = collections.OrderedDict()
_DSSIM_CACHE_MAX = 8


def _get_dssim_kernel(filter_size: int, sigma: float, channels: int,
                      dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    x = torch.arange(filter_size, dtype=dtype, device=device) - (filter_size - 1) / 2.0
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    kernel = k.unsqueeze(0) * k.unsqueeze(1)
    return kernel.view(1, 1, filter_size, filter_size).expand(channels, 1, filter_size, filter_size)


def dssim(img1: torch.Tensor, img2: torch.Tensor,
          max_val: float = 1.0, filter_size: int = 11,
          filter_sigma: float = 1.5) -> torch.Tensor:
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    ch = img1.shape[1]
    key = (filter_size, filter_sigma, ch, str(img1.device), str(img1.dtype))
    if key in _dssim_kernel_cache:
        kernel = _dssim_kernel_cache[key]
    else:
        kernel = _get_dssim_kernel(filter_size, filter_sigma, ch, img1.dtype, img1.device)
        if len(_dssim_kernel_cache) >= _DSSIM_CACHE_MAX:
            _dssim_kernel_cache.popitem(last=False)
        _dssim_kernel_cache[key] = kernel

    pad = filter_size // 2

    def reducer(x):
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        return F.conv2d(x, kernel, groups=ch)

    mean0 = reducer(img1)
    mean1 = reducer(img2)
    num0 = mean0 * mean1 * 2.0
    den0 = mean0 ** 2 + mean1 ** 2
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(img1 * img2) * 2.0
    den1 = reducer(img1 ** 2 + img2 ** 2)
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    ssim_val = (luminance * cs).mean(dim=[2, 3])
    return (1.0 - ssim_val) / 2.0


def style_loss(target: torch.Tensor, style: torch.Tensor,
               gaussian_blur_radius: float = 0.0,
               loss_weight: float = 1.0) -> torch.Tensor:
    if gaussian_blur_radius > 0.0:
        target = gaussian_blur(target, gaussian_blur_radius)
        style = gaussian_blur(style, gaussian_blur_radius)

    ch = target.shape[1]
    c_mean = target.mean(dim=[2, 3], keepdim=True)
    s_mean = style.mean(dim=[2, 3], keepdim=True)
    c_var = target.var(dim=[2, 3], keepdim=True, unbiased=False)
    s_var = style.var(dim=[2, 3], keepdim=True, unbiased=False)
    c_std = torch.sqrt(c_var + 1e-5)
    s_std = torch.sqrt(s_var + 1e-5)

    mean_loss = ((c_mean - s_mean) ** 2).sum(dim=[1, 2, 3])
    std_loss = ((c_std - s_std) ** 2).sum(dim=[1, 2, 3])
    return (mean_loss + std_loss) * (loss_weight / ch)


def total_variation_mse(images: torch.Tensor) -> torch.Tensor:
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]
    return (pixel_dif1 ** 2).sum(dim=[1, 2, 3]) + (pixel_dif2 ** 2).sum(dim=[1, 2, 3])


def dloss(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def apply_blur_out_mask(target: torch.Tensor, mask: torch.Tensor,
                        res: int) -> torch.Tensor:
    sigma = res / 128
    msk_anti = 1 - mask
    x = gaussian_blur(target * msk_anti, sigma)
    y = 1 - gaussian_blur(mask, sigma)
    y = torch.where(y == 0, torch.ones_like(y), y)
    return target * mask + (x / y) * msk_anti


def blur_mask(mask: torch.Tensor, blur_sigma: float) -> torch.Tensor:
    return torch.clamp(gaussian_blur(mask, blur_sigma), 0, 0.5) * 2


def dssim_filter_sizes(res: int) -> tuple[int, int]:
    return max(1, int(res / 11.6)), max(1, int(res / 23.2))


def gan_discriminator_loss(D, pred_masked: torch.Tensor,
                           target_masked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_d, pred_d2 = D(pred_masked)
    target_d, target_d2 = D(target_masked)
    ones_p = torch.ones_like(pred_d)
    zeros_p = torch.zeros_like(pred_d)
    g_loss = dloss(ones_p, pred_d) + dloss(torch.ones_like(pred_d2), pred_d2)

    pred_d_d, pred_d2_d = D(pred_masked.detach())
    d_loss = (dloss(torch.ones_like(target_d), target_d) + dloss(zeros_p, pred_d_d)) * 0.5
    d_loss = d_loss + (dloss(torch.ones_like(target_d2), target_d2) + dloss(torch.zeros_like(pred_d2), pred_d2_d)) * 0.5
    return g_loss, d_loss


def gan_discriminator_loss_dual(D, pred_src: torch.Tensor, target_src: torch.Tensor,
                                pred_dst: torch.Tensor, target_dst: torch.Tensor
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    ps_d, ps_d2 = D(pred_src)
    pd_d, pd_d2 = D(pred_dst)
    ones = torch.ones_like(ps_d)
    g_loss = (dloss(ones, ps_d) + dloss(torch.ones_like(ps_d2), ps_d2)
              + dloss(ones, pd_d) + dloss(torch.ones_like(pd_d2), pd_d2))

    ts_d, ts_d2 = D(target_src)
    td_d, td_d2 = D(target_dst)
    ps_d_d, ps_d2_d = D(pred_src.detach())
    pd_d_d, pd_d2_d = D(pred_dst.detach())
    zeros = torch.zeros_like(ps_d_d)
    d_loss = (dloss(torch.ones_like(ts_d), ts_d) + dloss(zeros, ps_d_d)
              + dloss(torch.ones_like(td_d), td_d) + dloss(torch.zeros_like(pd_d_d), pd_d_d)
              + dloss(torch.ones_like(ts_d2), ts_d2) + dloss(torch.zeros_like(ps_d2_d), ps_d2_d)
              + dloss(torch.ones_like(td_d2), td_d2) + dloss(torch.zeros_like(pd_d2_d), pd_d2_d)) * (1.0 / 8)
    return g_loss, d_loss
