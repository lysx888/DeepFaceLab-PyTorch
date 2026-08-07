import collections
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from faceswap.core.saehd_utils import gaussian_blur


def _gaussian_kernel_2d(size: int, sigma: float,
                        dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    x = torch.arange(size, dtype=dtype, device=device) - size // 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k.unsqueeze(1) * k.unsqueeze(0)


_dssim_kernel_cache: collections.OrderedDict = collections.OrderedDict()
_DSSIM_CACHE_MAX = 8


def dssim(t1: torch.Tensor, t2: torch.Tensor,
           filter_size: int = 11, max_val: float = 1.0) -> torch.Tensor:
    t1 = t1.float()
    t2 = t2.float()
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    sigma = 1.5
    C = t1.shape[1]
    cache_key = (filter_size, sigma, C)
    kernel_cpu = _dssim_kernel_cache.get(cache_key)
    if kernel_cpu is None:
        k = _gaussian_kernel_2d(filter_size, sigma, torch.float32, torch.device('cpu'))
        kernel_cpu = k.expand(C, 1, filter_size, filter_size).contiguous()
        if len(_dssim_kernel_cache) >= _DSSIM_CACHE_MAX:
            _dssim_kernel_cache.popitem(last=False)
        _dssim_kernel_cache[cache_key] = kernel_cpu
    else:
        _dssim_kernel_cache.move_to_end(cache_key)
    kernel = kernel_cpu.to(t1.device)
    pad = filter_size // 2

    mu1 = F.conv2d(t1, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(t2, kernel, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(t1 ** 2, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(t2 ** 2, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(t1 * t2, kernel, padding=pad, groups=C) - mu1_mu2

    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    result = (1.0 - ssim_map) / 2.0
    return result


def ms_ssim(t1: torch.Tensor, t2: torch.Tensor,
            filter_size: int = 11, max_val: float = 1.0,
            weights: tuple[float, ...] = (0.5, 0.3, 0.2)) -> torch.Tensor:
    t1 = t1.float()
    t2 = t2.float()
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    sigma = 1.5
    C = t1.shape[1]

    cache_key = (filter_size, sigma, C)
    kernel_cpu = _dssim_kernel_cache.get(cache_key)
    if kernel_cpu is None:
        k = _gaussian_kernel_2d(filter_size, sigma, torch.float32, torch.device('cpu'))
        kernel_cpu = k.expand(C, 1, filter_size, filter_size).contiguous()
        if len(_dssim_kernel_cache) >= _DSSIM_CACHE_MAX:
            _dssim_kernel_cache.popitem(last=False)
        _dssim_kernel_cache[cache_key] = kernel_cpu
    else:
        _dssim_kernel_cache.move_to_end(cache_key)
    kernel = kernel_cpu.to(t1.device)
    pad = filter_size // 2

    valid_weights = []
    ms_ssim_val = 1.0

    for scale_idx, w in enumerate(weights):
        if t1.shape[2] < filter_size or t1.shape[3] < filter_size:
            break
        valid_weights.append(w)

        mu1 = F.conv2d(t1, kernel, padding=pad, groups=C)
        mu2 = F.conv2d(t2, kernel, padding=pad, groups=C)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = torch.clamp(F.conv2d(t1 ** 2, kernel, padding=pad, groups=C) - mu1_sq, min=0.0)
        sigma2_sq = torch.clamp(F.conv2d(t2 ** 2, kernel, padding=pad, groups=C) - mu2_sq, min=0.0)
        sigma12 = F.conv2d(t1 * t2, kernel, padding=pad, groups=C) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ms_ssim_val = ms_ssim_val * ssim_map.mean(dim=[1, 2, 3]).clamp(0, 1) ** w

        if scale_idx < len(weights) - 1:
            t1 = F.avg_pool2d(t1, 2)
            t2 = F.avg_pool2d(t2, 2)

    if len(valid_weights) < len(weights):
        w_sum = sum(valid_weights)
        if w_sum > 0:
            ms_ssim_val = ms_ssim_val ** (1.0 / w_sum) if w_sum != 1.0 else ms_ssim_val

    return (1.0 - ms_ssim_val) / 2.0


def style_loss(pred: torch.Tensor, target: torch.Tensor,
               blur_radius: float = 1.0,
               loss_weight: float = 1.0) -> torch.Tensor:
    if blur_radius > 0:
        pred = gaussian_blur(pred, blur_radius)
        target = gaussian_blur(target, blur_radius)

    def _gram(x):
        x = x.float()
        N, C, H, W = x.shape
        feat = x.reshape(N, C, H * W)
        G = torch.bmm(feat, feat.transpose(1, 2))
        return G / (C * C * H * W)

    G_pred = _gram(pred)
    G_target = _gram(target).detach()
    return loss_weight * F.mse_loss(G_pred, G_target)


class VGGFeatureExtractor(nn.Module):
    _FACE_MEAN = [0.440, 0.369, 0.351]
    _FACE_STD = [0.273, 0.245, 0.250]
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, layer_ids: tuple[int, ...] = (3, 8, 13, 19),
                 norm_mode: str = 'face'):
        super().__init__()
        from torchvision import models as tv_models
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self._layer_ids = sorted(layer_ids)
        self._norm_mode = norm_mode
        if norm_mode == 'imagenet':
            m, s = self._IMAGENET_MEAN, self._IMAGENET_STD
        elif norm_mode == 'face':
            m, s = self._FACE_MEAN, self._FACE_STD
        else:
            m, s = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.register_buffer('mean',
                             torch.tensor(m, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std',
                             torch.tensor(s, dtype=torch.float32).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.mean) / self.std
        results = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._layer_ids:
                results.append(x)
        return results
