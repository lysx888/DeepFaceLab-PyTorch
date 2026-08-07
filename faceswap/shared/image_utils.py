from typing import Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
from pathlib import Path

from faceswap.shared.logger import get_logger

_logger = get_logger("image_utils")

_KORNIA_AVAILABLE = False
try:
    import kornia
    _KORNIA_AVAILABLE = True
except ImportError:
    _logger.warning("kornia not available, falling back to OpenCV/PyTorch for image operations.")


class ImageUtils:

    @staticmethod
    def warp_affine_gpu(
        image: "torch.Tensor",
        M: "torch.Tensor",
        dsize: tuple[int, int],
    ) -> "torch.Tensor":
        import torch
        if _KORNIA_AVAILABLE and image.is_cuda:
            import kornia.geometry.transform as kt
            if M.dim() == 2:
                M = M.unsqueeze(0)
            return kt.warp_affine(image.unsqueeze(0) if image.dim() == 3 else image, M, dsize).squeeze(0)
        import torch.nn.functional as F
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if M.dim() == 2:
            M = M.unsqueeze(0)
        grid = F.affine_grid(M[:, :2, :], image.shape[:2] + dsize, align_corners=False)
        return F.grid_sample(image, grid, align_corners=False, mode="bilinear", padding_mode="zeros").squeeze(0)

    @staticmethod
    def augment_gpu(
        image: "torch.Tensor",
        rotation: float = 0.0,
        scale: float = 1.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        hue: float = 0.0,
        saturation: float = 0.0,
    ) -> "torch.Tensor":
        import torch
        if _KORNIA_AVAILABLE and image.is_cuda:
            import kornia.augmentation as ka
            aug_list = ka.AugmentationSequential(
                ka.RandomAffine(degrees=rotation, scale=(scale, scale), p=1.0) if rotation != 0.0 or scale != 1.0 else None,
                ka.ColorJiggle(brightness=brightness, contrast=contrast, hue=hue, saturation=saturation, p=1.0) if any(v != 0.0 for v in [brightness, contrast, hue, saturation]) else None,
            )
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return aug_list(image).squeeze(0)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if rotation != 0.0 or scale != 1.0:
            import torch.nn.functional as F
            angle_rad = rotation * 3.14159265 / 180.0
            cos_a = torch.cos(torch.tensor(angle_rad, device=image.device)) * scale
            sin_a = torch.sin(torch.tensor(angle_rad, device=image.device)) * scale
            theta = torch.zeros(1, 2, 3, device=image.device)
            theta[0, 0, 0] = cos_a
            theta[0, 0, 1] = sin_a
            theta[0, 1, 0] = -sin_a
            theta[0, 1, 1] = cos_a
            grid = F.affine_grid(theta, image.shape, align_corners=False)
            image = F.grid_sample(image, grid, align_corners=False, mode="bilinear", padding_mode="zeros")
        if brightness != 0.0:
            image = image + brightness
        if contrast != 0.0:
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = (1.0 + contrast) * (image - mean) + mean
        return image.squeeze(0)

    @staticmethod
    def gaussian_blur_gpu(
        image: "torch.Tensor",
        kernel_size: int = 5,
        sigma: float = 1.0,
    ) -> "torch.Tensor":
        import torch
        if _KORNIA_AVAILABLE and image.is_cuda:
            import kornia.filters as kf
            return kf.gaussian_blur2d(image.unsqueeze(0) if image.dim() == 3 else image, (kernel_size, kernel_size), (sigma, sigma)).squeeze(0)
        import torch.nn.functional as F
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        k = kernel_size
        x = torch.arange(k, device=image.device) - k // 2
        gauss_1d = torch.exp(-x.float() ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        c = image.shape[1]
        kernel = kernel.expand(c, 1, k, k)
        pad = k // 2
        result = F.conv2d(image, kernel, padding=pad, groups=c)
        return result.squeeze(0)

    @staticmethod
    def resize_gpu(
        image: "torch.Tensor",
        size: tuple[int, int],
        interpolation: str = "bilinear",
    ) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F
        mode_map = {"bilinear": "bilinear", "nearest": "nearest", "bicubic": "bicubic"}
        mode = mode_map.get(interpolation, "bilinear")
        if image.dim() == 3:
            image = image.unsqueeze(0)
            return F.interpolate(image, size=size, mode=mode, align_corners=False if mode != "nearest" else None).squeeze(0)
        return F.interpolate(image, size=size, mode=mode, align_corners=False if mode != "nearest" else None)

    @staticmethod
    def numpy_to_tensor(img: npt.NDArray[np.uint8], device: str = "auto") -> "torch.Tensor":
        import torch
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        if tensor.dim() == 3 and tensor.shape[2] == 3:
            tensor = tensor.permute(2, 0, 1)
        if device == "auto":
            from faceswap.shared.config import auto_select_device
            device = auto_select_device().type
        return tensor.to(device)

    @staticmethod
    def tensor_to_numpy(tensor: "torch.Tensor") -> npt.NDArray[np.uint8]:
        img = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        return (img * 255).astype(np.uint8)


def imread_bgr(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> Optional[npt.NDArray]:
    return cv2.imread(str(path), flags)


def imwrite_bgr(path: Union[str, Path], img: npt.NDArray, params=None) -> bool:
    if params is not None:
        return cv2.imwrite(str(path), img, params)
    return cv2.imwrite(str(path), img)


def bgr_to_rgb(img: npt.NDArray) -> npt.NDArray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: npt.NDArray) -> npt.NDArray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr_to_hsv(img: npt.NDArray) -> npt.NDArray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hsv_to_bgr(img: npt.NDArray) -> npt.NDArray:
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
