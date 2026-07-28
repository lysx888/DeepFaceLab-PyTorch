"""
Loss functions (from original DFL Model.py):
  - DSSIM + MSE for reconstruction (resolution-dependent weights)
  - Mask MSE loss
  - Eyes/mouth priority loss
  - Face style loss (channel statistics matching)
  - Background style loss (DSSIM + MSE outside mask)
  - Optional GAN loss

Preview columns (DFL style):
  S  | SS (src→src recon) | D  | DD (dst→dst recon) | SD (swap: src face on dst)
"""

import io
import itertools
import json
import math
import random
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, RandomSampler

from DeepFaceLab.models.saehd_model import SAEHDModel
from DeepFaceLab.models.saehd_dataset import SAEHDDataset
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger
from DeepFaceLab.shared.torch_config import get_dataloader_config, get_non_blocking, worker_init_fn


class _DFLCosineScheduler:
    """DFL-style oscillating cosine LR: lr * (cos(iter * 2π / lr_cos) + 1) / 2.

    Unlike PyTorch's CosineAnnealingLR (monotonic decay), this oscillates
    between 0 and lr_base with period lr_cos iterations — similar to SGDR.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, lr_cos: int, base_lrs: list[float]):
        self._optimizer = optimizer
        self._lr_cos = lr_cos
        self._base_lrs = base_lrs
        self._step_count = 0

    def step(self):
        factor = (math.cos(self._step_count * 2.0 * math.pi / self._lr_cos) + 1.0) / 2.0
        for param_group, base_lr in zip(self._optimizer.param_groups, self._base_lrs):
            param_group["lr"] = base_lr * factor
        self._step_count += 1

    @property
    def current_factor(self) -> float:
        return (math.cos(self._step_count * 2.0 * math.pi / self._lr_cos) + 1.0) / 2.0

_logger = get_logger("saehd_trainer")

_MODEL_PREFIX = "SAEHD"
_SAVE_INTERVAL_SEC = 1200


# ---------------------------------------------------------------------------
# Loss functions (exact DFL implementations)
# ---------------------------------------------------------------------------

def _dssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, filter_size: int = 11) -> torch.Tensor:
    """DFL's DSSIM: 1 - SSIM, using uniform averaging filter."""
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    padding = filter_size // 2
    channels = pred.shape[1]
    kernel = torch.ones(1, 1, filter_size, filter_size, device=pred.device, dtype=pred.dtype) / (filter_size * filter_size)
    kernel = kernel.expand(channels, 1, -1, -1)

    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred ** 2, kernel, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, kernel, padding=padding, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_cross

    sigma_pred_sq = torch.clamp(sigma_pred_sq, min=0.0)
    sigma_target_sq = torch.clamp(sigma_target_sq, min=0.0)

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    return 1.0 - ssim_map


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Simple Gaussian blur using separable 1D convolutions."""
    if sigma <= 0:
        return x
    kernel_size = max(3, int(sigma * 4) | 1)  # odd
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    # Horizontal
    k_h = g.reshape(1, 1, 1, kernel_size).expand(x.shape[1], 1, 1, kernel_size)
    x = F.conv2d(x, k_h, padding=(0, kernel_size // 2), groups=x.shape[1])
    # Vertical
    k_v = g.reshape(1, 1, kernel_size, 1).expand(x.shape[1], 1, kernel_size, 1)
    x = F.conv2d(x, k_v, padding=(kernel_size // 2, 0), groups=x.shape[1])
    return x


def _style_loss(pred: torch.Tensor, target: torch.Tensor, gaussian_blur_radius: int = 0, loss_weight: float = 1.0) -> torch.Tensor:
    """DFL style loss: channel statistics matching (mean + std), matching DFL's formula.

    DFL: sum((c_mean-s_mean)^2 + (c_std-s_std)^2) * loss_weight / content_nc
    """
    if gaussian_blur_radius > 0:
        pred = _gaussian_blur(pred, gaussian_blur_radius)
        target = _gaussian_blur(target, gaussian_blur_radius)

    content_nc = pred.shape[1]
    pred_mean = pred.mean(dim=[2, 3])
    target_mean = target.mean(dim=[2, 3])
    pred_var = pred.var(dim=[2, 3])
    target_var = target.var(dim=[2, 3])
    pred_std = torch.sqrt(pred_var + 1e-5)
    target_std = torch.sqrt(target_var + 1e-5)

    mean_loss = ((pred_mean - target_mean) ** 2).sum(dim=1)
    std_loss = ((pred_std - target_std) ** 2).sum(dim=1)
    loss = (mean_loss + std_loss) * (loss_weight / content_nc)
    return loss.mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SAEHDTrainer:
    """DFL SAEHD trainer — exact replica of original training logic."""

    def __init__(self, device: str = "auto") -> None:
        self._device_str = device
        self._device = None
        self._stop_event = threading.Event()
        self._preview_event = threading.Event()
        self._save_event = threading.Event()
        self._loss_history: list[tuple[int, float]] = []
        self._loss_history_range = 0
        self._iter_count = 0
        self._model_dir = None
        self._preview_src_paths: list[Path] = []
        self._preview_dst_paths: list[Path] = []
        self._preview_n = 3
        self._preview_resolution = 128
        self._preview_cache: dict = {}

    def _resolve_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        if self._device_str == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self._device_str)
        if self._device.type == "cpu":
            _logger.warning("No CUDA GPU detected, training on CPU will be very slow")
        return self._device

    def request_stop(self): self._stop_event.set()
    def request_preview(self): self._preview_event.set()
    def request_save(self): self._save_event.set()
    def cycle_loss_range(self): self._loss_history_range = (self._loss_history_range + 1) % 3

    def train(
        self,
        src_aligned_dir: Path,
        dst_aligned_dir: Path,
        model_dir: Path,
        resolution: int = 128,
        architecture: str = "df",
        ae_dims: int = 256,
        e_dims: int = 64,
        d_dims: int = 64,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        adabelief: bool = True,
        use_amp: bool = True,
        random_warp: bool = True,
        random_flip: bool = True,
        random_hsv_power: float = 0.0,
        color_transfer: str = "none",
        face_style_power: float = 0.0,
        bg_style_power: float = 0.0,
        eyes_mouth_prio: bool = False,
        masked_training: bool = True,
        gan_power: float = 0.0,
        gradient_clip: bool = False,
        lr_dropout: str = "n",
        gan_dims: int = 16,
        gan_patch_size: int = None,
        pretrain: bool = False,
        blur_out_mask: bool = False,
        true_face_power: float = 0.0,
        uniform_yaw: bool = False,
        save_interval_min: float = 15.0,
        preview_interval_sec: float = 60,
        on_iter: Optional[Callable] = None,
        on_preview: Optional[Callable] = None,
        on_save: Optional[Callable] = None,
        on_log: Optional[Callable] = None,
    ) -> None:
        self._stop_event.clear()
        self._preview_event.clear()
        self._save_event.clear()
        self._loss_history = []
        device = self._resolve_device()
        src_dir = Path(src_aligned_dir)
        dst_dir = Path(dst_aligned_dir)
        model_dir = Path(model_dir)

        # Pretrain mode: force override training parameters (DFL behavior)
        if pretrain:
            gan_power = 0.0
            random_warp = False
            random_flip = True
            random_hsv_power = 0.0
            face_style_power = 0.0
            bg_style_power = 0.0
            lr_dropout = "n"
            uniform_yaw = True
            _logger.info("Pretrain mode: GAN=0, warp=False, styles=0, lr_dropout=n, uniform_yaw=True")

        if not src_dir.exists():
            raise ValueError(f"Source aligned directory not found: {src_dir}")
        if not dst_dir.exists():
            raise ValueError(f"Target aligned directory not found: {dst_dir}")

        # DFL uses [0,1] images. TFMDataset outputs [-1,1], so we convert in the loop.
        # Disable random_warp's inner HSV by passing random_hsv_power=0 to dataset;
        # we handle HSV at the same level as DFL.
        src_ds = SAEHDDataset(src_dir, resolution=resolution, is_src=True, augment=True,
                             random_hsv_power=random_hsv_power, random_warp=random_warp,
                             random_flip=random_flip, color_transfer=color_transfer,
                             uniform_yaw=uniform_yaw)
        dst_ds = SAEHDDataset(dst_dir, resolution=resolution, is_src=False, augment=True,
                             random_hsv_power=0.0, random_warp=random_warp,
                             random_flip=random_flip, color_transfer=color_transfer,
                             uniform_yaw=uniform_yaw)

        if color_transfer != "none":
            ct_pool = dst_ds.build_ct_sample_pool(n=100)
            src_ds._ct_sample_pool = ct_pool

        dl_cfg = get_dataloader_config("gpu_train" if device.type == "cuda" else "cpu_train",
                                       dataset_size=len(src_ds) + len(dst_ds))

        src_loader = DataLoader(src_ds, batch_size=batch_size,
                                sampler=RandomSampler(src_ds, replacement=True, num_samples=max(len(src_ds), batch_size * 50)),
                                num_workers=dl_cfg["num_workers"], pin_memory=dl_cfg["pin_memory"],
                                drop_last=True, worker_init_fn=worker_init_fn if dl_cfg["num_workers"] > 0 else None,
                                persistent_workers=dl_cfg["num_workers"] > 0)
        dst_loader = DataLoader(dst_ds, batch_size=batch_size,
                                sampler=RandomSampler(dst_ds, replacement=True, num_samples=max(len(dst_ds), batch_size * 50)),
                                num_workers=dl_cfg["num_workers"], pin_memory=dl_cfg["pin_memory"],
                                drop_last=True, worker_init_fn=worker_init_fn if dl_cfg["num_workers"] > 0 else None,
                                persistent_workers=dl_cfg["num_workers"] > 0)

        # Create model
        d_mask_dims = d_dims // 3 + (d_dims // 3) % 2
        model = SAEHDModel(resolution=resolution, architecture=architecture,
                           ae_dims=ae_dims, e_dims=e_dims, d_dims=d_dims,
                           d_mask_dims=d_mask_dims, gan_dims=gan_dims,
                           gan_patch_size=gan_patch_size).to(device)

        # Build discriminator if GAN is enabled
        discriminator = None
        disc_optimizer = None
        if gan_power != 0:
            discriminator = model.build_discriminator().to(device)
            disc_optimizer = opt_class(discriminator.parameters(), lr=learning_rate)
            _logger.info(f"GAN enabled: power={gan_power}, dims={gan_dims}, patch_size={model.gan_patch_size}")

        # Build CodeDiscriminator if true_face_power enabled (df architecture only)
        code_discriminator = None
        code_disc_optimizer = None
        if true_face_power != 0 and not model.use_liae:
            code_res = model.inter.get_out_res()
            code_discriminator = model.build_code_discriminator(code_res).to(device)
            code_disc_optimizer = opt_class(code_discriminator.parameters(), lr=learning_rate)
            _logger.info(f"True face power enabled: {true_face_power}, code_res={code_res}")

        # Optimizer: AdaBelief (DFL default) or Adam
        if adabelief:
            from DeepFaceLab.shared.adabelief import AdaBelief
            opt_class = AdaBelief
            _logger.info("Using AdaBelief optimizer")
        else:
            opt_class = torch.optim.Adam
            _logger.info("Using Adam optimizer")

        optimizer = opt_class(model.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

        # Pretrain + LIAE: freeze inter_AB (DFL design: inter_AB learns cross-identity mapping)
        frozen_params: set[str] = set()
        if pretrain and model.use_liae and not random_warp:
            for name, param in model.inter_AB.named_parameters():
                full_name = f"inter_AB.{name}"
                param.requires_grad = False
                frozen_params.add(full_name)
            _logger.info(f"Pretrain LIAE: inter_AB frozen ({len(frozen_params)} params)")

        # LR Dropout: generate fixed Bernoulli masks (DFL-style, once at init)
        lr_dropout_masks: dict[str, torch.Tensor] = {}
        lr_dropout_p = 0.3 if lr_dropout in ("y", "cpu") else 1.0
        if lr_dropout_p < 1.0:
            mask_device = torch.device("cpu") if lr_dropout == "cpu" else device
            for name, param in model.named_parameters():
                if param.requires_grad:
                    mask = torch.bernoulli(torch.full_like(param.data, lr_dropout_p, device=mask_device))
                    lr_dropout_masks[name] = mask
            _logger.info(f"LR Dropout enabled (p={lr_dropout_p}, device={lr_dropout}), {len(lr_dropout_masks)} masks generated")

        # DFL cosine LR scheduler (oscillating, lr_cos=500 when lr_dropout enabled)
        lr_cos = 500 if lr_dropout in ("y", "cpu") else 0
        lr_scheduler: Optional[_DFLCosineScheduler] = None
        if lr_cos > 0:
            base_lrs = [pg["lr"] for pg in optimizer.param_groups]
            lr_scheduler = _DFLCosineScheduler(optimizer, lr_cos, base_lrs)
            _logger.info(f"DFL cosine LR scheduler enabled (lr_cos={lr_cos})")

        # Load checkpoint if exists
        start_iter = self._load_checkpoint(model, optimizer, model_dir,
                                          lr_dropout_masks=lr_dropout_masks,
                                          lr_scheduler=lr_scheduler,
                                          discriminator=discriminator,
                                          disc_optimizer=disc_optimizer)

        # Save training config
        model_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "resolution": resolution, "architecture": architecture,
            "ae_dims": ae_dims, "e_dims": e_dims, "d_dims": d_dims,
            "d_mask_dims": d_mask_dims, "batch_size": batch_size,
            "learning_rate": learning_rate, "use_amp": use_amp,
            "random_warp": random_warp, "face_style_power": face_style_power,
            "bg_style_power": bg_style_power, "eyes_mouth_prio": eyes_mouth_prio,
            "masked_training": masked_training, "gan_power": gan_power,
            "lr_dropout": lr_dropout, "lr_cos": lr_cos,
        }
        FileManager.atomic_write(model_dir / f"{_MODEL_PREFIX}_training_config.json", json.dumps(config, indent=2))

        # Print summary
        total_params = sum(p.numel() for p in model.parameters())
        device_info = ""
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = getattr(torch.cuda.get_device_properties(0), 'total_memory',
                              getattr(torch.cuda.get_device_properties(0), 'total_mem', 0)) / 1024 ** 3
            device_info = f"{gpu_name} ({gpu_mem:.1f}GB)"

        summary_lines = [
            f"== SAEHD Model Summary ({architecture}) ==",
            f"  Resolution: {resolution}",
            f"  AE dims: {ae_dims}, E dims: {e_dims}, D dims: {d_dims}, D mask dims: {d_mask_dims}",
            f"  Architecture: {architecture}",
            f"  Total params: {total_params:,}",
            f"  Batch size: {batch_size}",
            f"  Learning rate: {learning_rate}",
            f"  Device: {device_info or 'CPU'}",
            f"  Start iter: {start_iter}",
            f"===========================================",
        ]
        summary = "\n".join(summary_lines)
        print(summary)
        if on_log is not None:
            for line in summary_lines:
                on_log(line, False)

        # Preview setup
        self._preview_src_paths = FileManager.find_images(src_dir)
        self._preview_dst_paths = FileManager.find_images(dst_dir)
        self._preview_n = min(4, batch_size, 800 // resolution)
        self._preview_resolution = resolution

        # DFL loss parameters
        dssim_filter_size = int(resolution / 11.6)
        blur_sigma = resolution / 128.0
        mask_blur_sigma = max(1, resolution // 32)

        # Training loop
        lock_path = model_dir / ".training_lock"
        lock_path.write_text(str(time.time()))

        last_preview_time = 0.0
        last_save_time = time.time()
        iter_count = start_iter
        self._iter_count = iter_count

        src_iter = iter(src_loader)
        dst_iter = iter(dst_loader)

        model.train()

        for _ in itertools.count():
            if self._stop_event.is_set():
                break

            try:
                src_batch = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_batch = next(src_iter)
            try:
                dst_batch = next(dst_iter)
            except StopIteration:
                dst_iter = iter(dst_loader)
                dst_batch = next(dst_iter)

            t0 = time.time()

            # SAEHDDataset returns [0,1] tensors with warped/target dual samples
            src_warped = src_batch["warped"].to(device, non_blocking=get_non_blocking())
            src_target = src_batch["target"].to(device, non_blocking=get_non_blocking())
            src_mask = src_batch["mask"].to(device, non_blocking=get_non_blocking())
            src_em_mask = src_batch["em_mask"].to(device, non_blocking=get_non_blocking())
            dst_warped = dst_batch["warped"].to(device, non_blocking=get_non_blocking())
            dst_target = dst_batch["target"].to(device, non_blocking=get_non_blocking())
            dst_mask = dst_batch["mask"].to(device, non_blocking=get_non_blocking())
            dst_em_mask = dst_batch["em_mask"].to(device, non_blocking=get_non_blocking())

            # Blurred masks for loss weighting (DFL style)
            src_mask_blur = torch.clamp(_gaussian_blur(src_mask, mask_blur_sigma), 0, 0.5) * 2.0
            dst_mask_blur = torch.clamp(_gaussian_blur(dst_mask, mask_blur_sigma), 0, 0.5) * 2.0

            # Blur out mask: smooth background near mask edge (DFL style)
            if blur_out_mask:
                blur_sigma = resolution / 128.0
                src_mask_anti = 1.0 - src_mask
                x = _gaussian_blur(src_target * src_mask_anti, blur_sigma)
                y = 1.0 - _gaussian_blur(src_mask, blur_sigma)
                y = torch.where(y == 0, torch.ones_like(y), y)
                src_target = src_target * src_mask + (x / y) * src_mask_anti

                dst_mask_anti = 1.0 - dst_mask
                x = _gaussian_blur(dst_target * dst_mask_anti, blur_sigma)
                y = 1.0 - _gaussian_blur(dst_mask, blur_sigma)
                y = torch.where(y == 0, torch.ones_like(y), y)
                dst_target = dst_target * dst_mask + (x / y) * dst_mask_anti

            with torch.amp.autocast(device.type, enabled=(use_amp and device.type == "cuda")):
                out = model(src_warped, dst_warped)

                pred_src_src = out["pred_src_src"]
                pred_src_srcm = out["pred_src_srcm"]
                pred_dst_dst = out["pred_dst_dst"]
                pred_dst_dstm = out["pred_dst_dstm"]
                pred_src_dst = out["pred_src_dst"]
                pred_src_dstm = out["pred_src_dstm"]
                pred_src_dst_no_grad = out["pred_src_dst_no_grad"]

                # --- Masked training: apply mask to images before loss ---
                if masked_training:
                    target_src_opt = src_target * src_mask_blur
                    target_dst_opt = dst_target * dst_mask_blur
                    pred_src_src_opt = pred_src_src * src_mask_blur
                    pred_dst_dst_opt = pred_dst_dst * dst_mask_blur
                else:
                    target_src_opt = src_target
                    target_dst_opt = dst_target
                    pred_src_src_opt = pred_src_src
                    pred_dst_dst_opt = pred_dst_dst

                # --- SRC loss (DFL style) ---
                if resolution < 256:
                    loss_src = 10.0 * _dssim(pred_src_src_opt, target_src_opt, filter_size=dssim_filter_size).mean()
                else:
                    loss_src = 5.0 * _dssim(pred_src_src_opt, target_src_opt, filter_size=dssim_filter_size).mean()
                    loss_src = loss_src + 5.0 * _dssim(pred_src_src_opt, target_src_opt, filter_size=int(resolution / 23.2)).mean()
                # MSE
                loss_src = loss_src + 10.0 * ((target_src_opt - pred_src_src_opt) ** 2).mean()
                # Mask loss
                loss_src = loss_src + 10.0 * ((src_mask - pred_src_srcm) ** 2).mean()
                # Eyes/mouth priority (DFL: use dedicated eyes_mouth mask)
                if eyes_mouth_prio:
                    loss_src = loss_src + 300.0 * ((src_target * src_em_mask - pred_src_src * src_em_mask).abs()).mean()

                # --- DST loss (DFL style) ---
                if resolution < 256:
                    loss_dst = 10.0 * _dssim(pred_dst_dst_opt, target_dst_opt, filter_size=dssim_filter_size).mean()
                else:
                    loss_dst = 5.0 * _dssim(pred_dst_dst_opt, target_dst_opt, filter_size=dssim_filter_size).mean()
                    loss_dst = loss_dst + 5.0 * _dssim(pred_dst_dst_opt, target_dst_opt, filter_size=int(resolution / 23.2)).mean()
                loss_dst = loss_dst + 10.0 * ((target_dst_opt - pred_dst_dst_opt) ** 2).mean()
                loss_dst = loss_dst + 10.0 * ((dst_mask - pred_dst_dstm) ** 2).mean()
                if eyes_mouth_prio:
                    loss_dst = loss_dst + 300.0 * ((dst_target * dst_em_mask - pred_dst_dst * dst_em_mask).abs()).mean()

                # --- Face style loss (DFL style) ---
                face_style_power_norm = face_style_power / 100.0
                if face_style_power_norm != 0:
                    style_mask = torch.clamp(_gaussian_blur(pred_src_dstm * pred_dst_dstm, mask_blur_sigma), 0, 1.0)
                    style_mask = style_mask.detach()
                    loss_src = loss_src + _style_loss(
                        pred_src_dst_no_grad * style_mask,
                        pred_dst_dst.detach() * pred_dst_dstm.detach(),
                        gaussian_blur_radius=resolution // 8,
                        loss_weight=10000.0 * face_style_power_norm,
                    )

                # --- Background style loss (DFL style) ---
                bg_style_power_norm = bg_style_power / 100.0
                if bg_style_power_norm != 0:
                    style_mask_blur = torch.clamp(_gaussian_blur(src_mask_blur, mask_blur_sigma), 0, 1.0)
                    style_mask_blur = style_mask_blur.detach()
                    style_anti_blur = 1.0 - style_mask_blur

                    target_dst_style_anti = dst_target * style_anti_blur
                    pred_style_anti = pred_src_dst * style_anti_blur

                    loss_src = loss_src + (10.0 * bg_style_power_norm) * _dssim(
                        pred_style_anti, target_dst_style_anti, filter_size=dssim_filter_size).mean()
                    loss_src = loss_src + (10.0 * bg_style_power_norm) * ((pred_style_anti - target_dst_style_anti) ** 2).mean()

                loss = loss_src + loss_dst

                # --- GAN generator loss (DFL style) ---
                if gan_power != 0 and discriminator is not None:
                    # GAN only on src-src self-reconstruction path (DFL design)
                    fake_d_center, fake_d_unet = discriminator(pred_src_src_opt)
                    ones_center = torch.ones_like(fake_d_center)
                    ones_unet = torch.ones_like(fake_d_unet)
                    gen_gan_loss = (F.binary_cross_entropy_with_logits(fake_d_center, ones_center) +
                                    F.binary_cross_entropy_with_logits(fake_d_unet, ones_unet))
                    loss = loss + gan_power * gen_gan_loss

                    # TV regularization (suppress GAN bright dots)
                    if masked_training:
                        tv_h = (pred_src_src[:, :, 1:, :] - pred_src_src[:, :, :-1, :]) ** 2
                        tv_w = (pred_src_src[:, :, :, 1:] - pred_src_src[:, :, :, :-1]) ** 2
                        loss = loss + 1e-6 * (tv_h.sum() + tv_w.sum())

                        # Anti-mask background constraint
                        src_anti_mask = 1.0 - src_mask_blur
                        loss = loss + 0.02 * ((pred_src_src * src_anti_mask - src_target * src_anti_mask) ** 2).mean()

                # --- True face power: CodeDiscriminator G loss (DFL style) ---
                if true_face_power != 0 and code_discriminator is not None:
                    src_code = model.encode(src_warped)
                    code_res = model.inter.get_out_res()
                    src_code_2d = src_code.reshape(src_code.shape[0], model.ae_dims, code_res, code_res)
                    src_code_d = code_discriminator(src_code_2d)
                    ones_code = torch.ones_like(src_code_d)
                    loss = loss + true_face_power * F.binary_cross_entropy_with_logits(src_code_d, ones_code)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if gradient_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # LR Dropout: apply fixed Bernoulli masks to gradients
            if lr_dropout_masks:
                scaler.unscale_(optimizer)
                for name, param in model.named_parameters():
                    if param.grad is not None and name in lr_dropout_masks:
                        param.grad.data.mul_(lr_dropout_masks[name])

            scaler.step(optimizer)
            scaler.update()

            # DFL cosine LR scheduler step
            if lr_scheduler is not None:
                lr_scheduler.step()

            # --- Discriminator step (DFL: 1:1 D/G ratio) ---
            if gan_power != 0 and discriminator is not None and disc_optimizer is not None:
                with torch.amp.autocast(device.type, enabled=(use_amp and device.type == "cuda")):
                    # Real: discriminator should output 1
                    real_d_center, real_d_unet = discriminator(src_target)
                    ones_c = torch.ones_like(real_d_center)
                    zeros_c = torch.zeros_like(real_d_center)
                    # Fake: discriminator should output 0 (detach from G)
                    fake_d_center, fake_d_unet = discriminator(pred_src_src_opt.detach())
                    zeros_u = torch.zeros_like(fake_d_unet)
                    ones_u = torch.ones_like(real_d_unet)

                    # D loss: BCE(real→1) + BCE(fake→0), dual output each 0.5 weight
                    disc_loss = 0.5 * (
                        F.binary_cross_entropy_with_logits(real_d_center, ones_c) +
                        F.binary_cross_entropy_with_logits(fake_d_center, zeros_c)
                    ) + 0.5 * (
                        F.binary_cross_entropy_with_logits(real_d_unet, ones_u) +
                        F.binary_cross_entropy_with_logits(fake_d_unet, zeros_u)
                    )

                disc_optimizer.zero_grad(set_to_none=True)
                scaler.scale(disc_loss).backward()
                scaler.step(disc_optimizer)

            # --- CodeDiscriminator step (true_face_power) ---
            if true_face_power != 0 and code_discriminator is not None and code_disc_optimizer is not None:
                with torch.amp.autocast(device.type, enabled=(use_amp and device.type == "cuda")):
                    code_res = model.inter.get_out_res()
                    src_code = model.encode(src_warped).detach()
                    dst_code = model.encode(dst_warped).detach()
                    src_code_2d = src_code.reshape(src_code.shape[0], model.ae_dims, code_res, code_res)
                    dst_code_2d = dst_code.reshape(dst_code.shape[0], model.ae_dims, code_res, code_res)

                    src_code_d = code_discriminator(src_code_2d)
                    dst_code_d = code_discriminator(dst_code_2d)
                    ones_code = torch.ones_like(dst_code_d)
                    zeros_code = torch.zeros_like(src_code_d)

                    code_disc_loss = 0.5 * (
                        F.binary_cross_entropy_with_logits(dst_code_d, ones_code) +
                        F.binary_cross_entropy_with_logits(src_code_d, zeros_code)
                    )

                code_disc_optimizer.zero_grad(set_to_none=True)
                scaler.scale(code_disc_loss).backward()
                scaler.step(code_disc_optimizer)

            iter_count += 1
            self._iter_count = iter_count
            iter_ms = (time.time() - t0) * 1000
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                _logger.warning(f"NaN/Inf loss at iter #{iter_count}, skipping")
                continue
            self._loss_history.append((iter_count, loss_val))
            if len(self._loss_history) > 10000:
                self._loss_history = self._loss_history[-5000:]

            if on_iter is not None:
                on_iter(iter_count, loss_val, iter_ms)

            now = time.time()

            # Save event
            if self._save_event.is_set() or (now - last_save_time) >= save_interval_min * 60:
                self._save_checkpoint(model, optimizer, iter_count, model_dir,
                                     lr_dropout_masks=lr_dropout_masks,
                                     lr_scheduler=lr_scheduler,
                                     discriminator=discriminator,
                                     disc_optimizer=disc_optimizer)
                model.save(model_dir)
                last_save_time = now
                self._preview_event.set()
                if on_save is not None:
                    on_save(iter_count)
                self._save_event.clear()

            # Preview
            need_preview = (now - last_preview_time) >= preview_interval_sec or self._preview_event.is_set()
            if on_preview is not None and need_preview:
                try:
                    preview_img = self._generate_preview(model, device, resolution)
                    on_preview(preview_img)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        _logger.warning("Preview OOM, skipping")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise
                last_preview_time = now
                self._preview_event.clear()

        # Final save
        self._save_checkpoint(model, optimizer, iter_count, model_dir,
                             lr_dropout_masks=lr_dropout_masks,
                             lr_scheduler=lr_scheduler,
                             discriminator=discriminator,
                             disc_optimizer=disc_optimizer)
        model.save(model_dir)
        try:
            lock_path.unlink()
        except OSError:
            pass
        if on_save is not None:
            on_save(iter_count)
        _logger.info(f"SAEHD training completed at iter #{iter_count}")

    # ---- Preview ----

    def _generate_preview(self, model: SAEHDModel, device: torch.device, resolution: int) -> np.ndarray:
        model.eval()
        n = self._preview_n

        src_samples = self._sample_images(self._preview_src_paths, n, resolution)
        dst_samples = self._sample_images(self._preview_dst_paths, n, resolution)

        sections = []
        if src_samples and dst_samples:
            rows = []
            for i in range(min(len(src_samples), len(dst_samples))):
                s_bgr, s_img_t = src_samples[i]
                d_bgr, d_img_t = dst_samples[i]
                row = self._preview_row(model, device, s_bgr, s_img_t, d_bgr, d_img_t)
                rows.append(row)
            if rows:
                sections.append(("SAEHD", np.vstack(rows)))

        model.train()

        if not sections:
            return np.zeros((resolution, resolution * 5, 3), dtype=np.uint8)

        idx = 0  # Only one section for now
        name, preview_bgr = sections[idx]
        h, w = preview_bgr.shape[:2]

        head = self._draw_head(w, name, 0, 1)
        chart = self._draw_loss_chart(w, 100) if len(self._loss_history) > 2 else np.zeros((100, w, 3), dtype=np.float32)

        final = np.vstack([head, chart, preview_bgr])
        return (np.clip(final, 0, 1) * 255).astype(np.uint8)

    def _preview_row(self, model: SAEHDModel, device: torch.device,
                     s_bgr: np.ndarray, s_img_t: torch.Tensor,
                     d_bgr: np.ndarray, d_img_t: torch.Tensor) -> np.ndarray:
        """DFL preview: S | SS*mask+S*(1-mask) | D | DD*mask+D*(1-mask) | SD*mask+D*(1-mask)"""
        S = s_bgr.astype(np.float32) / 255.0  # BGR
        D = d_bgr.astype(np.float32) / 255.0  # BGR

        s_t = s_img_t.to(device)
        d_t = d_img_t.to(device)

        with torch.inference_mode():
            out = model(s_t, d_t)

            # Get all masks (squeeze to 2D, then expand for broadcast)
            src_mask = out["pred_src_srcm"].squeeze().cpu().numpy()
            dst_mask = out["pred_dst_dstm"].squeeze().cpu().numpy()
            swap_mask = out["pred_src_dstm"].squeeze().cpu().numpy()
            src_mask_3 = src_mask[:, :, np.newaxis]
            dst_mask_3 = dst_mask[:, :, np.newaxis]
            sd_mask_3 = np.clip(dst_mask * swap_mask, 0, 1)[:, :, np.newaxis]

            # SS: src self-recon, mask composited with S background
            SS = out["pred_src_src"].squeeze(0).permute(1, 2, 0).cpu().numpy()
            SS = np.clip(SS, 0, 1)  # RGB
            S_rgb = S[:, :, ::-1]   # BGR→RGB
            SS_comp = SS * src_mask_3 + S_rgb * (1.0 - src_mask_3)
            SS_comp = np.clip(SS_comp, 0, 1)[:, :, ::-1].copy()  # RGB→BGR

            # DD: dst self-recon, mask composited with D background
            DD = out["pred_dst_dst"].squeeze(0).permute(1, 2, 0).cpu().numpy()
            DD = np.clip(DD, 0, 1)  # RGB
            D_rgb = D[:, :, ::-1]   # BGR→RGB
            DD_comp = DD * dst_mask_3 + D_rgb * (1.0 - dst_mask_3)
            DD_comp = np.clip(DD_comp, 0, 1)[:, :, ::-1].copy()  # RGB→BGR

            # SD: swap, mask composited with D background
            SD = out["pred_src_dst"].squeeze(0).permute(1, 2, 0).cpu().numpy()
            SD = np.clip(SD, 0, 1)  # RGB
            SD_comp = SD * sd_mask_3 + D_rgb * (1.0 - sd_mask_3)
            SD_comp = np.clip(SD_comp, 0, 1)[:, :, ::-1].copy()  # RGB→BGR

        row = np.concatenate([S, SS_comp, D, DD_comp, SD_comp], axis=1)
        return np.clip(row, 0, 1)

    def _sample_images(self, paths: list[Path], n: int, resolution: int) -> list[tuple[np.ndarray, torch.Tensor]]:
        if not paths:
            return []
        sampled = random.sample(paths, min(n, len(paths)))
        result = []
        for p in sampled:
            cache_key = f"{p.parent.name}_{p.parent.parent.name}_{p.name}_{resolution}"
            cached = self._preview_cache.get(cache_key)
            if cached is not None:
                result.append(cached)
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            resized = cv2.resize(img, (resolution, resolution))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # [0,1] range for SAEHD
            img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            entry = (resized, img_t)
            self._preview_cache[cache_key] = entry
            result.append(entry)
        return result

    # ---- Checkpoint ----

    def _save_checkpoint(self, model: SAEHDModel, optimizer: torch.optim.Optimizer, iteration: int, model_dir: Path,
                        lr_dropout_masks: Optional[dict] = None,
                        lr_scheduler: Optional[_DFLCosineScheduler] = None,
                        discriminator: Optional[nn.Module] = None,
                        disc_optimizer: Optional[torch.optim.Optimizer] = None):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "iter": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_dropout_masks": {k: v.cpu() for k, v in lr_dropout_masks.items()} if lr_dropout_masks else {},
        }
        if lr_scheduler is not None:
            ckpt["lr_scheduler_step"] = lr_scheduler._step_count
        if discriminator is not None:
            ckpt["disc_state_dict"] = discriminator.state_dict()
            ckpt["disc_optimizer_state_dict"] = disc_optimizer.state_dict()
        buf = io.BytesIO()
        torch.save(ckpt, buf)
        FileManager.atomic_write(model_dir / f"{_MODEL_PREFIX}_ckpt.pt", buf.getvalue())

    def _load_checkpoint(self, model: SAEHDModel, optimizer: torch.optim.Optimizer, model_dir: Path,
                        lr_dropout_masks: Optional[dict] = None,
                        lr_scheduler: Optional[_DFLCosineScheduler] = None,
                        discriminator: Optional[nn.Module] = None,
                        disc_optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        ckpt_path = Path(model_dir) / f"{_MODEL_PREFIX}_ckpt.pt"
        if not ckpt_path.exists():
            return 0
        try:
            data = open(str(ckpt_path), "rb").read()
            ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            iteration = ckpt.get("iter", 0)
            saved_masks = ckpt.get("lr_dropout_masks", {})
            if lr_dropout_masks is not None and saved_masks:
                for k, v in saved_masks.items():
                    if k in lr_dropout_masks:
                        lr_dropout_masks[k] = v.to(lr_dropout_masks[k].device)
            if lr_scheduler is not None:
                lr_scheduler._step_count = ckpt.get("lr_scheduler_step", 0)
            if discriminator is not None and "disc_state_dict" in ckpt:
                discriminator.load_state_dict(ckpt["disc_state_dict"])
            if disc_optimizer is not None and "disc_optimizer_state_dict" in ckpt:
                disc_optimizer.load_state_dict(ckpt["disc_optimizer_state_dict"])
            _logger.info(f"Resumed SAEHD training from iter #{iteration}")
            return iteration
        except Exception as e:
            _logger.warning(f"Failed to load checkpoint: {e}")
            return 0

    # ---- UI helpers (reused from TFM trainer) ----

    def _draw_head(self, width: int, name: str, idx: int, total: int) -> np.ndarray:
        ts = datetime.now().strftime("%H:%M:%S")
        lines = [
            f"[{ts}] [s]:save  [p]:update  [space]:next  [l]:range  [Enter]:stop",
            f'Preview: "{name}" [{idx + 1}/{total}]',
        ]
        line_h = 20
        head_h = line_h * len(lines)
        head = np.zeros((head_h, width, 3), dtype=np.uint8)
        pil_img = Image.fromarray(head)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("consola.ttf", 14)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except Exception:
                font = ImageFont.load_default()
        for i, line in enumerate(lines):
            y = i * line_h + 2
            draw.text((6, y), line, fill=(200, 200, 200), font=font)
        head_rgb = np.array(pil_img, dtype=np.float32) / 255.0
        return head_rgb[:, :, ::-1].copy()

    def _draw_loss_chart(self, width: int, height: int) -> np.ndarray:
        chart = np.zeros((height, width, 3), dtype=np.uint8)
        if len(self._loss_history) < 2:
            return chart.astype(np.float32) / 255.0

        range_labels = ["all", "last 1k", "last 100"]
        range_limits = [0, 1000, 100]
        limit = range_limits[self._loss_history_range]
        history = self._loss_history[-limit:] if limit > 0 else self._loss_history

        iters = [h[0] for h in history]
        losses = [h[1] for h in history]

        abs_max = np.mean(losses[len(losses) // 5:]) * 2
        if abs_max <= 0:
            abs_max = 1.0

        lh_len = len(losses)
        l_per_col = lh_len / max(width, 1)

        for col in range(width):
            start_i = int(col * l_per_col)
            end_i = min(int((col + 1) * l_per_col) + 1, lh_len)
            col_losses = losses[start_i:end_i]
            if not col_losses:
                continue
            p_max = max(col_losses)
            p_min = min(col_losses)
            ph_max = int(np.clip((p_max / abs_max) * (height - 1), 0, height - 1))
            ph_min = int(np.clip((p_min / abs_max) * (height - 1), 0, height - 1))
            for ph in range(ph_min, ph_max + 1):
                chart[height - ph - 1, col] = (255, 200, 0)

        for i in range(6):
            y = int(i * (height - 1) / 5)
            chart[y, :] = (60, 60, 60)

        last_iter = iters[-1] if iters else 0
        range_label = range_labels[self._loss_history_range]
        chart_rgb = cv2.cvtColor(chart, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(chart_rgb)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("consola.ttf", 13)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", 13)
            except Exception:
                font = ImageFont.load_default()
        draw.text((6, height - 18), f"Iter: {last_iter}  Range: {range_label}", fill=(200, 200, 200), font=font)
        result_bgr = np.array(pil_img, dtype=np.float32) / 255.0
        return result_bgr[:, :, ::-1].copy()
