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
from DeepFaceLab.models.tfm_dataset import TFMDataset
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger
from DeepFaceLab.shared.torch_config import get_dataloader_config, get_non_blocking, worker_init_fn

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
    """DFL style loss: Gram matrix matching with optional Gaussian blur."""
    if gaussian_blur_radius > 0:
        pred = _gaussian_blur(pred, gaussian_blur_radius)
        target = _gaussian_blur(target, gaussian_blur_radius)

    # Channel statistics: mean and std across spatial dims
    pred_mean = pred.mean(dim=[2, 3])
    target_mean = target.mean(dim=[2, 3])
    pred_std = pred.std(dim=[2, 3])
    target_std = target.std(dim=[2, 3])

    loss = ((pred_mean - target_mean) ** 2).mean() + ((pred_std - target_std) ** 2).mean()
    return loss * loss_weight


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

        if not src_dir.exists():
            raise ValueError(f"Source aligned directory not found: {src_dir}")
        if not dst_dir.exists():
            raise ValueError(f"Target aligned directory not found: {dst_dir}")

        # DFL uses [0,1] images. TFMDataset outputs [-1,1], so we convert in the loop.
        # Disable random_warp's inner HSV by passing random_hsv_power=0 to dataset;
        # we handle HSV at the same level as DFL.
        src_ds = TFMDataset(src_dir, resolution=resolution, is_src=True, augment=True,
                            random_hsv_power=random_hsv_power, random_warp=random_warp,
                            random_flip=random_flip, color_transfer=color_transfer)
        dst_ds = TFMDataset(dst_dir, resolution=resolution, is_src=False, augment=True,
                            random_hsv_power=0.0, random_warp=random_warp,
                            random_flip=random_flip, color_transfer=color_transfer)

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
                           ae_dims=ae_dims, e_dims=e_dims, d_dims=d_dims, d_mask_dims=d_mask_dims).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

        # Load checkpoint if exists
        start_iter = self._load_checkpoint(model, optimizer, model_dir)

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

            # Convert from [-1,1] (TFMDataset) to [0,1] (DFL convention)
            src_img = (src_batch["image"].to(device, non_blocking=get_non_blocking()) + 1.0) / 2.0
            src_mask = src_batch["mask"].to(device, non_blocking=get_non_blocking())
            dst_img = (dst_batch["image"].to(device, non_blocking=get_non_blocking()) + 1.0) / 2.0
            dst_mask = dst_batch["mask"].to(device, non_blocking=get_non_blocking())

            # Blurred masks for loss weighting (DFL style)
            src_mask_blur = torch.clamp(_gaussian_blur(src_mask, mask_blur_sigma), 0, 0.5) * 2.0
            dst_mask_blur = torch.clamp(_gaussian_blur(dst_mask, mask_blur_sigma), 0, 0.5) * 2.0

            with torch.amp.autocast(device.type, enabled=(use_amp and device.type == "cuda")):
                out = model(src_img, dst_img)

                pred_src_src = out["pred_src_src"]
                pred_src_srcm = out["pred_src_srcm"]
                pred_dst_dst = out["pred_dst_dst"]
                pred_dst_dstm = out["pred_dst_dstm"]
                pred_src_dst = out["pred_src_dst"]
                pred_src_dstm = out["pred_src_dstm"]
                pred_src_dst_no_grad = out["pred_src_dst_no_grad"]

                # --- Masked training: apply mask to images before loss ---
                if masked_training:
                    target_src_opt = src_img * src_mask_blur
                    target_dst_opt = dst_img * dst_mask_blur
                    pred_src_src_opt = pred_src_src * src_mask_blur
                    pred_dst_dst_opt = pred_dst_dst * dst_mask_blur
                else:
                    target_src_opt = src_img
                    target_dst_opt = dst_img
                    pred_src_src_opt = pred_src_src
                    pred_dst_dst_opt = pred_dst_dst

                # --- SRC loss (DFL style) ---
                # DSSIM
                if resolution < 256:
                    loss_src = 10.0 * _dssim(pred_src_src_opt, target_src_opt, filter_size=dssim_filter_size).mean()
                else:
                    loss_src = 5.0 * _dssim(pred_src_src_opt, target_src_opt, filter_size=dssim_filter_size).mean()
                    loss_src = loss_src + 5.0 * _dssim(pred_src_src_opt, target_src_opt, filter_size=int(resolution / 23.2)).mean()
                # MSE
                loss_src = loss_src + 10.0 * ((target_src_opt - pred_src_src_opt) ** 2).mean()
                # Mask loss
                loss_src = loss_src + 10.0 * ((src_mask - pred_src_srcm) ** 2).mean()
                # Eyes/mouth priority
                if eyes_mouth_prio:
                    # Use src_mask as eyes_mouth mask approximation (simplified)
                    em_mask = src_mask
                    loss_src = loss_src + 300.0 * ((src_img * em_mask - pred_src_src * em_mask).abs()).mean()

                # --- DST loss (DFL style) ---
                if resolution < 256:
                    loss_dst = 10.0 * _dssim(pred_dst_dst_opt, target_dst_opt, filter_size=dssim_filter_size).mean()
                else:
                    loss_dst = 5.0 * _dssim(pred_dst_dst_opt, target_dst_opt, filter_size=dssim_filter_size).mean()
                    loss_dst = loss_dst + 5.0 * _dssim(pred_dst_dst_opt, target_dst_opt, filter_size=int(resolution / 23.2)).mean()
                loss_dst = loss_dst + 10.0 * ((target_dst_opt - pred_dst_dst_opt) ** 2).mean()
                loss_dst = loss_dst + 10.0 * ((dst_mask - pred_dst_dstm) ** 2).mean()
                if eyes_mouth_prio:
                    em_mask = dst_mask
                    loss_dst = loss_dst + 300.0 * ((dst_img * em_mask - pred_dst_dst * em_mask).abs()).mean()

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

                    target_dst_style_anti = dst_img * style_anti_blur
                    pred_style_anti = pred_src_dst * style_anti_blur

                    loss_src = loss_src + (10.0 * bg_style_power_norm) * _dssim(
                        pred_style_anti, target_dst_style_anti, filter_size=dssim_filter_size).mean()
                    loss_src = loss_src + (10.0 * bg_style_power_norm) * ((pred_style_anti - target_dst_style_anti) ** 2).mean()

                loss = loss_src + loss_dst

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if gradient_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            iter_count += 1
            self._iter_count = iter_count
            iter_ms = (time.time() - t0) * 1000
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                _logger.warning(f"NaN/Inf loss at iter #{iter_count}, skipping")
                continue
            self._loss_history.append((iter_count, loss_val))

            if on_iter is not None:
                on_iter(iter_count, loss_val, iter_ms)

            now = time.time()

            # Save event
            if self._save_event.is_set() or (now - last_save_time) >= save_interval_min * 60:
                self._save_checkpoint(model, optimizer, iter_count, model_dir)
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
        self._save_checkpoint(model, optimizer, iter_count, model_dir)
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
        """DFL preview: S | SS | D | DD | SD"""
        S = s_bgr.astype(np.float32) / 255.0
        D = d_bgr.astype(np.float32) / 255.0

        # Convert to [0,1] for model
        s_t = s_img_t.to(device)
        d_t = d_img_t.to(device)

        with torch.inference_mode():
            # SS: src self-reconstruction
            out_s = model(s_t, s_t)
            SS = out_s["pred_src_src"].squeeze().permute(1, 2, 0).cpu().numpy()
            SS = np.clip(SS, 0, 1)[:, :, ::-1].copy()

            # DD: dst self-reconstruction
            out_d = model(d_t, d_t)
            DD = out_d["pred_dst_dst"].squeeze().permute(1, 2, 0).cpu().numpy()
            DD = np.clip(DD, 0, 1)[:, :, ::-1].copy()

            # SD: swap (src face on dst structure)
            out_swap = model(s_t, d_t)
            SD = out_swap["pred_src_dst"].squeeze().permute(1, 2, 0).cpu().numpy()
            SD = np.clip(SD, 0, 1)[:, :, ::-1].copy()

        row = np.concatenate([S, SS, D, DD, SD], axis=1)
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

    def _save_checkpoint(self, model: SAEHDModel, optimizer: torch.optim.Optimizer, iteration: int, model_dir: Path):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "iter": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        buf = io.BytesIO()
        torch.save(ckpt, buf)
        FileManager.atomic_write(model_dir / f"{_MODEL_PREFIX}_ckpt.pt", buf.getvalue())

    def _load_checkpoint(self, model: SAEHDModel, optimizer: torch.optim.Optimizer, model_dir: Path) -> int:
        ckpt_path = Path(model_dir) / f"{_MODEL_PREFIX}_ckpt.pt"
        if not ckpt_path.exists():
            return 0
        try:
            data = open(str(ckpt_path), "rb").read()
            ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            iteration = ckpt.get("iter", 0)
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
