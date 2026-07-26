import colorsys
import io
import itertools
import json
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, RandomSampler

from DeepFaceLab.core.insightface_adapter import InsightFaceAdapter
from DeepFaceLab.core.metadata_manager import MetadataManager
from DeepFaceLab.models.tfm_model import TFMModel, _TFM_PRESETS
from DeepFaceLab.models.tfm_dataset import TFMDataset
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger
from DeepFaceLab.shared.torch_config import get_dataloader_config, get_non_blocking, worker_init_fn

_logger = get_logger("tfm_trainer")

_MODEL_PREFIX = "TFM"
_SAVE_INTERVAL_SEC = 1200
_PREVIEW_INTERVAL_SEC = 60

_LANDMARK_EYE_R = [35, 41, 40, 42, 39, 37, 33, 36]
_LANDMARK_EYE_L = [93, 96, 94, 95, 89, 90, 87, 91]
_LANDMARK_NOSE = [72, 73, 74, 86, 75, 76, 77, 78, 79, 80, 85, 84, 83, 82, 81]
_LANDMARK_MOUTH = [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54]
_LANDMARK_JAW = [1,9,10,11,12,13,14,15,16,2,3,4,5,6,7,8,0,24,23,22,21,20,19,18,32,31,30,29,28,27,26,25,17]


def _gaussian_2d(shape: tuple[int, int], center: tuple[float, float], sigma: float) -> np.ndarray:
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    d2 = (x - center[0]) ** 2 + (y - center[1]) ** 2
    return np.exp(-d2 / (2 * sigma * sigma))


def _build_region_weight_map(
    resolution: int,
    landmarks: np.ndarray,
    eye_priority: float = 1.0,
    mouth_priority: float = 1.0,
    nose_priority: float = 1.0,
    jaw_priority: float = 1.0,
) -> np.ndarray:
    wmap = np.ones((resolution, resolution), dtype=np.float32)
    if landmarks is None or landmarks.max() == 0:
        return wmap
    sigma = resolution / 16.0

    def _add_region(indices, priority):
        nonlocal wmap
        if priority <= 1.0:
            return
        pts = landmarks[indices]
        valid = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
        if len(valid) == 0:
            return
        center = valid.mean(axis=0).astype(np.float64)
        g = _gaussian_2d((resolution, resolution), (center[0], center[1]), sigma)
        wmap += (priority - 1.0) * g

    _add_region(_LANDMARK_EYE_R + _LANDMARK_EYE_L, eye_priority)
    _add_region(_LANDMARK_MOUTH, mouth_priority)
    _add_region(_LANDMARK_NOSE, nose_priority)
    _add_region(_LANDMARK_JAW, jaw_priority)
    return wmap


def _weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight_map is None:
        return nn.functional.l1_loss(pred, target)
    diff = (pred - target).abs()
    if diff.shape[1] == 3 and weight_map.shape[1] == 1:
        diff = diff * weight_map
    else:
        diff = diff * weight_map
    return diff.mean()


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = "#" * filled + " " * (width - filled)
    return f"{bar}| {current}/{total}"


def _format_model_summary(model_name: str, iteration: int, options: dict, device_info: str = "") -> str:
    keys = list(options.keys())
    vals = [str(options[k]) for k in keys]
    keys += ["Model name", "Current iteration"]
    vals += [model_name, str(iteration)]
    if device_info:
        keys += ["Device"]
        vals += [device_info]
    w_name = max(len(k) for k in keys) + 1
    w_val = max(len(v) for v in vals) + 1
    w_total = w_name + w_val + 2
    lines = []
    lines.append(f'=={" Model Summary ":=^{w_total}}==')
    lines.append(f'=={" " * w_total}==')
    lines.append(f'=={"Model name":>{w_name}}: {model_name:<{w_val}}==')
    lines.append(f'=={" " * w_total}==')
    lines.append(f'=={"Current iteration":>{w_name}}: {str(iteration):<{w_val}}==')
    lines.append(f'=={" " * w_total}==')
    lines.append(f'=={" Model Options ":-^{w_total}}==')
    lines.append(f'=={" " * w_total}==')
    for k, v in options.items():
        lines.append(f'=={k:>{w_name}}: {str(v):<{w_val}}==')
    lines.append(f'=={" " * w_total}==')
    if device_info:
        lines.append(f'=={" Running On ":-^{w_total}}==')
        lines.append(f'=={" " * w_total}==')
        lines.append(f'=={"Device":>{w_name}}: {device_info:<{w_val}}==')
        lines.append(f'=={" " * w_total}==')
    lines.append(f'=={"=" * w_total}==')
    return "\n".join(lines)


class TFMTrainer:
    def __init__(self, device: str = "auto") -> None:
        self._device_str = device
        self._device = None
        self._stop_event = threading.Event()
        self._preview_event = threading.Event()
        self._save_event = threading.Event()
        self._loss_history: list[tuple[int, float]] = []
        self._preview_page = 0
        self._loss_history_range = 0
        self._model = None
        self._optimizer = None
        self._disc_optimizer = None
        self._scaler = None
        self._iter_count = 0
        self._model_dir = None
        self._log_need_newline = True
        self._preview_src_paths: list[Path] = []
        self._preview_dst_paths: list[Path] = []
        self._preview_n = 3
        self._preview_resolution = 128
        self._preview_cache: dict[str, tuple[np.ndarray, torch.Tensor, torch.Tensor]] = {}

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

    def request_stop(self):
        self._stop_event.set()

    def request_preview(self):
        self._preview_event.set()

    def request_save(self):
        self._save_event.set()

    def cycle_loss_range(self):
        self._loss_history_range = (self._loss_history_range + 1) % 3

    def train(
        self,
        src_aligned_dir: Path,
        dst_aligned_dir: Path,
        model_dir: Path,
        resolution: int = 128,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        face_type: str = "whole_face",
        use_amp: bool = True,
        random_warp: bool = True,
        gan_power: float = 0.0,
        random_hsv_power: float = 0.0,
        lr_schedule: str = "constant",
        gradient_clip: float = 1.0,
        random_flip: bool = True,
        color_transfer: str = "none",
        model_preset: str = "medium",
        window_size: int = 8,
        skip_strength: float = 0.5,
        gradient_checkpoint: bool = False,
        use_compile: bool = False,
        eye_priority: float = 1.0,
        mouth_priority: float = 1.0,
        nose_priority: float = 1.0,
        jaw_priority: float = 1.0,
        face_style_power: float = 0.0,
        bg_style_power: float = 0.0,
        perceptual_weight: float = 0.0,
        uniform_yaw_sampling: bool = False,
        enable_mask: bool = True,
        save_interval_min: float = 15.0,
        preview_interval_sec: float = 60,
        on_iter: Optional[Callable[[int, float, float], None]] = None,
        on_preview: Optional[Callable[[np.ndarray], None]] = None,
        on_save: Optional[Callable[[int], None]] = None,
        on_log: Optional[Callable[[str, bool], None]] = None,
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

        adapter = InsightFaceAdapter(ctx_id=0 if device.type == "cuda" else -1)

        src_cache_path = model_dir / "TFM_identity_cache_src.npz"
        dst_cache_path = model_dir / "TFM_identity_cache_dst.npz"

        if src_cache_path.exists():
            src_id_cache = TFMDataset.load_identity_cache(src_cache_path)
        else:
            if on_log is not None:
                on_log("Building source identity cache...", False)
            src_id_cache = {}
            src_imgs = FileManager.find_images(src_dir)
            src_total = len(src_imgs)
            for si, img_path in enumerate(src_imgs):
                img = cv2.imread(str(img_path))
                if img is not None:
                    try:
                        faces = adapter.detect_faces(img, max_num=1)
                        if faces and faces[0].embedding is not None:
                            src_id_cache[img_path.name] = faces[0].embedding
                    except Exception:
                        pass
                if (si + 1) % 10 == 0 or si + 1 == src_total:
                    if on_log is not None:
                        on_log(f"Loading src{_progress_bar(si + 1, src_total)}", True)
            if on_log is not None:
                on_log(f"Loading src{_progress_bar(src_total, src_total)}", False)
            model_dir.mkdir(parents=True, exist_ok=True)
            TFMDataset.save_identity_cache(src_id_cache, src_cache_path)

        if dst_cache_path.exists():
            dst_id_cache = TFMDataset.load_identity_cache(dst_cache_path)
        else:
            if on_log is not None:
                on_log("Building target identity cache...", False)
            dst_id_cache = {}
            dst_imgs = FileManager.find_images(dst_dir)
            dst_total = len(dst_imgs)
            for di, img_path in enumerate(dst_imgs):
                img = cv2.imread(str(img_path))
                if img is not None:
                    try:
                        faces = adapter.detect_faces(img, max_num=1)
                        if faces and faces[0].embedding is not None:
                            dst_id_cache[img_path.name] = faces[0].embedding
                    except Exception:
                        pass
                if (di + 1) % 10 == 0 or di + 1 == dst_total:
                    if on_log is not None:
                        on_log(f"Loading dst{_progress_bar(di + 1, dst_total)}", True)
            if on_log is not None:
                on_log(f"Loading dst{_progress_bar(dst_total, dst_total)}", False)
            model_dir.mkdir(parents=True, exist_ok=True)
            TFMDataset.save_identity_cache(dst_id_cache, dst_cache_path)

        src_ds = TFMDataset(src_dir, resolution=resolution, is_src=True, identity_cache=src_id_cache, augment=True, random_hsv_power=random_hsv_power, random_warp=random_warp, color_transfer=color_transfer)
        dst_ds = TFMDataset(dst_dir, resolution=resolution, is_src=False, identity_cache=dst_id_cache, augment=True, random_hsv_power=random_hsv_power, random_warp=random_warp, color_transfer=color_transfer)

        dl_cfg = get_dataloader_config("gpu_train" if device.type == "cuda" else "cpu_train", dataset_size=len(src_ds) + len(dst_ds))

        def _make_yaw_sampler(dataset):
            if not uniform_yaw_sampling:
                return RandomSampler(dataset, replacement=True, num_samples=max(len(dataset), batch_size * 50))
            yaws = []
            for i in range(len(dataset)):
                meta = dataset._metadata_cache.get(dataset._image_paths[i].name)
                if meta is not None and meta.yaw is not None:
                    yaws.append(abs(meta.yaw))
                else:
                    yaws.append(0.5)
            yaws = np.array(yaws)
            bins = np.linspace(0, yaws.max() + 1e-6, 11)
            bin_idx = np.digitize(yaws, bins) - 1
            bin_counts = np.bincount(bin_idx, minlength=10).astype(np.float64)
            bin_weights = 1.0 / (bin_counts + 1)
            sample_weights = torch.from_numpy(bin_weights[bin_idx]).float()
            return torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=max(len(dataset), batch_size * 50), replacement=True)

        src_sampler = _make_yaw_sampler(src_ds)
        dst_sampler = _make_yaw_sampler(dst_ds)

        src_loader = DataLoader(
            src_ds,
            batch_size=batch_size,
            sampler=src_sampler,
            num_workers=dl_cfg["num_workers"],
            pin_memory=dl_cfg["pin_memory"],
            drop_last=True,
            worker_init_fn=worker_init_fn if dl_cfg["num_workers"] > 0 else None,
            persistent_workers=dl_cfg["num_workers"] > 0,
            prefetch_factor=4 if dl_cfg["num_workers"] > 0 else None,
        )
        dst_loader = DataLoader(
            dst_ds,
            batch_size=batch_size,
            sampler=dst_sampler,
            num_workers=dl_cfg["num_workers"],
            pin_memory=dl_cfg["pin_memory"],
            drop_last=True,
            worker_init_fn=worker_init_fn if dl_cfg["num_workers"] > 0 else None,
            persistent_workers=dl_cfg["num_workers"] > 0,
            prefetch_factor=4 if dl_cfg["num_workers"] > 0 else None,
        )

        model = TFMModel.from_preset(
            preset=model_preset,
            resolution=resolution,
            gan_power=gan_power,
            window_size=window_size,
            gradient_checkpoint=gradient_checkpoint,
        ).to(device)

        if use_compile and device.type == "cuda" and hasattr(torch, "compile"):
            try:
                model.encoder = torch.compile(model.encoder, mode="reduce-overhead")
                model.decoder = torch.compile(model.decoder, mode="reduce-overhead")
                model.wplus_mapper = torch.compile(model.wplus_mapper, mode="reduce-overhead")
                if on_log is not None:
                    on_log("torch.compile() enabled (reduce-overhead)", False)
            except Exception as e:
                _logger.warning(f"torch.compile() failed: {e}, continuing without compilation")

        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

        disc_optimizer = None
        if gan_power > 0 and model.discriminator is not None:
            disc_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate * 0.5)

        lr_scheduler = None
        if lr_schedule == "cosine_annealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200000, eta_min=1e-7)

        self._save_config(model_dir, {
            "resolution": resolution,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "face_type": face_type,
            "use_amp": use_amp,
            "random_warp": random_warp,
            "gan_power": gan_power,
            "random_hsv_power": random_hsv_power,
            "lr_schedule": lr_schedule,
            "gradient_clip": gradient_clip,
            "random_flip": random_flip,
            "color_transfer": color_transfer,
            "model_preset": model_preset,
            "window_size": window_size,
            "skip_strength": skip_strength,
            "gradient_checkpoint": gradient_checkpoint,
            "eye_priority": eye_priority,
            "mouth_priority": mouth_priority,
            "nose_priority": nose_priority,
            "jaw_priority": jaw_priority,
            "face_style_power": face_style_power,
            "bg_style_power": bg_style_power,
            "perceptual_weight": perceptual_weight,
            "uniform_yaw_sampling": uniform_yaw_sampling,
            "enable_mask": enable_mask,
            "save_interval_min": save_interval_min,
            "preview_interval_sec": preview_interval_sec,
        })

        perceptual_loss_fn = None
        if perceptual_weight > 0:
            from DeepFaceLab.models.tfm_model import VGGPerceptualLoss
            perceptual_loss_fn = VGGPerceptualLoss().to(device).eval()

        lock_path = model_dir / ".training_lock"
        model_dir.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(str(time.time()))

        start_iter = self._load_checkpoint(model, optimizer, disc_optimizer, model_dir)

        total_params = sum(p.numel() for p in model.parameters())
        device_info = ""
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            _props = torch.cuda.get_device_properties(0)
            gpu_mem = getattr(_props, 'total_memory', getattr(_props, 'total_mem', 0)) / 1024 ** 3
            device_info = f"{gpu_name} ({gpu_mem:.1f}GB)"

        summary = _format_model_summary(
            model_name=f"TFM_{model_preset}",
            iteration=start_iter,
            options={
                "resolution": resolution,
                "face_type": face_type,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_amp": use_amp,
                "model_preset": model_preset,
                "window_size": window_size,
                "gan_power": gan_power,
                "perceptual_weight": perceptual_weight,
                "random_warp": random_warp,
                "random_flip": random_flip,
                "lr_schedule": lr_schedule,
                "gradient_clip": gradient_clip,
                "gradient_checkpoint": gradient_checkpoint,
                "use_compile": use_compile,
                "skip_strength": skip_strength,
                "eye_priority": eye_priority,
                "mouth_priority": mouth_priority,
                "nose_priority": nose_priority,
                "jaw_priority": jaw_priority,
                "face_style_power": face_style_power,
                "bg_style_power": bg_style_power,
                "random_hsv_power": random_hsv_power,
                "color_transfer": color_transfer,
                "uniform_yaw_sampling": uniform_yaw_sampling,
                "enable_mask": enable_mask,
                "save_interval_min": save_interval_min,
                "preview_interval_sec": preview_interval_sec,
            },
            device_info=device_info,
        )
        print(summary)
        if on_log is not None:
            for line in summary.split("\n"):
                on_log(line, False)
            on_log("", False)

        self._preview_src_paths = FileManager.find_images(src_dir)
        self._preview_dst_paths = FileManager.find_images(dst_dir)
        self._preview_n = min(3, batch_size, 800 // resolution)
        self._preview_resolution = resolution

        self._model = model
        self._optimizer = optimizer
        self._disc_optimizer = disc_optimizer
        self._scaler = scaler
        self._model_dir = model_dir

        last_preview_time = 0.0
        last_save_time = time.time()
        iter_count = start_iter
        self._iter_count = iter_count

        src_iter = iter(src_loader)
        dst_iter = iter(dst_loader)

        for _ in itertools.count():
            if self._stop_event.is_set():
                break

            model.train()

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

            src_img = src_batch["image"].to(device, non_blocking=get_non_blocking()).to(memory_format=torch.channels_last)
            src_id = src_batch["identity"].to(device, non_blocking=get_non_blocking())
            src_mask = src_batch["mask"].to(device, non_blocking=get_non_blocking())
            src_lm = src_batch["landmarks"]
            dst_img = dst_batch["image"].to(device, non_blocking=get_non_blocking()).to(memory_format=torch.channels_last)
            dst_id = dst_batch["identity"].to(device, non_blocking=get_non_blocking())
            dst_mask = dst_batch["mask"].to(device, non_blocking=get_non_blocking())
            dst_lm = dst_batch["landmarks"]

            src_weight = torch.ones_like(src_mask)
            dst_weight = torch.ones_like(dst_mask)
            if enable_mask:
                src_weight = src_mask
                dst_weight = dst_mask
            need_region_weight = (eye_priority > 1.0 or mouth_priority > 1.0 or nose_priority > 1.0 or jaw_priority > 1.0)
            if need_region_weight:
                src_wmap_np = np.stack([
                    _build_region_weight_map(resolution, src_lm[b].numpy(), eye_priority, mouth_priority, nose_priority, jaw_priority)
                    for b in range(src_lm.shape[0])
                ])
                src_weight = src_weight * torch.from_numpy(src_wmap_np).unsqueeze(1).to(device)
                dst_wmap_np = np.stack([
                    _build_region_weight_map(resolution, dst_lm[b].numpy(), eye_priority, mouth_priority, nose_priority, jaw_priority)
                    for b in range(dst_lm.shape[0])
                ])
                dst_weight = dst_weight * torch.from_numpy(dst_wmap_np).unsqueeze(1).to(device)

            with torch.amp.autocast(device.type, enabled=(use_amp and device.type == "cuda")):
                src_enc_feat, src_w_plus = model.encode(src_img)
                src_recon = model.decode(src_w_plus, src_id, src_enc_feat)
                loss_src = _weighted_l1_loss(src_recon, src_img, src_weight)

                dst_enc_feat, dst_w_plus = model.encode(dst_img)
                dst_recon = model.decode(dst_w_plus, dst_id, dst_enc_feat)
                loss_dst = _weighted_l1_loss(dst_recon, dst_img, dst_weight)

                swap_recon = model.decode(dst_w_plus, src_id, dst_enc_feat)
                loss_swap = _weighted_l1_loss(swap_recon, src_img, src_weight) + \
                            _weighted_l1_loss(swap_recon, dst_recon.detach(), dst_weight)

                loss = loss_src + loss_dst + loss_swap

                if iter_count % 5 == 0:
                    swap_rev = model.decode(src_w_plus, dst_id, src_enc_feat)
                    loss_swap_rev = _weighted_l1_loss(swap_rev, dst_img, dst_weight) + \
                                    _weighted_l1_loss(swap_rev, src_recon.detach(), src_weight)
                    loss = loss + loss_swap_rev

                if face_style_power > 0 and enable_mask:
                    swap_face = swap_recon * dst_weight + dst_img * (1.0 - dst_weight)
                    loss = loss + face_style_power * _weighted_l1_loss(swap_recon, swap_face.detach(), dst_weight)

                if bg_style_power > 0 and enable_mask:
                    swap_bg = swap_recon * (1.0 - dst_weight) + dst_img * dst_weight
                    loss = loss + bg_style_power * _weighted_l1_loss(swap_recon, swap_bg.detach(), 1.0 - dst_weight)

                if perceptual_weight > 0 and perceptual_loss_fn is not None:
                    loss = loss + perceptual_weight * perceptual_loss_fn(src_recon, src_img)
                    loss = loss + perceptual_weight * perceptual_loss_fn(dst_recon, dst_img)
                    loss = loss + perceptual_weight * perceptual_loss_fn(swap_recon, src_img)

                if gan_power > 0 and model.discriminator is not None:
                    disc_real_src = model.discriminate(src_img)
                    disc_fake_src = model.discriminate(src_recon.detach())
                    disc_real_dst = model.discriminate(dst_img)
                    disc_fake_swap = model.discriminate(swap_recon.detach())
                    disc_loss = nn.functional.binary_cross_entropy_with_logits(disc_real_src, torch.ones_like(disc_real_src)) + \
                                nn.functional.binary_cross_entropy_with_logits(disc_fake_src, torch.zeros_like(disc_fake_src)) + \
                                nn.functional.binary_cross_entropy_with_logits(disc_real_dst, torch.ones_like(disc_real_dst)) + \
                                nn.functional.binary_cross_entropy_with_logits(disc_fake_swap, torch.zeros_like(disc_fake_swap))

                    if disc_optimizer is not None:
                        disc_optimizer.zero_grad()
                        scaler.scale(disc_loss).backward()
                        scaler.step(disc_optimizer)

                    gen_gan_loss = nn.functional.binary_cross_entropy_with_logits(
                        model.discriminate(src_recon), torch.ones_like(disc_real_src)
                    )
                    loss = loss + gan_power * gen_gan_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            iter_count += 1
            self._iter_count = iter_count
            iter_ms = (time.time() - t0) * 1000
            loss_val = loss.item()
            self._loss_history.append((iter_count, loss_val))

            if on_iter is not None:
                on_iter(iter_count, loss_val, iter_ms)

            now = time.time()

            if self._save_event.is_set():
                if on_log is not None:
                    on_log("", False)
                self._save_checkpoint(model, optimizer, disc_optimizer, iter_count, model_dir)
                last_save_time = now
                self._save_event.clear()
                self._preview_event.set()
                if on_save is not None:
                    on_save(iter_count)

            need_preview = (now - last_preview_time) >= preview_interval_sec or self._preview_event.is_set()
            if on_preview is not None and need_preview:
                try:
                    preview_img = self._generate_preview(model, device, resolution)
                    on_preview(preview_img)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        _logger.warning("Preview generation OOM, skipping")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise
                last_preview_time = now
                self._preview_event.clear()

            if (now - last_save_time) >= save_interval_min * 60:
                if on_log is not None:
                    on_log("", False)
                self._save_checkpoint(model, optimizer, disc_optimizer, iter_count, model_dir)
                last_save_time = now
                self._preview_event.set()
                if on_save is not None:
                    on_save(iter_count)

        self._save_checkpoint(model, optimizer, disc_optimizer, iter_count, model_dir)
        model.save(model_dir)
        try:
            lock_path.unlink()
        except OSError:
            pass
        if on_save is not None:
            on_save(iter_count)
        _logger.info(f"TFM training completed at iter #{iter_count}")

    def _generate_preview(
        self,
        model: TFMModel,
        device: torch.device,
        resolution: int,
    ) -> np.ndarray:
        model.eval()
        n = self._preview_n

        src_samples = self._sample_images(self._preview_src_paths, n, resolution)
        dst_samples = self._sample_images(self._preview_dst_paths, n, resolution)

        sections = []

        if src_samples and dst_samples:
            rows = []
            for i in range(min(len(src_samples), len(dst_samples))):
                s_bgr, s_img_t, s_id = src_samples[i]
                d_bgr, d_img_t, d_id = dst_samples[i]
                row = self._preview_row_dfl(model, device, resolution, s_bgr, s_img_t, s_id, d_bgr, d_img_t, d_id)
                rows.append(row)
            if rows:
                sections.append(("TFM swap", np.vstack(rows)))

        if src_samples:
            rows = []
            for img_bgr, img_t, identity in src_samples:
                row = self._preview_row_recon(model, device, resolution, img_bgr, img_t, identity)
                rows.append(row)
            if rows:
                sections.append(("TFM src recon", np.vstack(rows)))

        model.train()

        if not sections:
            return np.zeros((resolution, resolution * 5, 3), dtype=np.uint8)

        idx = self._preview_page % len(sections)
        name, preview_bgr = sections[idx]
        h, w = preview_bgr.shape[:2]

        head = self._draw_head(w, name, idx, len(sections))
        chart = self._draw_loss_chart(w, 100) if len(self._loss_history) > 2 else np.zeros((100, w, 3), dtype=np.float32)

        final = np.vstack([head, chart, preview_bgr])
        return (np.clip(final, 0, 1) * 255).astype(np.uint8)

    def _preview_row_dfl(
        self,
        model: TFMModel,
        device: torch.device,
        resolution: int,
        s_bgr: np.ndarray,
        s_img_t: torch.Tensor,
        s_id: torch.Tensor,
        d_bgr: np.ndarray,
        d_img_t: torch.Tensor,
        d_id: torch.Tensor,
    ) -> np.ndarray:
        S = s_bgr.astype(np.float32) / 255.0
        D = d_bgr.astype(np.float32) / 255.0

        s_img_t = s_img_t.to(device)
        s_id = s_id.to(device)
        d_img_t = d_img_t.to(device)
        d_id = d_id.to(device)

        with torch.inference_mode():
            ss_recon = model(s_img_t, s_id)
            SS = ((ss_recon.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0)
            SS = np.clip(SS, 0, 1)[:, :, ::-1].copy()

            dd_recon = model(d_img_t, d_id)
            DD = ((dd_recon.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0)
            DD = np.clip(DD, 0, 1)[:, :, ::-1].copy()

            sd_recon = model(d_img_t, s_id)
            SD = ((sd_recon.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0)
            SD = np.clip(SD, 0, 1)[:, :, ::-1].copy()

        row = np.concatenate([S, SS, D, DD, SD], axis=1)
        return np.clip(row, 0, 1)

    def _preview_row_recon(
        self,
        model: TFMModel,
        device: torch.device,
        resolution: int,
        img_bgr: np.ndarray,
        img_t: torch.Tensor,
        identity: torch.Tensor,
    ) -> np.ndarray:
        I = img_bgr.astype(np.float32) / 255.0
        img_t = img_t.to(device)
        identity = identity.to(device)

        with torch.inference_mode():
            recon = model(img_t, identity)
            recon_np = ((recon.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0)
            recon_np = np.clip(recon_np, 0, 1)
            recon_bgr = recon_np[:, :, ::-1].copy()

        diff = np.abs(I - recon_bgr) * 3.0
        diff = np.clip(diff, 0, 1)
        blank = np.zeros_like(I)

        row = np.concatenate([I, recon_bgr, diff, blank, blank], axis=1)
        return np.clip(row, 0, 1)

    def _sample_images(self, paths: list[Path], n: int, resolution: int) -> list[tuple[np.ndarray, torch.Tensor, torch.Tensor]]:
        if not paths:
            return []
        sampled = random.sample(paths, min(n, len(paths)))
        result = []
        for p in sampled:
            cache_key = f"{p.parent.name}_{p.name}_{resolution}"
            cached = self._preview_cache.get(cache_key)
            if cached is not None:
                result.append(cached)
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            resized = cv2.resize(img, (resolution, resolution))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

            meta = MetadataManager.load(p)
            if meta is not None and meta.arcface_embedding is not None:
                identity = torch.from_numpy(meta.arcface_embedding.astype(np.float32)).unsqueeze(0)
            else:
                identity = torch.zeros(1, 512, dtype=torch.float32)

            entry = (resized, img_t, identity)
            self._preview_cache[cache_key] = entry
            result.append(entry)
        return result

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
        head_bgr = head_rgb[:, :, ::-1].copy()
        return head_bgr

    def _draw_loss_chart(self, width: int, height: int) -> np.ndarray:
        chart = np.zeros((height, width, 3), dtype=np.uint8)
        if len(self._loss_history) < 2:
            return chart.astype(np.float32) / 255.0

        range_labels = ["all", "last 1k", "last 100"]
        range_limits = [0, 1000, 100]
        limit = range_limits[self._loss_history_range]
        if limit > 0:
            history = self._loss_history[-limit:]
        else:
            history = self._loss_history

        iters = [h[0] for h in history]
        losses = [h[1] for h in history]

        abs_max = np.mean(losses[len(losses) // 5:]) * 2
        if abs_max <= 0:
            abs_max = 1.0

        lh_len = len(losses)
        l_per_col = lh_len / width

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

        last_iter = iters[-1]
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
        result_bgr = result_bgr[:, :, ::-1].copy()
        return result_bgr

    def _save_checkpoint(self, model: TFMModel, optimizer: torch.optim.Optimizer, disc_optimizer, iteration: int, model_dir: Path):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "iter": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if disc_optimizer is not None:
            ckpt["disc_optimizer_state_dict"] = disc_optimizer.state_dict()
        buf = io.BytesIO()
        torch.save(ckpt, buf)
        FileManager.atomic_write(model_dir / f"{_MODEL_PREFIX}_ckpt.pt", buf.getvalue())

    def _load_checkpoint(self, model: TFMModel, optimizer: torch.optim.Optimizer, disc_optimizer, model_dir: Path) -> int:
        ckpt_path = Path(model_dir) / f"{_MODEL_PREFIX}_ckpt.pt"
        if not ckpt_path.exists():
            return 0
        try:
            data = open(str(ckpt_path), "rb").read()
            ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if disc_optimizer is not None and "disc_optimizer_state_dict" in ckpt:
                disc_optimizer.load_state_dict(ckpt["disc_optimizer_state_dict"])
            iteration = ckpt.get("iter", 0)
            _logger.info(f"Resumed TFM training from iter #{iteration}")
            return iteration
        except Exception as e:
            _logger.warning(f"Failed to load checkpoint: {e}")
            return 0

    def _save_config(self, model_dir: Path, config: dict) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_dir / f"{_MODEL_PREFIX}_training_config.json"
        FileManager.atomic_write(config_path, json.dumps(config, indent=2))

    def _load_config(self, model_dir: Path) -> dict:
        config_path = Path(model_dir) / f"{_MODEL_PREFIX}_training_config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}
