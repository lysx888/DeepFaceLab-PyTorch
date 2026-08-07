import io
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.models.faceset_dataset import FacesetDataset
from faceswap.models.xseg.xseg_model_wrapper import XSegModel, XSegTrainingConfig
from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger
from faceswap.shared.image_utils import bgr_to_rgb, rgb_to_bgr
from faceswap.shared.torch_config import get_dataloader_config, get_non_blocking, worker_init_fn

_logger = get_logger("xseg_trainer")

_SAVE_INTERVAL_SEC = 1500
_BACKUP_INTERVAL_SEC = 7200


class XSegTrainer:
    def __init__(self, device: str = "auto") -> None:
        self._device_str = device
        self._device = None
        self._stop_event = threading.Event()
        self._preview_event = threading.Event()
        self._save_event = threading.Event()
        self._loss_history: list[tuple[int, float]] = []
        self._preview_page = 0
        self._loss_history_range = 0
        self._xseg_model: Optional[XSegModel] = None
        self._iter_count = 0
        self._model_dir = None
        self._preview_train_paths: list[Path] = []
        self._preview_train_meta: dict = {}
        self._preview_train_render_mask = None
        self._preview_src_all: list[Path] = []
        self._preview_dst_all: list[Path] = []
        self._preview_n: int = 3
        self._preview_resolution = 256

    def _resolve_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        if self._device_str == "auto":
            from faceswap.shared.config import auto_select_device
            self._device = auto_select_device()
        else:
            self._device = torch.device(self._device_str)
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
        batch_size: int = 4,
        target_iter: int = 100000,
        learning_rate: float = 1e-4,
        face_type: str = "wf",
        amp_mode: str = "bf16",
        pretrain: bool = False,
        pretrain_iter: int = 10000,
        lr_dropout: float = 0.3,
        pretrain_data_dir: Optional[Path] = None,
        on_iter: Optional[Callable[[int, float, float], None]] = None,
        on_preview: Optional[Callable[[np.ndarray], None]] = None,
        on_save: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._stop_event.clear()
        self._preview_event.clear()
        self._save_event.clear()
        device = self._resolve_device()

        config = XSegTrainingConfig(
            resolution=384 if face_type == "head" else 256,
            face_type=face_type,
            batch_size=batch_size,
            learning_rate=learning_rate,
            pretrain=pretrain,
            target_iter=target_iter,
            amp_mode=amp_mode,
            lr_dropout=lr_dropout,
            pretrain_iter=pretrain_iter,
        )

        from faceswap.shared.torch_config import configure_torch
        configure_torch("gpu_train")

        xseg_model = XSegModel(config, Path(model_dir), device)
        self._xseg_model = xseg_model
        model = xseg_model.xseg_net
        resolution = config.resolution
        self._model_dir = Path(model_dir)

        restored_loss = xseg_model.get_aux_state().get('loss_history', [])
        if restored_loss:
            self._loss_history = restored_loss

        optimizer = xseg_model._optimizers_dict['xseg_opt']
        scheduler = xseg_model._scheduler

        bce_ds = None
        src_dir = Path(src_aligned_dir)
        dst_dir = Path(dst_aligned_dir)
        if src_dir.exists():
            try:
                src_ds = FacesetDataset(src_dir, resolution=resolution, augment=True)
            except ValueError:
                src_ds = None
        else:
            src_ds = None
        if dst_dir.exists():
            try:
                dst_ds = FacesetDataset(dst_dir, resolution=resolution, augment=True)
            except ValueError:
                dst_ds = None
        else:
            dst_ds = None

        datasets = [ds for ds in [src_ds, dst_ds] if ds is not None]
        if datasets:
            bce_ds = FacesetDataset.merge(datasets)
            bce_paths = bce_ds.image_paths
            bce_meta = bce_ds.metadata_cache
        else:
            bce_paths = []
            bce_meta = {}

        if not bce_paths and not pretrain:
            raise ValueError("No annotated faces found. Please annotate faces with XSeg editor first.")

        pretrain_ds = None
        if pretrain:
            pretrain_dir = Path(pretrain_data_dir) if pretrain_data_dir else src_dir
            if pretrain_dir.exists():
                try:
                    pretrain_ds = FacesetDataset(pretrain_dir, resolution=resolution, augment=True, pretrain_mode=True)
                except ValueError:
                    pass
            if pretrain_ds is None and pretrain:
                _logger.warning(f"Pretrain data not found in {pretrain_dir}, falling back to src aligned data")
                if src_dir.exists():
                    try:
                        pretrain_ds = FacesetDataset(src_dir, resolution=resolution, augment=True, pretrain_mode=True)
                    except ValueError:
                        pass
            if pretrain_ds is None:
                raise ValueError("No face images found for pretrain. Provide pretrain_data_dir or ensure src aligned dir has images.")

        from faceswap.shared.config import is_gpu_device
        is_gpu = is_gpu_device(device)
        _ds_size = len(bce_ds) if bce_ds is not None else 0
        if pretrain_ds is not None:
            _ds_size = max(_ds_size, len(pretrain_ds))
        dl_cfg = get_dataloader_config("gpu_train" if is_gpu else "cpu_train", dataset_size=_ds_size)

        bce_loader = None
        if bce_paths:
            steps_per_epoch = max(len(bce_ds), batch_size * 50)
            bce_loader = DataLoader(
                bce_ds,
                batch_size=batch_size,
                sampler=RandomSampler(bce_ds, replacement=True, num_samples=steps_per_epoch),
                num_workers=dl_cfg["num_workers"],
                pin_memory=dl_cfg["pin_memory"],
                drop_last=True,
                worker_init_fn=worker_init_fn if dl_cfg["num_workers"] > 0 else None,
                persistent_workers=dl_cfg["num_workers"] > 0,
                prefetch_factor=dl_cfg.get("prefetch_factor"),
            )

        pretrain_loader = None
        if pretrain_ds is not None:
            pt_steps = max(len(pretrain_ds), batch_size * 50)
            pretrain_loader = DataLoader(
                pretrain_ds,
                batch_size=batch_size,
                sampler=RandomSampler(pretrain_ds, replacement=True, num_samples=pt_steps),
                num_workers=dl_cfg["num_workers"],
                pin_memory=dl_cfg["pin_memory"],
                drop_last=True,
                worker_init_fn=worker_init_fn if dl_cfg["num_workers"] > 0 else None,
                persistent_workers=dl_cfg["num_workers"] > 0,
                prefetch_factor=dl_cfg.get("prefetch_factor"),
            )

        from faceswap.core.amp_utils import AMPManager
        amp = AMPManager(device, amp_mode=amp_mode)
        criterion = torch.nn.BCEWithLogitsLoss()

        iter_count = xseg_model.get_aux_state().get('iter_count', 0)
        self._iter_count = iter_count

        _N_PREVIEW = min(4, batch_size, 800 // resolution)

        n_bce = len(bce_paths)
        n_pt = len(pretrain_ds) if pretrain_ds else 0
        _logger.info(f"XSeg training started: {n_bce} annotated images, {n_pt} pretrain images, target {target_iter} iters, device={device}")
        if pretrain:
            _logger.info(f"Pretrain: grayscale self-reconstruction for first {pretrain_iter} iters (DSSIM+MSE, skip=0)")
        _logger.info('Press "Stop" to stop training and save model.')

        self._preview_train_paths = bce_paths
        self._preview_train_meta = bce_meta
        self._preview_train_render_mask = bce_ds._render_mask if bce_ds else None
        self._preview_src_all = FileManager.find_images(Path(src_aligned_dir)) if Path(src_aligned_dir).exists() else []
        dst_preview_dir = Path(dst_aligned_dir) if dst_aligned_dir is not None and Path(dst_aligned_dir).exists() else Path(src_aligned_dir)
        self._preview_dst_all = FileManager.find_images(dst_preview_dir)
        self._preview_n = _N_PREVIEW
        self._preview_resolution = resolution
        self._pretrain_ds = pretrain_ds

        last_save_time = time.time()
        last_backup_time = time.time()

        lock_file = Path(model_dir) / ".training_lock"
        lock_file.write_text(str(os.getpid()))

        saved_config = self._load_xseg_config(model_dir)
        last_pretrain = saved_config.get("pretrain", None)
        if last_pretrain is False and pretrain:
            _logger.warning("Pretrain not allowed: model has already completed pretrain phase. Starting normal training.")
            pretrain = False
        elif pretrain and iter_count >= pretrain_iter:
            _logger.info(f"Pretrain skipped: current iter {iter_count} >= pretrain_iter {pretrain_iter}")
            pretrain = False

        pt_iter = iter(pretrain_loader) if pretrain_loader else None
        bce_iter = iter(bce_loader) if bce_loader else None

        if on_preview is not None:
            preview_img = self._generate_preview(model, device, resolution, pretrain and iter_count < pretrain_iter)
            on_preview(preview_img)

        try:
            xseg_model.save(iter_count)
            _logger.info(f"Initial save at iter {iter_count}")
            while not self._stop_event.is_set() and iter_count < target_iter:
                t_iter_start = time.time()
                cur_pretrain = pretrain and iter_count < pretrain_iter
                if not hasattr(self, '_last_pretrain') or self._last_pretrain is None:
                    self._last_pretrain = cur_pretrain

                if not cur_pretrain and bce_iter is None:
                    _logger.error("No training data available, stopping")
                    break

                if cur_pretrain != getattr(self, '_last_pretrain', None):
                    self._last_pretrain = cur_pretrain
                    if not cur_pretrain:
                        pretrain = False
                        _logger.info(f"Pretrain complete at iter {iter_count}, switching to BCE mask segmentation")

                if cur_pretrain and pt_iter is not None:
                    try:
                        batch = next(pt_iter)
                    except StopIteration:
                        pt_iter = iter(pretrain_loader)
                        batch = next(pt_iter)
                    imgs = batch["image"].to(device, non_blocking=get_non_blocking())
                    targets = batch["target"].to(device, non_blocking=get_non_blocking())

                    with amp.autocast():
                        pred = model(imgs, skip_enabled=False, pretrain=True)
                        mse = F.mse_loss(pred, targets)
                        from faceswap.models.saehd.losses import dssim
                        fs1 = max(3, int(resolution / 11.6))
                        fs2 = max(3, int(resolution / 23.2))
                        fs1 = fs1 if fs1 % 2 == 1 else fs1 + 1
                        fs2 = fs2 if fs2 % 2 == 1 else fs2 + 1
                        d1 = dssim(targets, pred, max_val=1.0, filter_size=fs1).mean()
                        d2 = dssim(targets, pred, max_val=1.0, filter_size=fs2).mean()
                        loss = 5.0 * d1 + 5.0 * d2 + 10.0 * mse

                elif bce_iter is not None:
                    try:
                        batch = next(bce_iter)
                    except StopIteration:
                        bce_iter = iter(bce_loader)
                        batch = next(bce_iter)
                    imgs = batch["image"].to(device, non_blocking=get_non_blocking())
                    masks = batch["mask"].to(device, non_blocking=get_non_blocking())

                    with amp.autocast():
                        pred = model(imgs, skip_enabled=True, pretrain=False)
                        loss = criterion(pred, masks)
                else:
                    break

                optimizer.zero_grad()
                amp.scale(loss).backward()

                amp.unscale_(optimizer)

                if lr_dropout > 0 and lr_dropout < 1.0:
                    for pg in optimizer.param_groups:
                        for p in pg['params']:
                            if p.grad is not None and np.random.random() > lr_dropout:
                                p.grad.zero_()

                amp.step(optimizer)
                amp.update()
                scheduler.step()

                iter_count += 1
                self._iter_count = iter_count
                iter_ms = (time.time() - t_iter_start) * 1000
                loss_val = loss.item()
                self._loss_history.append((iter_count, loss_val))
                if len(self._loss_history) > 10000:
                    del self._loss_history[:5000]

                if on_iter is not None and not self._stop_event.is_set():
                    on_iter(iter_count, loss_val, iter_ms)

                now = time.time()

                if self._save_event.is_set() and not self._stop_event.is_set():
                    xseg_model._aux_state['loss_history'] = list(self._loss_history)
                    xseg_model.save(iter_count)
                    self._save_event.clear()
                    last_save_time = now
                    if on_preview is not None:
                        preview_img = self._generate_preview(model, device, resolution, cur_pretrain)
                        on_preview(preview_img)
                    if on_save is not None:
                        on_save(iter_count)

                if self._preview_event.is_set() and on_preview is not None and not self._stop_event.is_set():
                    preview_img = self._generate_preview(model, device, resolution, cur_pretrain)
                    on_preview(preview_img)
                    self._preview_event.clear()

                if (now - last_save_time) >= _SAVE_INTERVAL_SEC:
                    xseg_model._aux_state['loss_history'] = list(self._loss_history)
                    xseg_model.save(iter_count)
                    last_save_time = now
                    if now - last_backup_time >= _BACKUP_INTERVAL_SEC:
                        last_backup_time = now
                        xseg_model.create_backup()
                        _logger.info(f"Auto-backup at iter {iter_count}")
                    if on_preview is not None:
                        preview_img = self._generate_preview(model, device, resolution, cur_pretrain)
                        on_preview(preview_img)
                    if on_save is not None:
                        on_save(iter_count)
        finally:
            xseg_model._aux_state['loss_history'] = list(self._loss_history)
            xseg_model.save(iter_count)
            self._save_xseg_config(model_dir, resolution, face_type, batch_size, learning_rate, pretrain, target_iter, amp_mode, lr_dropout)
            if lock_file.exists():
                lock_file.unlink()
        _logger.info(f"XSeg training completed at iter #{iter_count}")

    def _load_xseg_config(self, model_dir: Path) -> dict:
        config_path = Path(model_dir) / "XSeg_config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_xseg_config(self, model_dir: Path, resolution: int, face_type: str, batch_size: int, lr: float, pretrain: bool, target_iter: int, amp_mode: str, lr_dropout: float):
        config = {
            "resolution": resolution,
            "face_type": face_type,
            "batch_size": batch_size,
            "learning_rate": lr,
            "pretrain": pretrain,
            "target_iter": target_iter,
            "amp_mode": amp_mode,
            "lr_dropout": lr_dropout,
        }
        try:
            (Path(model_dir) / "XSeg_config.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8")
        except Exception as e:
            _logger.warning(f"Failed to save XSeg config: {e}")

    def _generate_preview(
        self,
        model,
        device: torch.device,
        resolution: int,
        is_pretrain: bool = False,
    ) -> np.ndarray:
        model.eval()

        n = self._preview_n

        if is_pretrain:
            pt_ds = getattr(self, '_pretrain_ds', None)
            pt_paths = pt_ds.image_paths if pt_ds else []
            pt_samples = self._sample_images(pt_paths, n)
            n_samples = min(n, len(pt_samples))

            sections = []
            if pt_samples:
                st = []
                for img in pt_samples[:n_samples]:
                    row = self._preview_row_pretrain(model, device, resolution, img)
                    st.append(row)
                if st:
                    sections.append(("XSeg pretrain (gray recon)", np.vstack(st)))
        else:
            train_samples = self._sample_train_images(n)
            src_samples = self._sample_images(self._preview_src_all, n)
            dst_samples = self._sample_images(self._preview_dst_all, n)

            n_samples = min(n, max(len(train_samples), len(src_samples), len(dst_samples)))

            sections = []

            if train_samples:
                st = []
                for img, gt_mask in train_samples[:n_samples]:
                    row = self._preview_row_train(model, device, resolution, img, gt_mask)
                    st.append(row)
                if st:
                    sections.append(("XSeg training faces", np.vstack(st)))

            if src_samples:
                st = []
                for img in src_samples[:n_samples]:
                    row = self._preview_row_infer(model, device, resolution, img)
                    st.append(row)
                if st:
                    sections.append(("XSeg src faces", np.vstack(st)))

            if dst_samples:
                st = []
                for img in dst_samples[:n_samples]:
                    row = self._preview_row_infer(model, device, resolution, img)
                    st.append(row)
                if st:
                    sections.append(("XSeg dst faces", np.vstack(st)))

        model.train()

        if not sections:
            return np.zeros((resolution, resolution * 3, 3), dtype=np.uint8)

        idx = self._preview_page % len(sections)
        name, preview_bgr = sections[idx]
        (h, w, _) = preview_bgr.shape

        head = self._draw_head(w, name, idx, len(sections))
        chart = self._draw_loss_chart(w, 100)

        final = np.vstack([head, chart, preview_bgr])
        return np.clip(final, 0, 255).astype(np.uint8)

    def _sample_train_images(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        paths = self._preview_train_paths
        if not paths:
            return []
        sampled = random.sample(paths, min(n, len(paths)))
        result = []
        r = self._preview_resolution
        for p in sampled:
            img = cv2.imread(str(p))
            if img is None:
                continue
            resized = cv2.resize(img, (r, r), interpolation=cv2.INTER_AREA)
            meta = self._preview_train_meta.get(str(p))
            mask = self._preview_train_render_mask(img.shape[:2], meta)
            mask = cv2.resize(mask, (r, r), interpolation=cv2.INTER_LINEAR)
            result.append((resized, mask))
        return result

    def _sample_images(self, paths: list[Path], n: int) -> list[np.ndarray]:
        if not paths:
            return []
        sampled = random.sample(paths, min(n, len(paths)))
        result = []
        r = self._preview_resolution
        for p in sampled:
            img = cv2.imread(str(p))
            if img is None:
                continue
            result.append(cv2.resize(img, (r, r), interpolation=cv2.INTER_AREA))
        return result

    def _preview_row_pretrain(
        self,
        model,
        device: torch.device,
        resolution: int,
        img: np.ndarray,
    ) -> np.ndarray:
        img_rgb = bgr_to_rgb(img)
        I = img_rgb.astype(np.float32) / 255.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        G = np.stack([gray, gray, gray], axis=-1)
        img_t = torch.from_numpy(I).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            pred = model(img_t, skip_enabled=False, pretrain=True).squeeze().cpu().numpy()
        pred = np.clip(pred, 0.0, 1.0)

        col1 = I
        col2 = pred
        col3 = G

        row = np.concatenate([col1, col2, col3], axis=1)
        row = np.clip(row, 0, 1)
        return rgb_to_bgr((row * 255).astype(np.uint8))

    def _preview_row_train(
        self,
        model,
        device: torch.device,
        resolution: int,
        img: np.ndarray,
        gt_mask: np.ndarray,
    ) -> np.ndarray:
        img_rgb = bgr_to_rgb(img)
        I = img_rgb.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(img_t, skip_enabled=True, pretrain=False).squeeze().cpu()
            pred = torch.sigmoid(logits).numpy()

        M = gt_mask.astype(np.float32) / 255.0
        M3 = np.repeat(M[:, :, np.newaxis], 3, axis=2)
        IM3 = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
        green = np.zeros_like(I)
        green[:, :, 1] = 1.0

        col1 = I * M3 + 0.5 * I * (1 - M3) + 0.5 * green * (1 - M3)
        col2 = IM3
        col3 = I * IM3 + 0.5 * I * (1 - IM3) + 0.5 * green * (1 - IM3)

        row = np.concatenate([col1, col2, col3], axis=1)
        row = np.clip(row, 0, 1)
        return rgb_to_bgr((row * 255).astype(np.uint8))

    def _preview_row_infer(
        self,
        model,
        device: torch.device,
        resolution: int,
        img: np.ndarray,
    ) -> np.ndarray:
        img_rgb = bgr_to_rgb(img)
        I = img_rgb.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(img_t, skip_enabled=True, pretrain=False).squeeze().cpu()
            pred = torch.sigmoid(logits).numpy()

        IM3 = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
        green = np.zeros_like(I)
        green[:, :, 1] = 1.0

        col1 = I
        col2 = IM3
        col3 = I * IM3 + 0.5 * I * (1 - IM3) + 0.5 * green * (1 - IM3)

        row = np.concatenate([col1, col2, col3], axis=1)
        row = np.clip(row, 0, 1)
        return rgb_to_bgr((row * 255).astype(np.uint8))

    def _draw_head(self, width: int, name: str, idx: int, total: int) -> np.ndarray:
        from datetime import datetime
        from faceswap.core.preview_utils import draw_head_bar
        ts = datetime.now().strftime("%H:%M:%S")
        lines = [
            f"[{ts}] [s]:save  [p]:update  [space]:next  [l]:range  [Enter]:stop",
            f'Preview: "{name}" [{idx + 1}/{total}]',
        ]
        return draw_head_bar(width, lines, line_height=20, font_size=14)

    def _draw_loss_chart(self, width: int, height: int) -> np.ndarray:
        from faceswap.core.preview_utils import draw_loss_chart
        return draw_loss_chart(
            width, height, self._loss_history, self._loss_history_range,
            loss_names=["loss"],
            loss_colors=[(255, 200, 0)],
        )

    def apply_trained_mask(
        self,
        aligned_dir: Path,
        model_dir: Path,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> int:
        device = self._resolve_device()
        config = self._load_xseg_config(model_dir)
        resolution = config.get("resolution", 256)

        weight_path = Path(model_dir) / "xseg_net.pth"
        if not weight_path.exists():
            raise FileNotFoundError(f"XSeg model not found: {weight_path}")

        from faceswap.models.xseg_model import XSegNet
        model = XSegNet(resolution=resolution).to(device)
        state = torch.load(str(weight_path), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        aligned_dir = Path(aligned_dir)

        all_paths = FileManager.find_images(aligned_dir)
        total = len(all_paths)
        t0 = time.time()
        count = 0
        for idx, img_path in enumerate(all_paths):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            resized = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
            resized_rgb = bgr_to_rgb(resized)
            img_t = torch.from_numpy(resized_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.inference_mode():
                logits = model(img_t, skip_enabled=True, pretrain=False).squeeze().cpu()
                mask_pred = torch.sigmoid(logits).numpy()

            mask_pred[mask_pred < 0.1] = 0.0
            mask_resized = cv2.resize(mask_pred, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            meta = MetadataManager.load(img_path)
            if meta is None:
                continue
            meta.xseg_mask = FaceMetadata.encode_xseg_mask(mask_uint8)
            MetadataManager.save(img_path, meta)
            count += 1

            if progress_callback is not None:
                elapsed = time.time() - t0
                progress_callback(idx + 1, total, elapsed)

        _logger.info(f"Applied XSeg mask to {count} faces in {aligned_dir}")
        return count

    def remove_trained_mask(self, aligned_dir: Path, progress_callback: Optional[Callable[[int, int, float], None]] = None) -> int:
        aligned_dir = Path(aligned_dir)
        all_paths = FileManager.find_images(aligned_dir)
        total = len(all_paths)
        t0 = time.time()
        count = 0
        for idx, img_path in enumerate(all_paths):
            meta = MetadataManager.load(img_path)
            if meta is None or meta.xseg_mask is None:
                continue
            meta.xseg_mask = None
            MetadataManager.save(img_path, meta)
            count += 1

            if progress_callback is not None:
                elapsed = time.time() - t0
                progress_callback(idx + 1, total, elapsed)

        _logger.info(f"Removed trained masks from {count} faces in {aligned_dir}")
        return count

    def apply_generic_mask(
        self,
        aligned_dir: Path,
        generic_model_dir: Path,
    ) -> int:
        return self.apply_trained_mask(aligned_dir, generic_model_dir)
