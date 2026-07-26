import io
import itertools
import json
import random
import threading
import time
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, RandomSampler

from DeepFaceLab.core.metadata_manager import MetadataManager
from DeepFaceLab.models.xseg_model import XSegNet
from DeepFaceLab.models.faceset_dataset import FacesetDataset
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger
from DeepFaceLab.shared.torch_config import get_dataloader_config, get_non_blocking, worker_init_fn

_logger = get_logger("xseg_trainer")

_MODEL_PREFIX = "XSeg"
_RESOLUTION = 256
_PREVIEW_INTERVAL_SEC = 60
_SAVE_INTERVAL_SEC = 1200


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
        self._model = None
        self._optimizer = None
        self._iter_count = 0
        self._model_dir = None
        self._preview_train_paths: list[Path] = []
        self._preview_train_meta: dict = {}
        self._preview_train_render_mask = None
        self._preview_src_all: list[Path] = []
        self._preview_dst_all: list[Path] = []
        self._preview_n: int = 3
        self._preview_resolution = _RESOLUTION

    def _resolve_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        if self._device_str == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        on_iter: Optional[Callable[[int, float, float], None]] = None,
        on_preview: Optional[Callable[[np.ndarray], None]] = None,
        on_save: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._stop_event.clear()
        self._preview_event.clear()
        self._save_event.clear()
        self._loss_history = []
        device = self._resolve_device()
        resolution = _RESOLUTION

        src_ds = None
        dst_ds = None
        for d in [Path(src_aligned_dir), Path(dst_aligned_dir)]:
            if not d.exists():
                continue
            try:
                ds = FacesetDataset(d, resolution=resolution, augment=True)
                if src_ds is None:
                    src_ds = ds
                else:
                    dst_ds = ds
            except ValueError:
                pass

        if src_ds is None and dst_ds is None:
            raise ValueError("No annotated faces found. Please annotate faces with XSeg editor first.")

        datasets = []
        for ds in [src_ds, dst_ds]:
            if ds is not None:
                datasets.append(ds)

        combined = FacesetDataset.merge(datasets)
        combined_paths = combined._image_paths
        combined_meta = combined._metadata_cache

        dl_cfg = get_dataloader_config("gpu_train" if device.type == "cuda" else "cpu_train", dataset_size=len(combined))
        steps_per_epoch = max(len(combined), batch_size * 50)
        loader = DataLoader(
            combined,
            batch_size=batch_size,
            sampler=RandomSampler(combined, replacement=True, num_samples=steps_per_epoch),
            num_workers=dl_cfg["num_workers"],
            pin_memory=dl_cfg["pin_memory"],
            drop_last=True,
            worker_init_fn=worker_init_fn if dl_cfg["num_workers"] > 0 else None,
        )

        model = XSegNet(resolution=resolution).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        criterion = torch.nn.BCEWithLogitsLoss()

        start_iter = self._load_checkpoint(model, optimizer, model_dir)

        _N_PREVIEW = min(4, batch_size, 800 // resolution)

        _logger.info(f"XSeg training started: {len(combined)} annotated images, target {target_iter} iters, device={device}")
        _logger.info(f"Model Options: face_type={face_type}, batch_size={batch_size}, lr={learning_rate}, resolution={resolution}")
        _logger.info('Press "Stop" to stop training and save model.')

        self._preview_train_paths = combined_paths
        self._preview_train_meta = combined_meta
        self._preview_train_render_mask = combined._render_mask
        self._preview_src_all = FileManager.find_images(Path(src_aligned_dir)) if Path(src_aligned_dir).exists() else []
        dst_dir = Path(dst_aligned_dir) if dst_aligned_dir is not None and Path(dst_aligned_dir).exists() else Path(src_aligned_dir)
        self._preview_dst_all = FileManager.find_images(dst_dir)
        self._preview_n = _N_PREVIEW
        self._preview_resolution = resolution

        last_preview_time = 0.0
        last_save_time = time.time()
        iter_count = start_iter
        self._model = model
        self._optimizer = optimizer
        self._iter_count = iter_count
        self._model_dir = model_dir

        for epoch in itertools.count():
            if self._stop_event.is_set() or iter_count >= target_iter:
                break

            model.train()
            for batch in loader:
                if self._stop_event.is_set() or iter_count >= target_iter:
                    break

                t0 = time.time()
                imgs = batch["image"].to(device, non_blocking=get_non_blocking())
                masks = batch["mask"].to(device, non_blocking=get_non_blocking())

                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    pred = model(imgs)
                    loss = criterion(pred, masks)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                iter_count += 1
                self._iter_count = iter_count
                iter_ms = (time.time() - t0) * 1000
                loss_val = loss.item()
                self._loss_history.append((iter_count, loss_val))

                if on_iter is not None:
                    on_iter(iter_count, loss_val, iter_ms)

                now = time.time()

                if self._save_event.is_set():
                    self._save_checkpoint(model, optimizer, iter_count, model_dir)
                    last_save_time = now
                    self._save_event.clear()
                    self._preview_event.set()
                    if on_save is not None:
                        on_save(iter_count)

                need_preview = (now - last_preview_time) >= _PREVIEW_INTERVAL_SEC or self._preview_event.is_set()
                if on_preview is not None and need_preview:
                    preview_img = self._generate_preview(model, device, resolution)
                    on_preview(preview_img)
                    last_preview_time = now
                    self._preview_event.clear()

                if (now - last_save_time) >= _SAVE_INTERVAL_SEC:
                    self._save_checkpoint(model, optimizer, iter_count, model_dir)
                    last_save_time = now
                    self._preview_event.set()
                    if on_save is not None:
                        on_save(iter_count)

        self._save_checkpoint(model, optimizer, iter_count, model_dir)
        self._save_final(model, model_dir, resolution, face_type, batch_size, learning_rate)
        if on_save is not None:
            on_save(iter_count)
        _logger.info(f"XSeg training completed at iter #{iter_count}")

    def _generate_preview(
        self,
        model: XSegNet,
        device: torch.device,
        resolution: int,
    ) -> np.ndarray:
        model.eval()

        n = self._preview_n
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
        chart = self._draw_loss_chart(w, 100) if len(self._loss_history) > 2 else np.zeros((100, w, 3), dtype=np.float32)

        final = np.vstack([head, chart, preview_bgr])
        return (np.clip(final, 0, 1) * 255).astype(np.uint8)

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
            resized = cv2.resize(img, (r, r))
            meta_key = str(p)
            meta = self._preview_train_meta.get(meta_key) or self._preview_train_meta.get(p.name)
            mask = self._preview_train_render_mask(img.shape[:2], meta)
            mask = cv2.resize(mask, (r, r))
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
            result.append(cv2.resize(img, (r, r)))
        return result

    def _preview_row_train(
        self,
        model: XSegNet,
        device: torch.device,
        resolution: int,
        img: np.ndarray,
        gt_mask: np.ndarray,
    ) -> np.ndarray:
        I = img.astype(np.float32) / 255.0
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        img_t = img_t * 2.0 - 1.0
        with torch.inference_mode():
            logits = model(img_t).squeeze().cpu()
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
        return np.clip(row, 0, 1)

    def _preview_row_infer(
        self,
        model: XSegNet,
        device: torch.device,
        resolution: int,
        img: np.ndarray,
    ) -> np.ndarray:
        I = img.astype(np.float32) / 255.0
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        img_t = img_t * 2.0 - 1.0
        with torch.inference_mode():
            logits = model(img_t).squeeze().cpu()
            pred = torch.sigmoid(logits).numpy()

        IM3 = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
        green = np.zeros_like(I)
        green[:, :, 1] = 1.0

        col1 = I
        col2 = IM3
        col3 = I * IM3 + 0.5 * I * (1 - IM3) + 0.5 * green * (1 - IM3)

        row = np.concatenate([col1, col2, col3], axis=1)
        return np.clip(row, 0, 1)

    def _draw_head(self, width: int, name: str, idx: int, total: int) -> np.ndarray:
        from datetime import datetime
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

    def apply_trained_mask(
        self,
        aligned_dir: Path,
        model_dir: Path,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> int:
        device = self._resolve_device()
        config = self._load_config(model_dir)
        resolution = config.get("resolution", _RESOLUTION)

        weight_path = Path(model_dir) / f"{_MODEL_PREFIX}.pt"
        if not weight_path.exists():
            raise FileNotFoundError(f"XSeg model not found: {weight_path}")

        model = XSegNet(resolution=resolution).to(device)
        state = torch.load(str(weight_path), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        aligned_dir = Path(aligned_dir)
        self._backup_annotated(aligned_dir)

        all_paths = FileManager.find_images(aligned_dir)
        total = len(all_paths)
        t0 = time.time()
        count = 0
        for idx, img_path in enumerate(all_paths):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            resized = cv2.resize(img, (resolution, resolution))
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(resized_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            img_t = img_t * 2.0 - 1.0

            with torch.inference_mode():
                logits = model(img_t).squeeze().cpu()
                mask_pred = torch.sigmoid(logits).numpy()

            mask_pred[mask_pred < 0.1] = 0.0
            mask_resized = cv2.resize(mask_pred, (w, h))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)

            meta = MetadataManager.load(img_path)
            if meta is None:
                continue
            meta.seg_ie_polys = self._mask_to_polys(mask_uint8)
            MetadataManager.save(img_path, meta)
            count += 1

            if progress_callback is not None:
                elapsed = time.time() - t0
                progress_callback(idx + 1, total, elapsed)

        _logger.info(f"Applied XSeg mask to {count} faces in {aligned_dir}")
        return count

    def _backup_annotated(self, aligned_dir: Path) -> int:
        import shutil
        aligned_dir = Path(aligned_dir)
        backup_dir = aligned_dir.parent / f"{aligned_dir.name}_xseg_backup"

        if backup_dir.exists():
            for f in backup_dir.iterdir():
                if f.is_file():
                    f.unlink()

        all_paths = FileManager.find_images(aligned_dir)
        count = 0
        for img_path in all_paths:
            meta = MetadataManager.load(img_path)
            if meta is None or meta.seg_ie_polys is None:
                continue
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(img_path), str(backup_dir / img_path.name))
            json_path = img_path.parent / f"{img_path.stem}.json"
            if json_path.exists():
                shutil.copy2(str(json_path), str(backup_dir / json_path.name))
            count += 1
        if count > 0:
            _logger.info(f"Backed up {count} hand-annotated faces to {backup_dir}")
        return count

    def remove_trained_mask(self, aligned_dir: Path, progress_callback: Optional[Callable[[int, int, float], None]] = None) -> int:
        import shutil
        aligned_dir = Path(aligned_dir)
        backup_dir = aligned_dir.parent / f"{aligned_dir.name}_xseg_backup"

        if backup_dir.exists():
            restored = 0
            for backup_json in backup_dir.glob("*.json"):
                dst_json = aligned_dir / backup_json.name
                if dst_json.exists():
                    shutil.copy2(str(backup_json), str(dst_json))
                    restored += 1
            for backup_img in FileManager.find_images(backup_dir):
                dst_img = aligned_dir / backup_img.name
                if dst_img.exists():
                    shutil.copy2(str(backup_img), str(dst_img))
                    restored += 1
            if restored > 0:
                _logger.info(f"Restored {restored} backup files from {backup_dir} to {aligned_dir}")

        all_paths = FileManager.find_images(aligned_dir)
        total = len(all_paths)
        t0 = time.time()
        count = 0
        for idx, img_path in enumerate(all_paths):
            meta = MetadataManager.load(img_path)
            if meta is None or meta.seg_ie_polys is None:
                continue
            meta.seg_ie_polys = None
            MetadataManager.save(img_path, meta)
            count += 1

            if progress_callback is not None:
                elapsed = time.time() - t0
                progress_callback(idx + 1, total, elapsed)

        _logger.info(f"Removed XSeg masks from {count} faces in {aligned_dir}")
        return count

    def apply_generic_mask(
        self,
        aligned_dir: Path,
        generic_model_dir: Path,
    ) -> int:
        return self.apply_trained_mask(aligned_dir, generic_model_dir)

    def _save_checkpoint(self, model: XSegNet, optimizer: torch.optim.Optimizer, iteration: int, model_dir: Path):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        torch.save({
            "iter": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, buf)
        FileManager.atomic_write(model_dir / f"{_MODEL_PREFIX}_ckpt.pt", buf.getvalue())

    def _load_checkpoint(self, model: XSegNet, optimizer: torch.optim.Optimizer, model_dir: Path) -> int:
        ckpt_path = Path(model_dir) / f"{_MODEL_PREFIX}_ckpt.pt"
        if not ckpt_path.exists():
            return 0
        try:
            data = open(str(ckpt_path), "rb").read()
            ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            iteration = ckpt.get("iter", 0)
            _logger.info(f"Resumed XSeg training from iter #{iteration}")
            return iteration
        except Exception as e:
            _logger.warning(f"Failed to load checkpoint: {e}")
            return 0

    def _save_final(self, model: XSegNet, model_dir: Path, resolution: int, face_type: str, batch_size: int, lr: float):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        FileManager.atomic_write(model_dir / f"{_MODEL_PREFIX}.pt", buf.getvalue())
        config = {
            "resolution": resolution,
            "face_type": face_type,
            "batch_size": batch_size,
            "learning_rate": lr,
        }
        FileManager.atomic_write(model_dir / f"{_MODEL_PREFIX}_config.json", json.dumps(config, indent=2))
        _logger.info(f"XSeg model saved to {model_dir}")

    def _load_config(self, model_dir: Path) -> dict:
        config_path = Path(model_dir) / f"{_MODEL_PREFIX}_config.json"
        if config_path.exists():
            try:
                return json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    @staticmethod
    def _mask_to_polys(mask_uint8: np.ndarray, epsilon: float = 2.0, min_area: int = 100) -> list[dict]:
        h, w = mask_uint8.shape[:2]
        binary = ((mask_uint8 > 64) * 255).astype(np.uint8)
        points = cv2.findNonZero(binary)
        if points is None:
            return []
        hull = cv2.convexHull(points)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(hull_mask, [hull], 255)
        exclude_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(binary))
        result = []
        approx_hull = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx_hull) >= 3:
            pts = [[float(min(pt[0][0], w - 1)), float(min(pt[0][1], h - 1))] for pt in approx_hull]
            result.append({"type": 1, "pts": pts})
        contours, _ = cv2.findContours(exclude_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            pts = [[float(min(pt[0][0], w - 1)), float(min(pt[0][1], h - 1))] for pt in approx]
            result.append({"type": 0, "pts": pts})
        return result
