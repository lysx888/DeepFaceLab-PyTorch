import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from faceswap.business.if_landmark_dataset import IFLandmarkDataset, FLIP_MAP_106
from faceswap.models.if_landmark.if_landmark_model import IFLandmarkModel, IFLandmarkTrainingConfig
from faceswap.shared.config import auto_select_device, is_gpu_device, get_num_workers
from faceswap.shared.logger import get_logger
from faceswap.shared.torch_config import (
    configure_torch, get_dataloader_config, get_non_blocking, worker_init_fn,
)

_logger = get_logger("if_landmark_trainer")

_SAVE_INTERVAL_SEC = 600
_INPUT_SIZE = 192
_NUM_LANDMARKS = 106
_HALF_SIZE = _INPUT_SIZE // 2


class IFLandmarkTrainer:
    def __init__(self, device: str = "auto") -> None:
        self._device_str = device
        self._device: Optional[torch.device] = None
        self._stop_event = threading.Event()
        self._preview_event = threading.Event()
        self._save_event = threading.Event()
        self._loss_history: list[list[float]] = []
        self._if_model: Optional[IFLandmarkModel] = None
        self._epoch_count = 0

    def _resolve_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        if self._device_str == "auto":
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

    def train(
        self,
        data_dir: Path,
        model_dir: Path,
        batch_size: int = 32,
        learning_rate: float = 0.1,
        max_epochs: int = 30,
        augment: bool = True,
        pretrained_onnx: Optional[str] = None,
        on_epoch: Optional[Callable[[int, float, float], None]] = None,
        on_preview: Optional[Callable[[np.ndarray], None]] = None,
        on_save: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._stop_event.clear()
        self._preview_event.clear()
        self._save_event.clear()
        device = self._resolve_device()

        config = IFLandmarkTrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            augment=augment,
            input_size=_INPUT_SIZE,
            data_dir=str(data_dir),
            pretrained_onnx=pretrained_onnx or '',
        )

        configure_torch("gpu_train" if is_gpu_device(device) else "cpu_train")

        if_model = IFLandmarkModel(config, Path(model_dir), device)
        self._if_model = if_model
        net = if_model.if_net
        optimizer = if_model.get_optimizers_dict()['if_opt']
        scheduler = if_model._scheduler

        start_epoch = if_model.get_aux_state().get('iter_count', 0)
        restored_loss = if_model.get_aux_state().get('loss_history', [])
        if restored_loss:
            self._loss_history = restored_loss

        if pretrained_onnx and not start_epoch:
            try:
                net.load_pretrained_onnx(pretrained_onnx)
            except Exception as e:
                _logger.warning(f"加载预训练权重失败: {e}")

        loss_fn = nn.L1Loss(reduction='mean')

        dataset = IFLandmarkDataset(
            data_dir=Path(data_dir),
            augment=augment,
            input_size=_INPUT_SIZE,
        )

        is_gpu = is_gpu_device(device)
        n_workers = get_num_workers(device)
        dl_cfg = get_dataloader_config(
            "gpu_train" if is_gpu else "cpu_train", dataset_size=len(dataset))

        effective_bs = min(batch_size, len(dataset))
        if effective_bs < batch_size:
            _logger.warning(f"批次大小 {batch_size} > 样本数 {len(dataset)}，自动调整为 {effective_bs}")

        loader = DataLoader(
            dataset,
            batch_size=effective_bs,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=dl_cfg["pin_memory"],
            drop_last=False,
            worker_init_fn=worker_init_fn if n_workers > 0 else None,
            persistent_workers=n_workers > 0,
            prefetch_factor=dl_cfg.get("prefetch_factor"),
        )

        last_save_time = time.time()
        smooth_loss = deque(maxlen=100)

        for epoch in range(start_epoch, max_epochs):
            if self._stop_event.is_set():
                break

            net.train()
            epoch_losses = []
            for batch_idx, batch in enumerate(loader):
                if self._stop_event.is_set():
                    break

                images = batch['image'].to(device, non_blocking=get_non_blocking())
                labels = batch['label'].to(device, non_blocking=get_non_blocking())

                optimizer.zero_grad()
                pred = net(images)
                loss = loss_fn(pred, labels) * 5.0
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                smooth_loss.append(loss_val)

                if self._preview_event.is_set():
                    self._preview_event.clear()
                    if on_preview is not None:
                        preview_img = self._generate_preview(net, dataset, device)
                        on_preview(preview_img)

                if self._save_event.is_set():
                    self._save_event.clear()
                    if_model.get_aux_state()['iter_count'] = epoch
                    if_model.get_aux_state()['loss_history'] = self._loss_history
                    if_model.save(epoch)
                    if on_save is not None:
                        on_save(epoch)
                    last_save_time = time.time()

                now = time.time()
                if now - last_save_time > _SAVE_INTERVAL_SEC:
                    if_model.get_aux_state()['iter_count'] = epoch
                    if_model.get_aux_state()['loss_history'] = self._loss_history
                    if_model.save(epoch)
                    if on_save is not None:
                        on_save(epoch)
                    last_save_time = now

            if self._stop_event.is_set():
                break

            scheduler.step()
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            smooth_avg = float(np.mean(smooth_loss)) if smooth_loss else 0.0
            self._loss_history.append([epoch, avg_loss])
            self._epoch_count = epoch + 1

            current_lr = optimizer.param_groups[0]['lr']
            if on_epoch is not None:
                on_epoch(epoch + 1, avg_loss, current_lr)

            _logger.info(
                f"Epoch {epoch + 1}/{max_epochs}  loss={avg_loss:.6f}  "
                f"smooth={smooth_avg:.6f}  lr={current_lr:.6f}")

            if_model.get_aux_state()['iter_count'] = epoch + 1
            if_model.get_aux_state()['loss_history'] = self._loss_history
            if_model.save(epoch + 1)
            if on_save is not None:
                on_save(epoch + 1)
            last_save_time = time.time()

        if not self._stop_event.is_set():
            onnx_path = Path(model_dir) / "if_landmark_2d106.onnx"
            try:
                if_model.export_onnx(onnx_path)
                _logger.info(f"ONNX model exported to {onnx_path}")
            except Exception as e:
                _logger.warning(f"ONNX export failed: {e}")

    def _generate_preview(
        self,
        net: nn.Module,
        dataset: IFLandmarkDataset,
        device: torch.device,
    ) -> np.ndarray:
        net.eval()
        n_samples = min(8, len(dataset))
        indices = np.random.choice(len(dataset), n_samples, replace=False)

        cols = 4
        rows = (n_samples + cols - 1) // cols
        cell_size = 192
        gap = 4
        canvas_w = cols * (cell_size + gap) + gap
        canvas_h = rows * (cell_size + gap) + gap
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                sample = dataset[idx]
                img_tensor = sample['image'].unsqueeze(0).to(device)
                pred = net(img_tensor)[0].cpu().numpy()

                img = sample['image'].numpy()
                img = img.transpose(1, 2, 0)
                img = (img * 128.0 + 127.5).clip(0, 255).astype(np.uint8)

                pred = pred.reshape(_NUM_LANDMARKS, 2)
                pred[:, 0] += 1.0
                pred[:, 1] += 1.0
                pred *= _HALF_SIZE

                gt = sample['label'].numpy().reshape(_NUM_LANDMARKS, 2)
                gt[:, 0] += 1.0
                gt[:, 1] += 1.0
                gt *= _HALF_SIZE

                vis = img.copy()
                for j in range(_NUM_LANDMARKS):
                    x, y = int(pred[j, 0]), int(pred[j, 1])
                    if 0 <= x < cell_size and 0 <= y < cell_size:
                        cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)
                for j in range(_NUM_LANDMARKS):
                    x, y = int(gt[j, 0]), int(gt[j, 1])
                    if 0 <= x < cell_size and 0 <= y < cell_size:
                        cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)

                row = i // cols
                col = i % cols
                y0 = gap + row * (cell_size + gap)
                x0 = gap + col * (cell_size + gap)
                canvas[y0:y0 + cell_size, x0:x0 + cell_size] = vis

        net.train()
        return canvas

    def generate_loss_chart(self, width: int = 600, height: int = 200) -> np.ndarray:
        if not self._loss_history:
            return np.full((height, width, 3), 30, dtype=np.uint8)

        canvas = np.full((height, width, 3), 30, dtype=np.uint8)
        epochs = [h[0] for h in self._loss_history]
        losses = [h[1] for h in self._loss_history]

        if len(epochs) < 2:
            return canvas

        max_epoch = max(epochs)
        max_loss = max(losses) if losses else 1.0
        min_loss = min(losses) if losses else 0.0
        loss_range = max(max_loss - min_loss, 1e-6)

        margin = 40
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin

        pts = []
        for ep, ls in zip(epochs, losses):
            x = margin + int((ep / max_epoch) * plot_w) if max_epoch > 0 else margin
            y = margin + plot_h - int(((ls - min_loss) / loss_range) * plot_h)
            pts.append((x, y))

        cv2.line(canvas, (margin, margin), (margin, height - margin), (100, 100, 100), 1)
        cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (100, 100, 100), 1)

        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], (0, 200, 0), 2)

        cv2.putText(canvas, f"loss={losses[-1]:.6f}", (margin, margin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"epoch={epochs[-1]}", (width - margin - 80, height - margin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return canvas
