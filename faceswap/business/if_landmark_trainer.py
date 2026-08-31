import gc
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
        self._loader: Optional[DataLoader] = None

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
        if self._loader is not None:
            try:
                it = getattr(self._loader, '_iterator', None)
                if it is not None:
                    it._shutdown_workers()
            except Exception:
                pass

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
        loss_type: str = "wing",
        on_epoch: Optional[Callable[[int, float, float], None]] = None,
        on_preview: Optional[Callable[[np.ndarray], None]] = None,
        on_save: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._stop_event.clear()
        self._preview_event.clear()
        self._save_event.clear()
        device = self._resolve_device()

        configure_torch("gpu_train" if is_gpu_device(device) else "cpu_train")

        start_epoch = 0
        restored_lr_steps = None
        ts_pth = Path(model_dir) / "IFLandmark_training_state.json"
        if ts_pth.exists():
            try:
                ts = json.loads(ts_pth.read_text(encoding='utf-8'))
                start_epoch = ts.get('iter', 0)
                restored_lr_steps = ts.get('lr_steps')
            except Exception:
                pass

        lr_steps = [
            start_epoch + int(0.50 * max_epochs),
            start_epoch + int(0.83 * max_epochs),
            start_epoch + int(0.93 * max_epochs),
        ]

        config = IFLandmarkTrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            augment=augment,
            input_size=_INPUT_SIZE,
            data_dir=str(data_dir),
            pretrained_onnx=pretrained_onnx or '',
            lr_steps=lr_steps,
        )

        if_model = IFLandmarkModel(config, Path(model_dir), device)
        self._if_model = if_model
        net = if_model.if_net
        optimizer = if_model.get_optimizers_dict()['if_opt']
        scheduler = if_model._scheduler

        if start_epoch > 0 and restored_lr_steps:
            config.lr_steps = restored_lr_steps
            for pg in optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * (0.1 ** len([m for m in restored_lr_steps if m <= scheduler.last_epoch]))
            _logger.info(f"续训: 复用上次lr_steps={restored_lr_steps}")
        if_model.get_aux_state()['lr_steps'] = config.lr_steps

        restored_loss = if_model.get_aux_state().get('loss_history', [])
        if restored_loss:
            self._loss_history = restored_loss

        if pretrained_onnx and not start_epoch:
            try:
                net.load_pretrained_onnx(pretrained_onnx)
            except Exception as e:
                _logger.warning(f"加载预训练权重失败: {e}")

        from faceswap.models.if_landmark.if_landmark_loss import IFLandmarkLoss
        loss_fn = IFLandmarkLoss(loss_type=loss_type, warmup_ratio=0.2).to(device)

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
        self._loader = loader
        n_batches = len(loader)

        last_save_time = time.time()
        smooth_loss = deque(maxlen=100)

        _logger.info(f"训练参数: max_epochs={max_epochs}, batch_size={batch_size}, lr={learning_rate}, start_epoch={start_epoch}, lr_steps={lr_steps}")
        if start_epoch > 0:
            _logger.info(f"续训: 从第 {start_epoch} 轮开始，额外训练 {max_epochs} 轮 (到第 {start_epoch + max_epochs} 轮)")

        for epoch in range(start_epoch, start_epoch + max_epochs):
            if self._stop_event.is_set():
                break

            net.train()
            epoch_losses = []
            for batch_idx, batch in enumerate(loader):
                if self._stop_event.is_set():
                    break

                images = batch['image'].to(device, non_blocking=get_non_blocking())
                labels = batch['label'].to(device, non_blocking=get_non_blocking())
                visibles = batch['visible'].to(device, non_blocking=get_non_blocking())

                optimizer.zero_grad()
                pred = net(images)
                progress = epoch / max(start_epoch + max_epochs - 1, 1)
                loss = loss_fn(pred, labels, progress=progress, visible=visibles)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                smooth_loss.append(loss_val)

                if batch_idx % 100 == 0:
                    vis_rate = visibles.mean().item()
                    try:
                        import psutil
                        proc = psutil.Process()
                        children = proc.children(recursive=True)
                        child_mem = sum(c.memory_info().rss for c in children)
                        total_mb = (proc.memory_info().rss + child_mem) / 1024**2
                        _logger.info(
                            f"Epoch {epoch + 1 - start_epoch}/{max_epochs}  batch {batch_idx}/{n_batches}  "
                            f"loss={loss_val:.6f}  lr={optimizer.param_groups[0]['lr']:.6f}  "
                            f"vis={vis_rate:.2%}  "
                            f"mem={total_mb:.0f}MB(workers={len(children)})")
                    except Exception:
                        _logger.info(
                            f"Epoch {epoch + 1 - start_epoch}/{max_epochs}  batch {batch_idx}/{n_batches}  "
                            f"loss={loss_val:.6f}  lr={optimizer.param_groups[0]['lr']:.6f}  "
                            f"vis={vis_rate:.2%}")

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

            if self._preview_event.is_set():
                self._preview_event.clear()
                if on_preview is not None:
                    preview_img = self._generate_preview(net, dataset, device)
                    on_preview(preview_img)

            scheduler.step()
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            smooth_avg = float(np.mean(smooth_loss)) if smooth_loss else 0.0
            self._loss_history.append([epoch, avg_loss])
            self._epoch_count = epoch + 1

            current_lr = optimizer.param_groups[0]['lr']
            if on_epoch is not None:
                on_epoch(epoch + 1 - start_epoch, avg_loss, current_lr)

            _logger.info(
                f"Epoch {epoch + 1 - start_epoch}/{max_epochs}  loss={avg_loss:.6f}  "
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

        try:
            it = getattr(loader, '_iterator', None)
            if it is not None:
                it._shutdown_workers()
        except Exception:
            pass
        self._loader = None
        gc.collect()
        time.sleep(1)

    def _generate_preview(
        self,
        net: nn.Module,
        dataset: IFLandmarkDataset,
        device: torch.device,
    ) -> np.ndarray:
        was_training = net.training
        net.eval()
        n_samples = min(4, len(dataset))
        indices = np.random.choice(len(dataset), n_samples, replace=False)

        cols = n_samples
        rows = 1
        cell_size = _INPUT_SIZE
        gap = 4
        canvas_w = cols * (cell_size + gap) + gap
        canvas_h = rows * (cell_size + gap) + gap
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                img_path, landmarks, bbox, gt_visible = dataset._samples[idx]
                img = dataset._read_image(img_path)
                if img is None:
                    continue

                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                max_wh = max(w, h)
                if max_wh < 1e-6:
                    continue
                center = np.array([(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2], dtype=np.float32)
                scale = _INPUT_SIZE / (max_wh * 1.5)
                M = np.array([
                    [scale, 0.0, _INPUT_SIZE / 2.0 - center[0] * scale],
                    [0.0, scale, _INPUT_SIZE / 2.0 - center[1] * scale],
                ], dtype=np.float32)

                aligned = cv2.warpAffine(img, M, (cell_size, cell_size),
                                         flags=cv2.INTER_LINEAR, borderValue=0)

                gt_a = np.zeros_like(landmarks)
                gt_a[:, 0] = M[0, 0] * landmarks[:, 0] + M[0, 1] * landmarks[:, 1] + M[0, 2]
                gt_a[:, 1] = M[1, 0] * landmarks[:, 0] + M[1, 1] * landmarks[:, 1] + M[1, 2]

                inp = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
                inp = (inp - 127.5) / 128.0
                inp_t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
                pred = net(inp_t)[0].cpu().numpy().reshape(_NUM_LANDMARKS, 2)
                pred[:, 0] += 1.0
                pred[:, 1] += 1.0
                pred *= _HALF_SIZE

                for j in range(_NUM_LANDMARKS):
                    x, y = int(pred[j, 0]), int(pred[j, 1])
                    if 0 <= x < cell_size and 0 <= y < cell_size:
                        cv2.circle(aligned, (x, y), 1, (0, 255, 0), -1)
                for j in range(_NUM_LANDMARKS):
                    x, y = int(gt_a[j, 0]), int(gt_a[j, 1])
                    if 0 <= x < cell_size and 0 <= y < cell_size:
                        color = (0, 0, 255) if gt_visible[j] else (128, 128, 128)
                        cv2.circle(aligned, (x, y), 1, color, -1)

                col = i % cols
                x0 = gap + col * (cell_size + gap)
                y0 = gap
                canvas[y0:y0 + cell_size, x0:x0 + cell_size] = aligned

        if was_training:
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
