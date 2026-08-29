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

from faceswap.business.scrfd_dataset import SCRFDDataset
from faceswap.models.scrfd.scrfd_arch import SCRFDNet, scrfd_detect
from faceswap.models.scrfd.scrfd_loss import SCRFDLoss
from faceswap.models.scrfd.scrfd_model import SCRFDModel, SCRFDTrainingConfig
from faceswap.shared.config import auto_select_device, is_gpu_device, get_num_workers
from faceswap.shared.logger import get_logger
from faceswap.shared.torch_config import (
    configure_torch, get_dataloader_config, get_non_blocking, worker_init_fn,
)

_logger = get_logger("scrfd_trainer")

_SAVE_INTERVAL_SEC = 600
_INPUT_SIZE = 640


class SCRFDTrainer:
    def __init__(self, device: str = "auto") -> None:
        self._device_str = device
        self._device: Optional[torch.device] = None
        self._stop_event = threading.Event()
        self._preview_event = threading.Event()
        self._save_event = threading.Event()
        self._loss_history: list[list[float]] = []
        self._scrfd_model: Optional[SCRFDModel] = None
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
        batch_size: int = 8,
        learning_rate: float = 0.01,
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

        config = SCRFDTrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            augment=augment,
            input_size=_INPUT_SIZE,
            data_dir=str(data_dir),
        )

        configure_torch("gpu_train" if is_gpu_device(device) else "cpu_train")

        scrfd_model = SCRFDModel(config, Path(model_dir), device)
        self._scrfd_model = scrfd_model
        net = scrfd_model.scrfd_net

        if pretrained_onnx and not scrfd_model.get_aux_state().get('iter_count', 0):
            try:
                net.load_pretrained_onnx(pretrained_onnx)
            except Exception as e:
                _logger.warning(f"Failed to load pretrained ONNX: {e}")
        optimizer = scrfd_model.get_optimizers_dict()['scrfd_opt']
        scheduler = scrfd_model._scheduler

        start_epoch = scrfd_model.get_aux_state().get('iter_count', 0)
        restored_loss = scrfd_model.get_aux_state().get('loss_history', [])
        if restored_loss:
            self._loss_history = restored_loss

        loss_fn = SCRFDLoss(input_size=_INPUT_SIZE).to(device)

        dataset = SCRFDDataset(
            data_dir=Path(data_dir),
            augment=augment,
            input_size=_INPUT_SIZE,
        )

        is_gpu = is_gpu_device(device)
        n_workers = get_num_workers(device)
        dl_cfg = get_dataloader_config(
            "gpu_train" if is_gpu else "cpu_train", dataset_size=len(dataset))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=dl_cfg["pin_memory"],
            drop_last=False,
            collate_fn=_scrfd_collate_fn,
            worker_init_fn=worker_init_fn if n_workers > 0 else None,
            persistent_workers=n_workers > 0,
            prefetch_factor=dl_cfg.get("prefetch_factor"),
        )

        warmup_iters = config.warmup_iters
        warmup_ratio = config.warmup_ratio
        base_lr = learning_rate
        global_iter = 0

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
                gt_bboxes = [b.to(device) for b in batch['gt_bboxes']]
                gt_labels = [l.to(device) for l in batch['gt_labels']]
                gt_keypointss = [k.to(device) for k in batch['gt_keypointss']]

                if global_iter < warmup_iters:
                    warmup_lr = base_lr * (warmup_ratio + (1 - warmup_ratio) * global_iter / warmup_iters)
                    for pg in optimizer.param_groups:
                        pg['lr'] = warmup_lr

                optimizer.zero_grad()
                cls_scores, bbox_preds, kps_preds = net(images)
                losses = loss_fn(cls_scores, bbox_preds, kps_preds,
                                 gt_bboxes, gt_labels, gt_keypointss)
                loss = losses['loss']
                loss.backward()
                optimizer.step()

                global_iter += 1
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
                    scrfd_model.get_aux_state()['iter_count'] = epoch
                    scrfd_model.get_aux_state()['loss_history'] = self._loss_history
                    scrfd_model.save(epoch)
                    if on_save is not None:
                        on_save(epoch)
                    last_save_time = time.time()

                now = time.time()
                if now - last_save_time > _SAVE_INTERVAL_SEC:
                    scrfd_model.get_aux_state()['iter_count'] = epoch
                    scrfd_model.get_aux_state()['loss_history'] = self._loss_history
                    scrfd_model.save(epoch)
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

            scrfd_model.get_aux_state()['iter_count'] = epoch + 1
            scrfd_model.get_aux_state()['loss_history'] = self._loss_history
            scrfd_model.save(epoch + 1)
            if on_save is not None:
                on_save(epoch + 1)
            last_save_time = time.time()

        if not self._stop_event.is_set():
            onnx_path = Path(model_dir) / "scrfd_custom.onnx"
            try:
                scrfd_model.export_onnx(onnx_path)
                _logger.info(f"ONNX model exported to {onnx_path}")
            except Exception as e:
                _logger.warning(f"ONNX export failed: {e}")

    def _generate_preview(
        self,
        net: SCRFDNet,
        dataset: SCRFDDataset,
        device: torch.device,
    ) -> np.ndarray:
        net.eval()
        n_samples = min(4, len(dataset))
        indices = np.random.choice(len(dataset), n_samples, replace=False)

        cell_size = _INPUT_SIZE
        gap = 4
        cols = 2
        rows = (n_samples + cols - 1) // cols
        canvas_w = cols * (cell_size + gap) + gap
        canvas_h = rows * (cell_size + gap) + gap
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                sample = dataset[idx]
                img_tensor = sample['image'].unsqueeze(0).to(device)

                img = sample['image'].numpy().transpose(1, 2, 0)
                img = (img * 128.0 + 127.5).clip(0, 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                gt_bboxes = sample['gt_bboxes'].numpy()
                gt_kps = sample['gt_keypointss'].numpy()

                for j in range(len(gt_bboxes)):
                    x1, y1, x2, y2 = gt_bboxes[j].astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for k in range(5):
                        px, py = int(gt_kps[j, k, 0]), int(gt_kps[j, k, 1])
                        cv2.circle(img, (px, py), 3, (0, 255, 0), -1)

                dets, kpss = scrfd_detect(net, img, input_size=_INPUT_SIZE,
                                         det_thresh=0.5, nms_thresh=0.4)
                for j in range(len(dets)):
                    x1, y1, x2, y2, score = dets[j]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if j < len(kpss):
                        for k in range(5):
                            px, py = int(kpss[j, k, 0]), int(kpss[j, k, 1])
                            cv2.circle(img, (px, py), 3, (0, 0, 255), -1)

                row = i // cols
                col = i % cols
                y0 = gap + row * (cell_size + gap)
                x0 = gap + col * (cell_size + gap)
                canvas[y0:y0 + cell_size, x0:x0 + cell_size] = img

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


def _scrfd_collate_fn(batch):
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'gt_bboxes': [b['gt_bboxes'] for b in batch],
        'gt_labels': [b['gt_labels'] for b in batch],
        'gt_keypointss': [b['gt_keypointss'] for b in batch],
        'image_path': [b['image_path'] for b in batch],
    }
