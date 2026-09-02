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
        batch_size: int = 8,
        learning_rate: float = 0.01,
        max_epochs: int = 30,
        augment: bool = True,
        pretrained_onnx: Optional[str] = None,
        deform_aug: bool = False,
        finetune_mode: bool = False,
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
        restored_last_lr = None
        restored_lr = None
        restored_dataset_size = None
        ts_pth = Path(model_dir) / "SCRFD_training_state.json"
        if ts_pth.exists():
            try:
                ts = json.loads(ts_pth.read_text(encoding='utf-8'))
                start_epoch = ts.get('iter', 0)
                restored_lr_steps = ts.get('lr_steps')
                restored_last_lr = ts.get('last_lr')
                restored_lr = ts.get('learning_rate')
                restored_dataset_size = ts.get('dataset_size')
            except Exception:
                pass

        if start_epoch > 0:
            _logger.info(f"续训(resume_from): 从 epoch {start_epoch} 恢复")
        is_finetune = finetune_mode
        effective_lr = learning_rate * 0.1 if is_finetune else learning_rate
        if is_finetune:
            if deform_aug:
                _logger.info(f"微调(非刚性形变): lr {learning_rate} → {effective_lr}, 分阶段+DeformAug")
            else:
                _logger.info(f"微调(刚性): lr {learning_rate} → {effective_lr}, 分阶段训练")

        target_epoch = start_epoch + max_epochs
        if start_epoch > 0 and restored_lr_steps and not all(s <= start_epoch for s in restored_lr_steps):
            lr_steps = restored_lr_steps
            _logger.info(f"续训: 复用lr_steps={lr_steps}")
        else:
            lr_steps = [start_epoch + int(0.67 * max_epochs), start_epoch + int(0.90 * max_epochs)]
            if start_epoch > 0 and restored_lr_steps:
                _logger.info(f"续训: 重置lr_steps={lr_steps} (已过全部衰减点)")

        config = SCRFDTrainingConfig(
            batch_size=batch_size,
            learning_rate=effective_lr,
            max_epochs=max_epochs,
            augment=augment,
            input_size=_INPUT_SIZE,
            data_dir=str(data_dir),
            pretrained_onnx=pretrained_onnx or '',
            lr_step_epochs=lr_steps,
        )

        scrfd_model = SCRFDModel(config, Path(model_dir), device)
        self._scrfd_model = scrfd_model
        net = scrfd_model.scrfd_net

        restored_loss = scrfd_model.get_aux_state().get('loss_history', [])
        if restored_loss:
            self._loss_history = restored_loss
        scrfd_model.get_aux_state()['lr_steps'] = lr_steps
        scrfd_model.get_aux_state()['learning_rate'] = learning_rate

        if is_finetune and start_epoch == 0:
            net.load_pretrained_onnx(pretrained_onnx)

        loss_fn = SCRFDLoss(input_size=_INPUT_SIZE).to(device)

        dataset = SCRFDDataset(
            data_dir=Path(data_dir),
            augment=augment,
            input_size=_INPUT_SIZE,
            is_finetune=is_finetune,
            deform_aug=deform_aug,
        )
        scrfd_model.get_aux_state()['dataset_size'] = len(dataset)

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
        self._loader = loader
        n_batches = len(loader)

        warmup_iters = min(config.warmup_iters, len(dataset) * max_epochs // (batch_size * 3))
        warmup_ratio = config.warmup_ratio
        base_lr = effective_lr
        global_iter = start_epoch * len(loader)

        last_save_time = time.time()
        smooth_loss = deque(maxlen=100)

        if is_finetune:
            _logger.info("微调: layerwise lr, 分阶段训练")

            groups = {'backbone': [], 'neck': [], 'convs': [], 'preds': []}
            for name, param in net.named_parameters():
                if not param.requires_grad:
                    continue
                if 'backbone' in name:
                    groups['backbone'].append(param)
                elif 'neck' in name:
                    groups['neck'].append(param)
                elif 'convs' in name:
                    groups['convs'].append(param)
                else:
                    groups['preds'].append(param)

            lr_mults = {'backbone': 0.01, 'neck': 0.05, 'convs': 0.1, 'preds': 1.0}
            param_groups = []
            for key in ['backbone', 'neck', 'convs', 'preds']:
                if groups[key]:
                    param_groups.append({
                        'params': groups[key],
                        'lr': base_lr * lr_mults[key],
                        'lr_mult': lr_mults[key],
                    })
                    _logger.info(f"  {key}: {len(groups[key])} params, lr_mult={lr_mults[key]}")
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=0.0005)
        else:
            _logger.info("训练: 所有层统一lr, BN正常训练")
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=base_lr, momentum=0.9, weight_decay=0.0005,
            )

        if start_epoch > 0:
            opt_pth = Path(model_dir) / "scrfd_opt.pth"
            if opt_pth.exists():
                try:
                    opt_state = torch.load(opt_pth, map_location=device, weights_only=False)
                    optimizer.load_state_dict(opt_state)
                    _logger.info(f"恢复optimizer状态: {opt_pth.name}")
                except Exception as e:
                    _logger.warning(f"恢复optimizer状态失败: {e}")

        scrfd_model.register_optimizer('scrfd_opt', optimizer)

        _logger.info(f"训练参数: max_epochs={max_epochs}, batch_size={batch_size}, lr={base_lr}, start_epoch={start_epoch}, target_epoch={target_epoch}, lr_steps={lr_steps}")
        if start_epoch > 0:
            _logger.info(f"续训: 从 epoch {start_epoch} 训练 {max_epochs} 轮 (到 epoch {target_epoch})")

        trained_epochs = 0
        if start_epoch == 0:
            try:
                scrfd_model.create_backup()
            except Exception as e:
                _logger.warning(f"训练前自动备份失败: {e}")

        if is_finetune and start_epoch == 0:
            stage1_end = int(max_epochs * 0.4)
        else:
            stage1_end = 0
        stage2_lr = effective_lr * 0.1
        resume_hold_lr = False
        if start_epoch > 0 and is_finetune:
            lr_changed = restored_lr is None or abs(learning_rate - restored_lr) / max(abs(restored_lr), 1e-9) >= 1e-2
            dataset_changed = restored_dataset_size is None or len(dataset) != restored_dataset_size
            if not lr_changed and not dataset_changed and restored_last_lr is not None:
                stage2_lr = restored_last_lr
                resume_hold_lr = True
            elif dataset_changed:
                _logger.info(f"数据集变化({restored_dataset_size}→{len(dataset)}), 从stage2_lr基准重新衰减")
            elif lr_changed:
                _logger.info(f"lr变化({restored_lr}→{learning_rate}), 从stage2_lr基准重新衰减")
        if is_finetune and max_epochs > 0:
            if stage1_end > 0:
                _logger.info(
                    f"分阶段微调: 阶段1(冻结backbone+neck,BN eval,layerwise lr) "
                    f"epoch {start_epoch}-{stage1_end}, "
                    f"阶段2(全解冻,BN eval恒等,lr={stage2_lr:.6f}) epoch {stage1_end}-{target_epoch}")
            else:
                _logger.info(
                    f"微调续训: 直接阶段2(全解冻,BN eval恒等,lr={stage2_lr:.6f}) "
                    f"epoch {start_epoch}-{target_epoch}")

        for epoch in range(start_epoch, target_epoch):
            if self._stop_event.is_set():
                break

            net.train()
            if is_finetune:
                if epoch < stage1_end:
                    for name, param in net.named_parameters():
                        if 'backbone' in name or 'neck' in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                    for m in net.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval()
                    if global_iter >= warmup_iters:
                        decay = 0.1 ** len([s for s in lr_steps if s <= epoch])
                        for pg in optimizer.param_groups:
                            lr_mult = pg.get('lr_mult', 1.0)
                            pg['lr'] = base_lr * lr_mult * decay
                else:
                    if epoch == stage1_end and stage1_end > 0:
                        _logger.info(f"阶段2开始: 解冻全部参数, BN保持eval, lr={stage2_lr:.6f}")
                    for param in net.parameters():
                        param.requires_grad = True
                    for m in net.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.eval()
                    s2_len = max(target_epoch - stage1_end, 1)
                    s2_pos = (epoch - stage1_end) / s2_len
                    if resume_hold_lr:
                        lr_now = stage2_lr
                    else:
                        lr_now = stage2_lr
                        if s2_pos >= 0.8:
                            lr_now = stage2_lr * 0.01
                        elif s2_pos >= 0.5:
                            lr_now = stage2_lr * 0.1
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_now
                    scrfd_model.get_aux_state()['last_lr'] = lr_now
            else:
                if global_iter >= warmup_iters:
                    decay = 0.1 ** len([s for s in lr_steps if s <= epoch])
                    for pg in optimizer.param_groups:
                        lr_mult = pg.get('lr_mult', 1.0)
                        pg['lr'] = base_lr * lr_mult * decay
            epoch_losses = []
            for batch_idx, batch in enumerate(loader):
                if self._stop_event.is_set():
                    break

                images = batch['image'].to(device, non_blocking=get_non_blocking())
                gt_bboxes = [b.to(device) for b in batch['gt_bboxes']]
                gt_labels = [l.to(device) for l in batch['gt_labels']]
                gt_keypointss = [k.to(device) for k in batch['gt_keypointss']]

                if start_epoch == 0 and global_iter < warmup_iters:
                    warmup_factor = (warmup_ratio + (1 - warmup_ratio) * global_iter / warmup_iters)
                    for pg in optimizer.param_groups:
                        lr_mult = pg.get('lr_mult', 1.0)
                        pg['lr'] = base_lr * lr_mult * warmup_factor

                optimizer.zero_grad()
                cls_scores, bbox_preds, kps_preds = net(images)
                losses = loss_fn(cls_scores, bbox_preds, kps_preds,
                                 gt_bboxes, gt_labels, gt_keypointss)
                loss = losses['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                optimizer.step()

                global_iter += 1
                loss_val = loss.item()
                epoch_losses.append(loss_val)
                smooth_loss.append(loss_val)

                if batch_idx % 100 == 0:
                    try:
                        import psutil
                        proc = psutil.Process()
                        children = proc.children(recursive=True)
                        child_mem = sum(c.memory_info().rss for c in children)
                        total_mb = (proc.memory_info().rss + child_mem) / 1024**2
                        _logger.info(
                            f"Epoch {epoch + 1}/{target_epoch}  batch {batch_idx}/{n_batches}  "
                            f"loss={loss_val:.6f}  lr={optimizer.param_groups[-1]['lr']:.6f}  "
                            f"mem={total_mb:.0f}MB(workers={len(children)})")
                    except Exception:
                        _logger.info(
                            f"Epoch {epoch + 1}/{target_epoch}  batch {batch_idx}/{n_batches}  "
                            f"loss={loss_val:.6f}  lr={optimizer.param_groups[-1]['lr']:.6f}")

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

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            smooth_avg = float(np.mean(smooth_loss)) if smooth_loss else 0.0
            self._loss_history.append([epoch, avg_loss])
            self._epoch_count = epoch + 1

            current_lr = optimizer.param_groups[-1]['lr']
            if on_epoch is not None:
                on_epoch(epoch + 1 - start_epoch, avg_loss, current_lr)

            _logger.info(
                f"Epoch {epoch + 1}/{target_epoch}  loss={avg_loss:.6f}  "
                f"smooth={smooth_avg:.6f}  lr={current_lr:.6f}")

            scrfd_model.get_aux_state()['iter_count'] = epoch + 1
            scrfd_model.get_aux_state()['loss_history'] = self._loss_history
            scrfd_model.save(epoch + 1)
            if on_save is not None:
                on_save(epoch + 1)
            last_save_time = time.time()
            trained_epochs += 1

        if not self._stop_event.is_set() and trained_epochs > 0:
            onnx_path = Path(model_dir) / "scrfd_custom.onnx"
            try:
                scrfd_model.export_onnx(onnx_path)
                _logger.info(f"ONNX model exported to {onnx_path}")
            except Exception as e:
                _logger.warning(f"ONNX export failed: {e}")
        elif trained_epochs == 0:
            _logger.warning(f"训练0轮，跳过ONNX导出。当前epoch={start_epoch}，如需继续训练请保持max_epochs={max_epochs}。")

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
        net: SCRFDNet,
        dataset: SCRFDDataset,
        device: torch.device,
    ) -> np.ndarray:
        net.eval()
        n_samples = min(4, len(dataset))
        indices = np.random.choice(len(dataset), n_samples, replace=False)

        bn_eval_modules = [m for m in net.modules()
                           if isinstance(m, nn.BatchNorm2d) and not m.training]

        _PREVIEW_SIZE = 320
        cell_size = _PREVIEW_SIZE
        gap = 4
        cols = n_samples
        rows = 1
        canvas_w = cols * (cell_size + gap) + gap
        canvas_h = rows * (cell_size + gap) + gap
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                img_path, bbox, kps = dataset._samples[idx]
                img = dataset._read_image(img_path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                gx1, gy1, gx2, gy2 = bbox
                cx = (gx1 + gx2) / 2.0
                cy = (gy1 + gy2) / 2.0
                fs = int(max(gx2 - gx1, gy2 - gy1, 20) * 1.4)
                half = int(fs / 2)
                x_lo = int(cx) - half
                y_lo = int(cy) - half
                x_hi = x_lo + fs
                y_hi = y_lo + fs
                sx = max(0, x_lo)
                sy = max(0, y_lo)
                ex = min(w, x_hi)
                ey = min(h, y_hi)
                crop = img[sy:ey, sx:ex].copy()
                pad_l = sx - x_lo
                pad_t = sy - y_lo
                pad_r = x_hi - ex
                pad_b = y_hi - ey
                if pad_l or pad_t or pad_r or pad_b:
                    crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                             cv2.BORDER_CONSTANT, value=0)
                display = cv2.resize(crop, (cell_size, cell_size))
                padded = display

                def _to_preview(pts_xy):
                    pts_xy = np.asarray(pts_xy, dtype=np.float32).copy()
                    pts_xy[:, 0] = (pts_xy[:, 0] - x_lo) * (cell_size / fs)
                    pts_xy[:, 1] = (pts_xy[:, 1] - y_lo) * (cell_size / fs)
                    return pts_xy

                gt_bb = _to_preview([[gx1, gy1], [gx2, gy2]])
                cv2.rectangle(padded, (int(gt_bb[0, 0]), int(gt_bb[0, 1])),
                              (int(gt_bb[1, 0]), int(gt_bb[1, 1])), (0, 255, 0), 2)
                gt_kps = _to_preview(kps)
                for k in range(5):
                    cv2.circle(padded, (int(gt_kps[k, 0]), int(gt_kps[k, 1])), 3, (0, 255, 0), -1)

                dets, kpss = scrfd_detect(net, img, input_size=_INPUT_SIZE,
                                         det_thresh=0.5, nms_thresh=0.45)
                for j in range(len(dets)):
                    x1, y1, x2, y2, score = dets[j]
                    if x2 < x_lo or x1 > x_hi or y2 < y_lo or y1 > y_hi:
                        continue
                    det_bb = _to_preview([[x1, y1], [x2, y2]])
                    cv2.rectangle(padded, (int(det_bb[0, 0]), int(det_bb[0, 1])),
                                  (int(det_bb[1, 0]), int(det_bb[1, 1])), (0, 0, 255), 2)
                    cv2.putText(padded, f"{score:.2f}", (int(det_bb[0, 0]), max(int(det_bb[0, 1]) - 5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    if j < len(kpss):
                        det_kps = _to_preview(kpss[j])
                        for k in range(5):
                            cv2.circle(padded, (int(det_kps[k, 0]), int(det_kps[k, 1])),
                                       3, (0, 0, 255), -1)

                col = i % cols
                x0 = gap + col * (cell_size + gap)
                y0 = gap
                canvas[y0:y0 + cell_size, x0:x0 + cell_size] = padded

        net.train()
        for m in bn_eval_modules:
            m.eval()
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
