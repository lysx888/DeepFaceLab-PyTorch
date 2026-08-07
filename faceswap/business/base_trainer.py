import gc
import os
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from faceswap.models.base_model import BaseModel
from faceswap.shared.logger import get_logger

_logger = get_logger("base_trainer")


@dataclass
class NaNDetectionLog:
    iter_count: int
    tensor_name: str
    nan_count: int
    inf_count: int
    total_count: int
    recent_loss_trend: list[float]


class BaseTrainer(ABC):
    def __init__(self, model: BaseModel,
                 src_aligned_dir: Path, dst_aligned_dir: Path,
                 device: Optional[torch.device] = None,
                 progress_callback: Optional[Callable] = None,
                 preview_callback: Optional[Callable] = None):
        self.model = model
        self.config = model.config
        self.model_dir = model.model_dir
        self.src_aligned_dir = Path(src_aligned_dir)
        self.dst_aligned_dir = Path(dst_aligned_dir)
        self.progress_callback = progress_callback
        self.preview_callback = preview_callback

        if device is not None:
            self.device = device
        else:
            from faceswap.shared.config import auto_select_device
            self.device = auto_select_device()
            if self.device.type == 'xpu':
                _logger.info("Using Intel XPU device")
            elif self.device.type == 'cpu':
                _logger.warning("No CUDA/XPU device found, using CPU")

        self._stop_requested = False
        self._save_requested = False
        self._preview_requested = False
        self._iter_count = model.get_aux_state().get('iter_count', 0)
        self._start_time = time.time()
        self._last_save_time = time.time()
        self._last_backup_time = time.time()
        self._save_interval_sec = 1500
        self._backup_interval_sec = 7200
        self._loss_history: list[tuple[int, float, float]] = list(
            model.get_aux_state().get('loss_history', []))
        self._loss_history_range = 0
        self._preview_page = 0
        self._preview_section_names = model.get_preview_section_names()
        self._ddp_enabled = False
        self._rank = 0
        self._world_size = 1
        self._loss_history_recent: deque[float] = deque(maxlen=10)
        self._consecutive_nan_count: int = 0
        self._smart_stop = None
        if getattr(self.config, 'smart_stop_enabled', False):
            from faceswap.business.smart_stop_detector import SmartStopDetector
            self._smart_stop = SmartStopDetector(
                window=getattr(self.config, 'smart_stop_window', 500),
                threshold=getattr(self.config, 'smart_stop_threshold', 0.1),
                enabled=True)

    @abstractmethod
    def create_datasets(self) -> tuple[Dataset, Dataset]:
        ...

    @abstractmethod
    def preprocess_batch(self, batch_src: dict, batch_dst: dict) -> tuple[dict, dict]:
        ...

    @abstractmethod
    def postprocess_step(self, losses: dict) -> tuple[float, float]:
        ...

    def request_stop(self) -> None:
        self._stop_requested = True

    def request_save(self) -> None:
        self._save_requested = True

    def _save_model(self, iter_count: int) -> None:
        if self._rank == 0:
            self.model._aux_state['loss_history'] = list(self._loss_history)
            self.model.save(iter_count)

    def request_preview(self) -> None:
        self._preview_requested = True

    def enable_ddp(self, world_size: int) -> None:
        self._ddp_enabled = True
        self._world_size = world_size

    def cycle_loss_range(self) -> None:
        self._loss_history_range = (self._loss_history_range + 1) % 3

    def next_preview_page(self) -> None:
        self._preview_page = (self._preview_page + 1) % len(self._preview_section_names)

    def _setup_amp(self) -> None:
        from faceswap.core.amp_utils import AMPManager
        amp_mode = getattr(self.config, 'amp_mode', 'fp32')
        self._amp = AMPManager(self.device, amp_mode)

    def train_one_step(self, batch_src: dict, batch_dst: dict) -> tuple[float, float]:
        _diag = self._iter_count < 3
        _t0 = time.time() if _diag else 0

        batch_src, batch_dst = self.preprocess_batch(batch_src, batch_dst)
        if _diag: _logger.info(f"[DIAG] step#{self._iter_count} preprocess: {time.time()-_t0:.3f}s")

        c = self.config
        amp = self._amp

        _t1 = time.time() if _diag else 0
        with amp.autocast():
            warped_src = batch_src['warped_image']
            warped_dst = batch_dst['warped_image']
            if getattr(c, 'gradient_checkpointing', False):
                fw = torch.utils.checkpoint.checkpoint(
                    self.model.forward, warped_src, warped_dst, use_reentrant=False)
            else:
                fw = self.model.forward(warped_src, warped_dst)
            if _diag: _logger.info(f"[DIAG] step#{self._iter_count} forward: {time.time()-_t1:.3f}s")

            has_nan = False
            for k, v in fw.items():
                if isinstance(v, torch.Tensor) and v.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    nan_c = v.isnan().sum().item()
                    inf_c = v.isinf().sum().item()
                    if nan_c > 0 or inf_c > 0:
                        log = NaNDetectionLog(
                            iter_count=self._iter_count, tensor_name=k,
                            nan_count=nan_c, inf_count=inf_c, total_count=v.numel(),
                            recent_loss_trend=list(self._loss_history_recent))
                        _logger.warning(
                            f"NaN/Inf in forward output '{k}': "
                            f"NaN={nan_c}/{v.numel()}, Inf={inf_c}/{v.numel()}, "
                            f"loss_trend={list(self._loss_history_recent)}")
                        has_nan = True
                        break

            if self._ddp_enabled and self._world_size > 1:
                nan_flag = torch.tensor([1.0 if has_nan else 0.0], device=self.device)
                torch.distributed.all_reduce(nan_flag, op=torch.distributed.ReduceOp.MAX)
                has_nan = nan_flag.item() > 0.5

            if has_nan:
                _logger.warning("Skipping step due to NaN in forward output")
                if self._ddp_enabled and self._world_size > 1:
                    dummy_parts = [v.sum() * 0.0 for v in fw.values()
                                   if isinstance(v, torch.Tensor) and v.is_floating_point() and not v.isnan().all()]
                    if dummy_parts:
                        dummy_loss = sum(dummy_parts)
                    else:
                        dummy_loss = torch.zeros(1, device=self.device, requires_grad=True).sum()
                    amp.scale(dummy_loss).backward()
                del fw
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                amp.update()
                return 0.0, 0.0

            losses = self.model.compute_loss(batch_src, batch_dst, fw)
            G_loss = losses['G_loss']
            if _diag: _logger.info(f"[DIAG] step#{self._iter_count} compute_loss: {time.time()-_t1:.3f}s  G_loss={G_loss.item():.5f}")

        src_dst_opt = self.model._optimizers_dict.get('src_dst_opt')
        D_code_opt = self.model._optimizers_dict.get('D_code_opt')
        D_src_opt = self.model._optimizers_dict.get('D_src_opt')

        g_loss_is_nan = torch.isnan(G_loss).any() or torch.isinf(G_loss).any()
        if self._ddp_enabled and self._world_size > 1:
            nan_flag = torch.tensor([1.0 if g_loss_is_nan else 0.0], device=self.device)
            torch.distributed.all_reduce(nan_flag, op=torch.distributed.ReduceOp.MAX)
            g_loss_is_nan = nan_flag.item() > 0.5

        if g_loss_is_nan:
            _logger.warning(f"Skipping step: G_loss={G_loss.item():.6f} (NaN/Inf)")
            if self._ddp_enabled and self._world_size > 1:
                dummy_loss = torch.zeros(1, device=self.device, requires_grad=True).sum()
                amp.scale(dummy_loss).backward()
            del fw, losses
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            amp.update()
            return 0.0, 0.0

        if self._ddp_enabled and self._world_size > 1:
            nan_flag = torch.tensor([0.0], device=self.device)
            torch.distributed.all_reduce(nan_flag, op=torch.distributed.ReduceOp.MAX)
            if nan_flag.item() > 0.5:
                _logger.warning("Skipping step: another rank detected NaN/Inf in loss")
                dummy_loss = G_loss * 0.0
                amp.scale(dummy_loss).backward()
                amp.update()
                return 0.0, 0.0

        _t2 = time.time() if _diag else 0
        src_dst_opt.zero_grad()
        amp.scale(G_loss).backward()

        amp.unscale_(src_dst_opt)

        if getattr(c, 'clipgrad', False):
            all_params = []
            for pg in src_dst_opt.param_groups:
                all_params.extend(pg['params'])
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)

        lr_dropout_rate = getattr(self.model, '_lr_dropout_rate', 1.0)
        if lr_dropout_rate != 1.0:
            with torch.no_grad():
                for p in src_dst_opt.param_groups[0]['params']:
                    if p.grad is not None:
                        dropout_mask = torch.bernoulli(
                            torch.empty_like(p.grad).fill_(lr_dropout_rate))
                        p.grad.data.mul_(dropout_mask)
                        del dropout_mask

        amp.step(src_dst_opt)
        amp.update()
        if _diag: _logger.info(f"[DIAG] step#{self._iter_count} backward+opt: {time.time()-_t2:.3f}s")

        D_code_loss = losses.get('D_code_loss')
        if D_code_loss is not None and D_code_opt is not None:
            D_code_opt.zero_grad()
            amp.scale(D_code_loss).backward()
            amp.step(D_code_opt)

        D_gan_loss = losses.get('D_gan_loss')
        if D_gan_loss is not None and D_src_opt is not None:
            D_src_opt.zero_grad()
            amp.scale(D_gan_loss).backward()
            amp.step(D_src_opt)

        result = self.postprocess_step(losses)
        del fw, losses, G_loss
        src_dst_opt.zero_grad()
        if _diag: _logger.info(f"[DIAG] step#{self._iter_count} total: {time.time()-_t0:.3f}s")
        return result

    def _record_step_result(self, src_loss: float, dst_loss: float) -> None:
        avg = (src_loss + dst_loss) / 2
        self._loss_history_recent.append(avg)
        if avg == 0.0 and src_loss == 0.0 and dst_loss == 0.0:
            self._consecutive_nan_count += 1
        else:
            self._consecutive_nan_count = 0
        if self._consecutive_nan_count >= 100:
            _logger.error("连续100步NaN/零损失，训练已发散，建议降低学习率或检查数据")
            self._stop_requested = True

    def train(self) -> None:
        if self._ddp_enabled and self._world_size > 1:
            from faceswap.core.ddp_utils import ddp_spawn
            ddp_spawn(self._ddp_train_fn, self._world_size)
            return
        self._train_single()

    def _ddp_train_fn(self, rank: int, world_size: int) -> None:
        from faceswap.core.ddp_utils import setup_process_group, cleanup_process_group, wrap_model_ddp
        setup_process_group(rank, world_size)
        self._rank = rank
        self.device = torch.device(f'cuda:{rank}')
        try:
            self.model.to(self.device)
            self.model = wrap_model_ddp(self.model, rank)
            self._train_single()
        finally:
            cleanup_process_group()

    def _train_single(self) -> None:
        from faceswap.shared.torch_config import configure_torch
        configure_torch("gpu_train")
        c = self.config
        _t0 = time.time()
        self._setup_amp()
        _logger.info(f"[DIAG] setup_amp: {time.time()-_t0:.2f}s")

        _t1 = time.time()
        if torch.cuda.is_available() and self.device.type == 'cuda':
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
            try:
                torch.cuda.set_per_process_memory_fraction(0.90, self.device)
            except Exception:
                pass
        _logger.info(f"[DIAG] cuda_config: {time.time()-_t1:.2f}s")

        _t2 = time.time()
        src_dataset, dst_dataset = self.create_datasets()
        _logger.info(f"[DIAG] create_datasets: {time.time()-_t2:.2f}s  "
                     f"src={len(src_dataset)} dst={len(dst_dataset)}")

        if getattr(c, 'ct_mode', 'none') != 'none':
            self._refresh_ct_dst_sample(src_dataset, dst_dataset)

        effective_bs = getattr(c, 'batch_size', 8)
        if len(src_dataset) < effective_bs or len(dst_dataset) < effective_bs:
            old_bs = effective_bs
            effective_bs = min(len(src_dataset), len(dst_dataset))
            if effective_bs < 1:
                raise ValueError(
                    f"Not enough training data: src={len(src_dataset)}, dst={len(dst_dataset)}. "
                    f"Need at least 1 image in each directory.")
            _logger.warning(f"Batch size reduced: {old_bs} -> {effective_bs} "
                           f"(src={len(src_dataset)}, dst={len(dst_dataset)})")

        from faceswap.shared.config import is_gpu_device, get_num_workers
        from faceswap.shared.torch_config import get_dataloader_config, worker_init_fn
        pin_mem = is_gpu_device(self.device)
        n_workers = get_num_workers(self.device)
        dl_cfg = get_dataloader_config("gpu_train" if pin_mem else "cpu_train",
                                       dataset_size=max(len(src_dataset), len(dst_dataset)))

        def _make_loader(dataset, bs):
            kw = dict(batch_size=bs, num_workers=n_workers, pin_memory=pin_mem,
                      drop_last=True, persistent_workers=n_workers > 0)
            if n_workers > 0:
                kw['prefetch_factor'] = dl_cfg.get("prefetch_factor", 2)
                kw['worker_init_fn'] = worker_init_fn
            weights = getattr(dataset, 'yaw_weights', None)
            use_yaw = weights is not None and len(weights) > 0 and getattr(c, 'uniform_yaw', False)
            if self._ddp_enabled:
                if use_yaw:
                    from faceswap.core.ddp_utils import WeightedDistributedSampler
                    sampler = WeightedDistributedSampler(
                        dataset, weights=weights, num_replicas=self._world_size,
                        rank=self._rank, shuffle=True)
                else:
                    from torch.utils.data.distributed import DistributedSampler
                    sampler = DistributedSampler(dataset, num_replicas=self._world_size,
                                                  rank=self._rank, shuffle=True)
                kw['sampler'] = sampler
            else:
                if use_yaw:
                    from torch.utils.data import WeightedRandomSampler
                    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
                    kw['sampler'] = sampler
                else:
                    kw['shuffle'] = True
            return DataLoader(dataset, **kw)

        _t5 = time.time()
        src_loader = _make_loader(src_dataset, effective_bs)
        dst_loader = _make_loader(dst_dataset, effective_bs)
        _logger.info(f"[DIAG] create_dataloaders: {time.time()-_t5:.2f}s  n_workers={n_workers}")

        if n_workers > 0:
            _t_warmup_dl = time.time()
            _logger.info("Pre-warming DataLoader workers...")
            _src_warmup = iter(src_loader)
            _dst_warmup = iter(dst_loader)
            _ = next(_src_warmup)
            _ = next(_dst_warmup)
            del _src_warmup, _dst_warmup, _
            _logger.info(f"[DIAG] dataloader_warmup (worker spawn + first batch): "
                         f"{time.time()-_t_warmup_dl:.2f}s")

        _t6 = time.time()
        self._warmup_compile(src_dataset, dst_dataset, effective_bs)
        _logger.info(f"[DIAG] warmup_compile: {time.time()-_t6:.2f}s")

        ddp_str = f" DDP(world_size={self._world_size})" if self._ddp_enabled else ""
        _logger.info(f"Training started: {getattr(c, 'archi', '?')} res={getattr(c, 'resolution', '?')} "
                     f"bs={effective_bs} device={self.device}{ddp_str}")
        _logger.info(f"[DIAG] total_startup: {time.time()-_t0:.2f}s")

        if self._rank == 0:
            lock_file = self.model_dir / ".training_lock"
            lock_file.write_text(str(os.getpid()))
        else:
            lock_file = None
        try:
                if self._rank == 0:
                    self._save_model(self._iter_count)
                    _logger.info(f"Initial save at iter {self._iter_count}")
                epoch = 0
                first_epoch = True
                while not self._stop_requested:
                    if self._ddp_enabled:
                        for loader in (src_loader, dst_loader):
                            if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
                                loader.sampler.set_epoch(epoch)
                        epoch += 1
                    src_iter = iter(src_loader)
                    dst_iter = iter(dst_loader)
                    steps_this_epoch = max(len(src_loader), len(dst_loader))
                    _t_first_batch = time.time()
                    _first_batch_logged = False
                    if first_epoch and self._rank == 0:
                        first_epoch = False
                        self._generate_and_send_preview(src_dataset, dst_dataset)
                    for step in range(steps_this_epoch):
                        if self._stop_requested:
                            break
                        t_iter_start = time.time()
                        try:
                            batch_src = next(src_iter)
                        except StopIteration:
                            src_iter = iter(src_loader)
                            batch_src = next(src_iter)
                        try:
                            batch_dst = next(dst_iter)
                        except StopIteration:
                            dst_iter = iter(dst_loader)
                            batch_dst = next(dst_iter)

                        self._update_lr()

                        if not _first_batch_logged:
                            _logger.info(f"[DIAG] first_batch_fetch: {time.time()-_t_first_batch:.2f}s")
                            _first_batch_logged = True

                        oom_retry_count = 0
                        while True:
                            try:
                                src_loss, dst_loss = self.train_one_step(batch_src, batch_dst)
                                break
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower() and oom_retry_count < 3:
                                    oom_retry_count += 1
                                    old_bs = getattr(c, 'batch_size', effective_bs)
                                    new_bs = max(1, old_bs // 2)
                                    _logger.warning(f"CUDA OOM: batch_size {old_bs} -> {new_bs} (retry {oom_retry_count}/3)")
                                    c.batch_size = new_bs
                                    effective_bs = new_bs
                                    torch.cuda.empty_cache()
                                    if new_bs <= 1:
                                        _logger.error("显存不足，batch_size已降至1仍无法训练")
                                        self._stop_requested = True
                                        src_loss, dst_loss = 0.0, 0.0
                                        break
                                    src_loader = _make_loader(src_dataset, effective_bs)
                                    dst_loader = _make_loader(dst_dataset, effective_bs)
                                else:
                                    raise

                        self._record_step_result(src_loss, dst_loss)
                        iter_ms = (time.time() - t_iter_start) * 1000.0
                        self._iter_count += 1

                        if self._smart_stop is not None:
                            self._smart_stop.update(self._iter_count, (src_loss + dst_loss) / 2)
                            if self._smart_stop.is_converged() and self.progress_callback is not None and not self._stop_requested:
                                self.progress_callback(
                                    self._iter_count, src_loss, dst_loss, iter_ms,
                                    getattr(c, 'lr', 0), converged=True)

                        self._loss_history.append((self._iter_count, src_loss, dst_loss))
                        if len(self._loss_history) > 10000:
                            del self._loss_history[:5000]

                        if self._iter_count % 2000 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        if getattr(c, 'ct_mode', 'none') != 'none' and self._iter_count % 5000 == 0:
                            self._refresh_ct_dst_sample(src_dataset, dst_dataset)

                        if self._save_requested and self._rank == 0:
                            if self.progress_callback is not None:
                                self.progress_callback(-1, 0, 0, 0, 0)
                            self._save_model(self._iter_count)
                            self._save_requested = False
                            _logger.info(f"Model saved at iter {self._iter_count}")
                            self._generate_and_send_preview(src_dataset, dst_dataset)

                        if self.progress_callback is not None and self._rank == 0 and not self._stop_requested:
                            elapsed = time.time() - self._start_time
                            self.progress_callback(
                                self._iter_count, src_loss, dst_loss, iter_ms,
                                getattr(c, 'lr', 0))

                        if self._preview_requested and not self._stop_requested and self.preview_callback is not None and self._rank == 0:
                            self._preview_requested = False
                            self._generate_and_send_preview(src_dataset, dst_dataset)

                        if not self._stop_requested and self._rank == 0:
                            now = time.time()
                            if now - self._last_save_time >= self._save_interval_sec:
                                self._last_save_time += self._save_interval_sec
                                if self.progress_callback is not None:
                                    self.progress_callback(-1, 0, 0, 0, 0)
                                self._save_model(self._iter_count)
                                _logger.info(f"Auto-saved at iter {self._iter_count}")
                                if now - self._last_backup_time >= self._backup_interval_sec:
                                    self._last_backup_time = now
                                    self.model.create_backup()
                                    _logger.info(f"Auto-backup at iter {self._iter_count}")
                                self._generate_and_send_preview(src_dataset, dst_dataset)

                        if getattr(c, 'target_iter', 0) > 0 and self._iter_count >= c.target_iter:
                            self._save_model(self._iter_count)
                            _logger.info(f"Target iter {c.target_iter} reached, training complete")
                            return
        finally:
            if self._rank == 0:
                self._save_model(self._iter_count)
                if lock_file is not None and lock_file.exists():
                    lock_file.unlink()
        _logger.info(f"Training stopped at iter {self._iter_count}")

    def _update_lr(self) -> None:
        c = self.config
        lr_cos = getattr(c, 'lr_cos', 0)
        if lr_cos > 0:
            import math as _math
            lr_mult = (_math.cos(self._iter_count * 2.0 * _math.pi / float(lr_cos)) + 1.0) / 2.0
            for opt_name, opt in self.model._optimizers_dict.items():
                base_lr = c.lr if opt_name == 'src_dst_opt' else getattr(c, 'lr', 1e-4)
                for pg in opt.param_groups:
                    pg['lr'] = base_lr * lr_mult

    @torch.no_grad()
    def _warmup_compile(self, src_dataset, dst_dataset, bs) -> None:
        has_compile = any('OptimizedModule' in type(m).__name__ or 'compile' in type(m).__module__
                         for m in self.model._modules_dict.values())
        if not has_compile:
            return
        _logger.info("Warmup: 编译模型中，请稍候...")
        c = self.config
        res = getattr(c, 'resolution', 128)
        dummy_src = torch.randn(1, 3, res, res, device=self.device, dtype=torch.float32)
        dummy_dst = torch.randn(1, 3, res, res, device=self.device, dtype=torch.float32)
        try:
            with self._amp.autocast():
                fw = self.model.forward(dummy_src, dummy_dst)
                _ = self.model.compute_loss(
                    {'warped_image': dummy_src, 'target_image': dummy_src,
                     'target_mask': torch.ones(1, 1, res, res, device=self.device),
                     'target_em_mask': torch.ones(1, 1, res, res, device=self.device)},
                    {'warped_image': dummy_dst, 'target_image': dummy_dst,
                     'target_mask': torch.ones(1, 1, res, res, device=self.device),
                     'target_em_mask': torch.ones(1, 1, res, res, device=self.device)},
                    fw)
            del dummy_src, dummy_dst, fw
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _logger.info("Warmup: 模型编译完成")
        except Exception as e:
            _logger.warning(f"Warmup compile failed (non-fatal): {e}")

    def _refresh_ct_dst_sample(self, src_dataset, dst_dataset) -> None:
        c = self.config
        idx = random.randint(0, len(dst_dataset.image_paths) - 1)
        dst_img = cv2.imread(str(dst_dataset.image_paths[idx]))
        if dst_img is not None:
            dst_img_r = cv2.resize(dst_img, (c.resolution, c.resolution))
            dst_meta = dst_dataset._get_meta(dst_dataset.image_paths[idx])
            dst_mask = dst_dataset._render_full_mask(dst_img.shape[:2], dst_meta)
            dst_mask = cv2.resize(dst_mask, (c.resolution, c.resolution),
                                  interpolation=cv2.INTER_LINEAR)
            src_dataset.set_dst_sample_for_ct(dst_img_r, dst_mask)

    def _generate_and_send_preview(self, src_dataset, dst_dataset) -> None:
        if self.preview_callback is None or self._stop_requested:
            return
        try:
            self.model.eval()
            preview_bgr = self._generate_preview(src_dataset, dst_dataset)
            self.model.train()
            self.preview_callback(preview_bgr)
        except Exception as e:
            _logger.warning(f"Preview generation failed: {e}")

    @torch.no_grad()
    def _generate_preview(self, src_dataset, dst_dataset) -> np.ndarray:
        c = self.config
        res = getattr(c, 'resolution', 128)
        n_samples = min(4, len(src_dataset), len(dst_dataset))
        if n_samples < 1:
            return np.zeros((res, res * 5, 3), dtype=np.uint8)

        src_indices = random.sample(range(len(src_dataset.image_paths)), n_samples)
        dst_indices = random.sample(range(len(dst_dataset.image_paths)), n_samples)

        sections = self.model.generate_preview_data(
            src_dataset, dst_dataset, src_indices, dst_indices)

        page = self._preview_page % len(self._preview_section_names)
        section_name = self._preview_section_names[page]
        section_rows = sections.get(section_name, [])

        if not section_rows:
            return np.zeros((res, res * 5, 3), dtype=np.uint8)

        from datetime import datetime
        from faceswap.core.preview_utils import compose_preview
        ts = datetime.now().strftime("%H:%M:%S")
        model_name = type(self.model).__name__
        head_lines = [
            f"[{ts}] [s]:save  [p]:update  [space]:next  [l]:range  [Enter]:stop",
            f'{model_name}  iter=#{self._iter_count}  arch={getattr(c, "archi", "?")}  '
            f'res={getattr(c, "resolution", "?")}  |  "{section_name}" [{page + 1}/{len(self._preview_section_names)}]',
        ]
        return compose_preview(
            section_rows, head_lines, self._loss_history, self._loss_history_range,
            loss_names=["src", "dst"], loss_colors=[(0, 180, 255), (0, 255, 120)],
        )
