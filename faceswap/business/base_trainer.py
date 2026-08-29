import gc
import os
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
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

SAVE_INTERVAL_SEC = 1500
BACKUP_INTERVAL_SEC = 7200
LOSS_HISTORY_RECENT_MAXLEN = 10
MAX_CONSECUTIVE_NAN = 100
CUDA_MEM_FRACTION = 0.90
LOSS_HISTORY_MAX = 10000
LOSS_HISTORY_TRIM = 5000
GC_COLLECT_INTERVAL = 2000
CT_REFRESH_INTERVAL = 5000
MAX_OOM_RETRIES = 3


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
        self._save_interval_sec = SAVE_INTERVAL_SEC
        self._backup_interval_sec = BACKUP_INTERVAL_SEC
        self._backup_interval_iter = max(0, getattr(self.config, 'backup_interval', 0))
        self._last_backup_iter = self._iter_count
        self._loss_history: list[tuple[int, float, float]] = list(
            model.get_aux_state().get('loss_history', []))
        self._loss_history_range = 0
        self._preview_page = 0
        self._preview_section_names = model.get_preview_section_names()
        self._ddp_enabled = False
        self._rank = 0
        self._world_size = 1
        self._loss_history_recent: deque[float] = deque(maxlen=LOSS_HISTORY_RECENT_MAXLEN)
        self._consecutive_nan_count: int = 0

    @abstractmethod
    def create_datasets(self) -> tuple[Dataset, Dataset]:
        ...

    @abstractmethod
    def preprocess_batch(self, batch_src: dict, batch_dst: dict) -> tuple[dict, dict]:
        ...

    def postprocess_step(self, losses: dict) -> tuple[float, float, float, float]:
        d_gan = losses.get('D_gan_loss')
        d_gan_val = d_gan.item() if d_gan is not None else 0.0
        return losses['src_loss'], losses['dst_loss'], 0.0, d_gan_val

    def save_model(self) -> None:
        self.model.save(self._iter_count)

    @torch.no_grad()
    def AE_merge(self, warped_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.config
        res = c.resolution
        if warped_dst.ndim == 3:
            warped_dst = warped_dst[None, ...]
        face_nchw = warped_dst.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        face_tensor = torch.from_numpy(face_nchw).to(self.device)

        pred, pred_mask, dst_mask = self.model.get_merge_face(face_tensor)

        bgr = pred[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        mask_prd = pred_mask[0, 0].cpu().numpy().astype(np.float32)
        mask_dst = dst_mask[0, 0].cpu().numpy().astype(np.float32)
        return bgr, mask_prd, mask_dst

    def predictor_func(self, face: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.config
        res = c.resolution
        if face.ndim == 3:
            face_in = face[None, ...]
        else:
            face_in = face
        face_resized = np.stack([cv2.resize(f, (res, res)) for f in face_in])

        bgr, mask_prd, mask_dst = self.AE_merge(face_resized)
        return bgr, mask_prd, mask_dst

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

    def train_one_step(self, batch_src: dict, batch_dst: dict) -> tuple[float, float, float, float]:
        self.model.register_aux_state('iter_count', self._iter_count)

        batch_src, batch_dst = self.preprocess_batch(batch_src, batch_dst)

        c = self.config
        amp = self._amp

        with amp.autocast():
            warped_src = batch_src['warped_image']
            warped_dst = batch_dst['warped_image']
            use_ckpt = False
            if use_ckpt and getattr(c, 'enable_torch_compile', False):
                use_ckpt = False
            if use_ckpt:
                fw = torch.utils.checkpoint.checkpoint(
                    self.model.forward, warped_src, warped_dst, use_reentrant=False)
            else:
                fw = self.model.forward(warped_src, warped_dst)

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
            return 0.0, 0.0, 0.0, 0.0

        fw_fp32 = {k: v.float() if isinstance(v, torch.Tensor) and v.dtype != torch.float32 else v
                   for k, v in fw.items()}
        losses = self.model.compute_loss(batch_src, batch_dst, fw_fp32)
        G_loss = losses['G_loss']

        src_dst_opt = self.model._optimizers_dict.get('src_dst_opt')
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
            return 0.0, 0.0, 0.0, 0.0

        if self._ddp_enabled and self._world_size > 1:
            nan_flag = torch.tensor([0.0], device=self.device)
            torch.distributed.all_reduce(nan_flag, op=torch.distributed.ReduceOp.MAX)
            if nan_flag.item() > 0.5:
                _logger.warning("Skipping step: another rank detected NaN/Inf in loss")
                dummy_loss = G_loss * 0.0
                amp.scale(dummy_loss).backward()
                amp.update()
                return 0.0, 0.0, 0.0, 0.0

        D_gan_loss = losses.get('D_gan_loss')
        need_d_gan = D_gan_loss is not None and D_src_opt is not None

        accum_steps = getattr(c, 'gradient_accumulation_steps', 1)
        is_accum_step = (self._iter_count + 1) % accum_steps == 0

        if (self._iter_count % accum_steps) == 0:
            src_dst_opt.zero_grad()
        amp.scale(G_loss / accum_steps).backward(retain_graph=need_d_gan)

        if is_accum_step:
            amp.unscale_(src_dst_opt)
            if getattr(c, 'clipgrad', False):
                all_params = []
                for pg in src_dst_opt.param_groups:
                    all_params.extend(pg['params'])
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)

            amp.step(src_dst_opt)

        if need_d_gan:
            if (self._iter_count % accum_steps) == 0:
                D_src_opt.zero_grad()
            amp.scale(D_gan_loss / accum_steps).backward()
            if is_accum_step:
                if getattr(c, 'clipgrad', False):
                    amp.unscale_(D_src_opt)
                    d_params = []
                    for pg in D_src_opt.param_groups:
                        d_params.extend(pg['params'])
                    torch.nn.utils.clip_grad_norm_(d_params, 1.0)
                amp.step(D_src_opt)

        if is_accum_step:
            amp.update()

        result = self.postprocess_step(losses)
        del fw, losses, G_loss
        return result

    def _record_step_result(self, src_loss: float, dst_loss: float) -> None:
        avg = (src_loss + dst_loss) / 2
        self._loss_history_recent.append(avg)
        if avg == 0.0 and src_loss == 0.0 and dst_loss == 0.0:
            self._consecutive_nan_count += 1
        else:
            self._consecutive_nan_count = 0
        if self._consecutive_nan_count >= MAX_CONSECUTIVE_NAN:
            _logger.error(f"连续{MAX_CONSECUTIVE_NAN}步NaN/零损失，训练已发散，建议降低学习率或检查数据")
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
        self._setup_amp()

        if torch.cuda.is_available() and self.device.type == 'cuda':
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
            try:
                torch.cuda.set_per_process_memory_fraction(CUDA_MEM_FRACTION, self.device)
            except Exception:
                pass

        src_dataset, dst_dataset = self.create_datasets()
        _logger.info(f"Datasets: src={len(src_dataset)} dst={len(dst_dataset)}")

        if getattr(c, 'ct_mode', 'none') != 'none':
            res = getattr(c, 'resolution', 256)
            self._ct_img_shared = torch.zeros(res, res, 3, dtype=torch.uint8).share_memory_()
            self._ct_mask_shared = torch.zeros(res, res, dtype=torch.uint8).share_memory_()
            self._ct_valid_shared = torch.zeros(1, dtype=torch.int32).share_memory_()
            src_dataset.set_ct_shared(self._ct_img_shared, self._ct_mask_shared, self._ct_valid_shared)
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

        src_loader = _make_loader(src_dataset, effective_bs)
        dst_loader = _make_loader(dst_dataset, effective_bs)

        if n_workers > 0:
            _logger.info("Pre-warming DataLoader workers...")
            _src_warmup = iter(src_loader)
            _dst_warmup = iter(dst_loader)
            _ = next(_src_warmup)
            _ = next(_dst_warmup)
            del _src_warmup, _dst_warmup, _

        self._warmup_compile(src_dataset, dst_dataset, effective_bs)

        if self._stop_requested:
            _logger.info("训练在warmup阶段被停止")
            return

        ddp_str = f" DDP(world_size={self._world_size})" if self._ddp_enabled else ""
        _logger.info(f"Training started: res={getattr(c, 'resolution', '?')} "
                     f"bs={effective_bs} device={self.device}{ddp_str}")

        _logger.info(
            f"Config: res={getattr(c,'resolution','?')} bs={effective_bs} "
            f"face_type={getattr(c,'face_type','?')} archi={getattr(c,'archi','?')} "
            f"opt={'AdaBelief' if getattr(c,'adabelief',False) else 'RMSprop'} "
            f"lr={getattr(c,'lr','?')} amp={getattr(c,'amp_mode','?')} "
            f"dims=ae{getattr(c,'ae_dims','?')}/e{getattr(c,'e_dims','?')}/d{getattr(c,'d_dims','?')}/dm{getattr(c,'d_mask_dims','?')} "
            f"true_face={getattr(c,'true_face_power',0)} "
            f"face_style={getattr(c,'face_style_power',0)} bg_style={getattr(c,'bg_style_power',0)} "
            f"gan={getattr(c,'gan_power',0)} "
            f"pretrain={getattr(c,'pretrain',False)}"
        )

        if self._rank == 0:
            lock_file = self.model_dir / ".training_lock"
            lock_file.write_text(str(os.getpid()))
        else:
            lock_file = None
        try:
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

                        try:
                            src_loss, dst_loss, _, d_gan_loss = self.train_one_step(batch_src, batch_dst)
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                torch.cuda.empty_cache()
                                raise RuntimeError(
                                    f"CUDA显存不足(OOM)。\n"
                                    f"当前配置: resolution={c.resolution}, batch_size={getattr(c,'batch_size',effective_bs)}, "
                                    f"ae_dims={getattr(c,'ae_dims','?')}, amp={getattr(c,'amp_mode','?')}\n"
                                    f"请尝试降低batch_size、降低resolution或降低ae_dims。"
                                ) from e
                            raise

                        self._record_step_result(src_loss, dst_loss)
                        iter_ms = (time.time() - t_iter_start) * 1000.0
                        self._iter_count += 1

                        self._loss_history.append((self._iter_count, src_loss, dst_loss))
                        if len(self._loss_history) > LOSS_HISTORY_MAX:
                            del self._loss_history[:LOSS_HISTORY_TRIM]

                        if self._iter_count % GC_COLLECT_INTERVAL == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        if getattr(c, 'ct_mode', 'none') != 'none' and self._iter_count % CT_REFRESH_INTERVAL == 0:
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
                                getattr(c, 'lr', 0),
                                d_gan_loss=d_gan_loss)

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
                                self._generate_and_send_preview(src_dataset, dst_dataset)
                            if self._backup_interval_iter > 0:
                                if self._iter_count - self._last_backup_iter >= self._backup_interval_iter:
                                    self._last_backup_iter = self._iter_count
                                    self.model.create_backup()
                                    _logger.info(f"Auto-backup at iter {self._iter_count}")
                            elif now - self._last_backup_time >= self._backup_interval_sec:
                                self._last_backup_time = now
                                self.model.create_backup()
                                _logger.info(f"Auto-backup at iter {self._iter_count}")

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

    def _warmup_compile(self, src_dataset, dst_dataset, bs) -> None:
        has_compile = any('OptimizedModule' in type(m).__name__ or 'compile' in type(m).__module__
                         for m in self.model._modules_dict.values())
        if not has_compile:
            return
        if self._stop_requested:
            return
        _logger.info("Warmup: 编译模型中，请稍候...")
        c = self.config
        res = getattr(c, 'resolution', 128)
        opt = self.model._optimizers_dict.get('src_dst_opt')

        if self._stop_requested:
            return
        dummy_src = torch.randn(bs, 3, res, res, device=self.device, dtype=torch.float32)
        dummy_dst = torch.randn(bs, 3, res, res, device=self.device, dtype=torch.float32)
        batch_src = {
            'warped_image': dummy_src, 'target_image': dummy_src,
            'target_mask': torch.ones(bs, 1, res, res, device=self.device),
            'target_em_mask': torch.ones(bs, 1, res, res, device=self.device),
            'target_vis_mask': torch.ones(bs, 1, res, res, device=self.device),
        }
        batch_dst = {
            'warped_image': dummy_dst, 'target_image': dummy_dst,
            'target_mask': torch.ones(bs, 1, res, res, device=self.device),
            'target_em_mask': torch.ones(bs, 1, res, res, device=self.device),
            'target_vis_mask': torch.ones(bs, 1, res, res, device=self.device),
        }
        try:
            with self._amp.autocast():
                fw = self.model.forward(dummy_src, dummy_dst)
                fw_fp32 = {k: v.float() if isinstance(v, torch.Tensor) and v.dtype != torch.float32 else v for k, v in fw.items()}
                losses = self.model.compute_loss(batch_src, batch_dst, fw_fp32)
                G_loss = losses['G_loss']
            opt.zero_grad()
            self._amp.scale(G_loss).backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            del dummy_src, dummy_dst, fw, losses, G_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            _logger.warning(f"  Warmup failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _logger.info("Warmup: 模型编译完成 (forward+backward, bs=%d)" % bs)

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
            if hasattr(self, '_ct_img_shared'):
                self._ct_img_shared.copy_(torch.from_numpy(dst_img_r))
                self._ct_mask_shared.copy_(torch.from_numpy(dst_mask))
                self._ct_valid_shared.fill_(1)
            else:
                src_dataset.set_dst_sample_for_ct(dst_img_r, dst_mask)

    def _generate_and_send_preview(self, src_dataset, dst_dataset) -> None:
        if self.preview_callback is None or self._stop_requested:
            return
        try:
            preview_bgr = self._generate_preview(src_dataset, dst_dataset)
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

        _saved = {}
        for ds in (src_dataset, dst_dataset):
            _saved[ds] = (ds._augment, ds._random_hsv_power, ds._ct_mode, ds._random_warp)
            ds._augment = False
            ds._random_hsv_power = 0.0
            ds._ct_mode = None
            ds._random_warp = False

        try:
            with self._amp.autocast():
                sections = self.model.generate_preview_data(
                    src_dataset, dst_dataset, src_indices, dst_indices)
        finally:
            for ds, (aug, hsv, ct, rw) in _saved.items():
                ds._augment = aug
                ds._random_hsv_power = hsv
                ds._ct_mode = ct
                ds._random_warp = rw

        page = self._preview_page % len(self._preview_section_names)
        section_name = self._preview_section_names[page]
        section_data = sections.get(section_name, [])

        if not section_data:
            return np.zeros((res, res * 5, 3), dtype=np.uint8)

        section_rows = []
        for row_tensors in section_data:
            imgs = []
            for tensor in row_tensors:
                arr = tensor.float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                imgs.append((arr * 255.0 + 0.5).astype(np.uint8))
            section_rows.append(np.concatenate(imgs, axis=1))

        if not section_rows:
            return np.zeros((res, res * 5, 3), dtype=np.uint8)

        from datetime import datetime
        from faceswap.core.preview_utils import compose_preview
        ts = datetime.now().strftime("%H:%M:%S")
        model_name = type(self.model).__name__
        head_lines = [
            f"[{ts}] [s]:save  [p]:update  [space]:next  [l]:range  [Enter]:stop",
            f'{model_name}  iter=#{self._iter_count}  '
            f'res={getattr(c, "resolution", "?")}  |  "{section_name}" [{page + 1}/{len(self._preview_section_names)}]',
        ]
        return compose_preview(
            section_rows, head_lines, self._loss_history, self._loss_history_range,
            loss_names=["src", "dst"], loss_colors=[(0, 180, 255), (0, 255, 120)],
        )
