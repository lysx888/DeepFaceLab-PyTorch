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
            if self.device.type == 'cpu':
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

    def cleanup(self) -> None:
        try:
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                self.model = None
        except Exception:
            pass
        try:
            import torch._dynamo
            torch._dynamo.reset()
        except Exception:
            pass
        from faceswap.business.vram_manager import cleanup_memory, synchronize
        synchronize()
        cleanup_memory(aggressive=True)

    def _save_model(self, iter_count: int) -> None:
        self.model._aux_state['loss_history'] = list(self._loss_history)
        self.model.save(iter_count)

    def request_preview(self) -> None:
        self._preview_requested = True

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
            fw = self.model.forward(warped_src, warped_dst)

        has_nan = False
        if self._iter_count % 100 == 0:
            for k, v in fw.items():
                if isinstance(v, torch.Tensor) and v.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    if not torch.isfinite(v).all().item():
                        nan_c = v.isnan().sum().item()
                        inf_c = v.isinf().sum().item()
                        _logger.warning(
                            f"NaN/Inf in forward output '{k}': "
                            f"NaN={nan_c}/{v.numel()}, Inf={inf_c}/{v.numel()}, "
                            f"iter={self._iter_count}")
                        has_nan = True
                        break

        if has_nan:
            _logger.warning("Skipping step due to NaN in forward output")
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

        if g_loss_is_nan:
            _logger.warning(f"Skipping step: G_loss={G_loss.item():.6f} (NaN/Inf)")
            del fw, losses
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            amp.update()
            return 0.0, 0.0, 0.0, 0.0

        D_gan_loss = losses.get('D_gan_loss')
        need_d_gan = D_gan_loss is not None and D_src_opt is not None

        accum_steps = getattr(c, 'gradient_accumulation_steps', 1)
        is_accum_step = (self._iter_count + 1) % accum_steps == 0

        if (self._iter_count % accum_steps) == 0:
            src_dst_opt.zero_grad()
            if need_d_gan:
                D_src_opt.zero_grad()

        # DFL 原版用 nn.gradients 分别对 D 和 G 的权重求导，D 只消费 d_loss
        # 梯度，G_loss 对 D 的梯度不计算。PyTorch 的 backward() 会对所有
        # requires_grad=True 的叶子节点计算梯度，G_loss 中的 GAN 项
        # (D(pred_masked) 未 detach D 权重) 会对 D 产生额外梯度。临时禁用
        # D 的 requires_grad 可阻止梯度流向 D 权重，同时保持梯度流过 D
        # 计算图到 G 输出（用于更新 G）。D_gan_loss 的计算图独立于 G_loss，
        # 先 backward 释放小图，G 的 backward 无需 retain_graph。
        if need_d_gan:
            amp.scale(D_gan_loss / accum_steps).backward()
            for pg in D_src_opt.param_groups:
                for p in pg['params']:
                    p.requires_grad_(False)
        amp.scale(G_loss / accum_steps).backward()
        if need_d_gan:
            for pg in D_src_opt.param_groups:
                for p in pg['params']:
                    p.requires_grad_(True)

        if is_accum_step:
            amp.unscale_(src_dst_opt)
            if getattr(c, 'clipgrad', False):
                all_params = []
                for pg in src_dst_opt.param_groups:
                    all_params.extend(pg['params'])
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)

            amp.step(src_dst_opt)

            if need_d_gan:
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
        self._train_single()

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

        if self._stop_requested:
            return

        if getattr(c, 'auto_batch', False) and self.device.type == 'cuda':
            from faceswap.shared.hardware_utils import probe_batch_size
            res = getattr(c, 'resolution', 256)

            class _ProbeWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self._model = model
                    # SAEHDModel 等不是 nn.Module，普通属性不会被注册为子模块，
                    # 导致 parameters() 为空 → SGD 报 "optimizer got an empty parameter list"。
                    # 把内部模块显式注册进来，让显存探测能真实反映模型占用。
                    for name, mod in model.get_modules_dict().items():
                        self.add_module(name, mod)
                def forward(self, x):
                    fw = self._model.forward(x, x)
                    if isinstance(fw, dict):
                        for v in fw.values():
                            if isinstance(v, torch.Tensor):
                                return v
                    return fw

            probed_bs = probe_batch_size(_ProbeWrapper(self.model), res, self.device, in_channels=3)
            if getattr(c, 'gan_power', 0) > 0:
                probed_bs = max(1, int(probed_bs * 0.7))
            c.batch_size = min(probed_bs, getattr(c, 'batch_size', 8))
            _logger.info(f"Auto batch size: {c.batch_size}")

        if self._stop_requested:
            return

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
            if use_yaw:
                from torch.utils.data import WeightedRandomSampler
                sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
                kw['sampler'] = sampler
            else:
                kw['shuffle'] = True
            return DataLoader(dataset, **kw)

        src_loader = _make_loader(src_dataset, effective_bs)
        dst_loader = _make_loader(dst_dataset, effective_bs)

        if self._stop_requested:
            return

        if n_workers > 0:
            _src_warmup = iter(src_loader)
            _dst_warmup = iter(dst_loader)
            _ = next(_src_warmup)
            _ = next(_dst_warmup)
            del _src_warmup, _dst_warmup, _

        self._warmup_compile(src_dataset, dst_dataset, effective_bs)

        if self._stop_requested:
            _logger.info("训练在warmup阶段被停止")
            return

        _logger.info(f"Training started: res={getattr(c, 'resolution', '?')} "
                     f"bs={effective_bs} device={self.device}")

        _logger.info(
            f"Config: res={getattr(c,'resolution','?')} bs={effective_bs} "
            f"face_type={getattr(c,'face_type','?')} archi={getattr(c,'archi','?')} "
            f"opt={'AdaBelief' if str(getattr(c,'adabelief','adabelief')).lower() in ('adabelief','true','1','yes') else 'RMSprop'} "
            f"lr={getattr(c,'lr','?')} amp={getattr(c,'amp_mode','?')} "
            f"dims=ae{getattr(c,'ae_dims','?')}/e{getattr(c,'e_dims','?')}/d{getattr(c,'d_dims','?')}/dm{getattr(c,'d_mask_dims','?')} "
            f"true_face={getattr(c,'true_face_power',0)} "
            f"face_style={getattr(c,'face_style_power',0)} bg_style={getattr(c,'bg_style_power',0)} "
            f"gan={getattr(c,'gan_power',0)} "
            f"pretrain={getattr(c,'pretrain',False)}"
        )

        lock_file = self.model_dir / ".training_lock"
        lock_file.write_text(str(os.getpid()))
        try:
                first_epoch = True
                while not self._stop_requested:
                    src_iter = iter(src_loader)
                    dst_iter = iter(dst_loader)
                    steps_this_epoch = max(len(src_loader), len(dst_loader))
                    if first_epoch:
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

                        if self._save_requested:
                            if self.progress_callback is not None:
                                self.progress_callback(-1, 0, 0, 0, 0)
                            self._save_model(self._iter_count)
                            self._save_requested = False
                            _logger.info(f"Model saved at iter {self._iter_count}")
                            self._generate_and_send_preview(src_dataset, dst_dataset)

                        if self.progress_callback is not None and not self._stop_requested:
                            elapsed = time.time() - self._start_time
                            self.progress_callback(
                                self._iter_count, src_loss, dst_loss, iter_ms,
                                getattr(c, 'lr', 0),
                                d_gan_loss=d_gan_loss)

                        if self._preview_requested and not self._stop_requested and self.preview_callback is not None:
                            self._preview_requested = False
                            self._generate_and_send_preview(src_dataset, dst_dataset)

                        if not self._stop_requested:
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
