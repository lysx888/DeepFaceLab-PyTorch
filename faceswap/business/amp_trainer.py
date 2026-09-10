"""AMP模型训练配置和入口。"""
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from typing import Optional, Callable
import torch

from faceswap.business.generic_trainer import GenericTrainer
from faceswap.models.amp.amp_model import AMPModel
from faceswap.shared.logger import get_logger

_logger = get_logger("amp_trainer")


class AMPTrainingConfig:
    def __init__(self, **kwargs):
        self.resolution = kwargs.get('resolution', 224)
        self.face_type = kwargs.get('face_type', 'wf')
        self.ae_dims = kwargs.get('ae_dims', 256)
        self.inter_dims = kwargs.get('inter_dims', 1024)
        self.e_dims = kwargs.get('e_dims', 64)
        self.d_dims = kwargs.get('d_dims', 64)
        d_mask_default = self.d_dims // 3
        d_mask_default += d_mask_default % 2
        self.d_mask_dims = kwargs.get('d_mask_dims', d_mask_default)
        self.morph_factor = kwargs.get('morph_factor', 0.5)
        self.uniform_yaw = kwargs.get('uniform_yaw', False)
        self.blur_out_mask = kwargs.get('blur_out_mask', False)
        self.lr_dropout = kwargs.get('lr_dropout', 'n')
        if self.lr_dropout not in ('n', 'y', 'cpu', 'ca'):
            raise ValueError(f"lr_dropout='{self.lr_dropout}' 无效，可选: n, y, cpu, ca")
        self.random_warp = kwargs.get('random_warp', True)
        self.random_src_flip = kwargs.get('random_src_flip', True)
        self.random_dst_flip = kwargs.get('random_dst_flip', True)
        self.ct_mode = kwargs.get('ct_mode', 'none')
        self.clipgrad = kwargs.get('clipgrad', False)
        self.gan_power = kwargs.get('gan_power', 0.0)
        self.gan_patch_size = kwargs.get('gan_patch_size', self.resolution // 8)
        self.gan_dims = kwargs.get('gan_dims', 16)
        self.batch_size = kwargs.get('batch_size', 8)
        self.lr = kwargs.get('lr', 5e-5)
        self.amp_mode = kwargs.get('amp_mode', 'fp16')

        if self.amp_mode == 'bf16' and not self.clipgrad:
            _logger.warning(
                "bf16模式下GradScaler被禁用，clipgrad是唯一的梯度保护。"
                "已自动开启clipgrad=True，防止大梯度破坏参数和污染优化器状态。")
            self.clipgrad = True
        self.target_iter = kwargs.get('target_iter', 0)
        self.backup_interval = kwargs.get('backup_interval', 0)
        self.enable_torch_compile = kwargs.get('enable_torch_compile', False)
        self.auto_batch = kwargs.get('auto_batch', False)
        self.gradient_accumulation_steps = max(1, kwargs.get('gradient_accumulation_steps', 1))
        self.eyes_mouth_prio = True
        self.visibility_loss_power = 0.0
        self.pretrain = kwargs.get('pretrain', False)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict) -> "AMPTrainingConfig":
        return cls(**d)


class AMPTrainer(GenericTrainer):
    def __init__(self, config: AMPTrainingConfig, model_dir: Path,
                 src_aligned_dir: Path, dst_aligned_dir: Path,
                 device: Optional[torch.device] = None,
                 progress_callback: Optional[Callable] = None,
                 preview_callback: Optional[Callable] = None):
        if config.pretrain:
            dst_aligned_dir = src_aligned_dir
        super().__init__(AMPModel, config, model_dir, src_aligned_dir, dst_aligned_dir,
                         device, progress_callback, preview_callback)

        if config.enable_torch_compile:
            _logger.info("torch.compile预热中，首次编译需要1-3分钟...")
            from faceswap.core.saehd_utils import apply_torch_compile
            apply_torch_compile(self.model, enable=True)

    def create_datasets(self):
        c = self.config
        if c.pretrain:
            uniform_yaw = True
            src_flip = True
            dst_flip = True
            ct_mode = 'none'
        else:
            uniform_yaw = getattr(c, 'uniform_yaw', False)
            src_flip = getattr(c, 'random_src_flip', True)
            dst_flip = getattr(c, 'random_dst_flip', True)
            ct_mode = getattr(c, 'ct_mode', 'none')

        from faceswap.business.saehd_dataset import SAEHDDataset
        from faceswap.shared.config import get_num_workers
        from faceswap.shared.hardware_utils import get_available_ram_gb, image_cache_size as _image_cache_size
        n_workers = max(1, get_num_workers(self.device))
        _cache_n = _image_cache_size(get_available_ram_gb(), workers=n_workers)

        src_dataset = SAEHDDataset(
            self.src_aligned_dir, resolution=c.resolution, face_type=c.face_type,
            random_warp=getattr(c, 'random_warp', True), random_flip=src_flip,
            random_hsv_power=0.0, ct_mode=ct_mode,
            uniform_yaw=uniform_yaw, is_src=True,
            need_em_mask=True, need_vis_mask=False,
            image_cache_size=_cache_n)
        dst_dataset = SAEHDDataset(
            self.dst_aligned_dir, resolution=c.resolution, face_type=c.face_type,
            random_warp=getattr(c, 'random_warp', True), random_flip=dst_flip,
            random_hsv_power=0.0, ct_mode='none',
            uniform_yaw=uniform_yaw, is_src=False,
            need_em_mask=True, need_vis_mask=False,
            image_cache_size=_cache_n)
        return src_dataset, dst_dataset
