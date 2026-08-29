"""Quick96模型训练配置和入口。"""
from pathlib import Path
from typing import Optional, Callable
import torch

from faceswap.business.generic_trainer import GenericTrainer
from faceswap.models.quick96.quick96_model import Quick96Model


class Quick96TrainingConfig:
    def __init__(self, **kwargs):
        self.resolution = 96
        self.face_type = 'wf'
        self.ae_dims = 128
        self.e_dims = 64
        self.d_dims = 64
        self.d_mask_dims = 16
        self.masked_training = True
        self.random_warp = True
        self.random_src_flip = kwargs.get('random_src_flip', True)
        self.random_dst_flip = kwargs.get('random_dst_flip', True)
        self.uniform_yaw = False
        self.ct_mode = 'none'
        self.eyes_mouth_prio = False
        self.visibility_loss_power = 0.0
        self.batch_size = kwargs.get('batch_size', 4)
        self.lr = 2e-4
        self.amp_mode = kwargs.get('amp_mode', 'fp16')
        self.target_iter = kwargs.get('target_iter', 0)
        self.backup_interval = kwargs.get('backup_interval', 0)
        self.enable_torch_compile = kwargs.get('enable_torch_compile', False)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict) -> "Quick96TrainingConfig":
        return cls(**d)


class Quick96Trainer(GenericTrainer):
    def __init__(self, config: Quick96TrainingConfig, model_dir: Path,
                 src_aligned_dir: Path, dst_aligned_dir: Path,
                 device: Optional[torch.device] = None,
                 progress_callback: Optional[Callable] = None,
                 preview_callback: Optional[Callable] = None):
        super().__init__(Quick96Model, config, model_dir, src_aligned_dir, dst_aligned_dir,
                         device, progress_callback, preview_callback)
