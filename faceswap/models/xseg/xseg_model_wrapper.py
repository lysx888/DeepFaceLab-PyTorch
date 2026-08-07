import json
from pathlib import Path

import torch
import torch.nn as nn

from faceswap.models.base_model import BaseModel
from faceswap.models.xseg_model import XSegNet
from faceswap.shared.logger import get_logger

_logger = get_logger("xseg_model_wrapper")


class XSegTrainingConfig:
    def __init__(self, **kwargs):
        self.resolution = kwargs.get('resolution', 256)
        self.face_type = kwargs.get('face_type', 'wf')
        self.batch_size = max(1, kwargs.get('batch_size', 4))
        self.learning_rate = max(1e-7, min(kwargs.get('learning_rate', 1e-4), 1e-2))
        self.pretrain = kwargs.get('pretrain', False)
        self.target_iter = max(0, kwargs.get('target_iter', 100000))
        self.amp_mode = kwargs.get('amp_mode', 'bf16')
        self.lr_dropout = max(0.0, min(kwargs.get('lr_dropout', 0.3), 1.0))
        self.pretrain_iter = max(0, kwargs.get('pretrain_iter', 10000))

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict) -> "XSegTrainingConfig":
        return cls(**d)


class XSegModel(BaseModel):
    _model_prefix = "XSeg"
    _config_filename = "XSeg_config.json"
    _param_labels = {
        "resolution": "分辨率",
        "face_type": "人脸类型",
        "batch_size": "批次大小",
        "learning_rate": "学习率",
        "pretrain": "预训练模式",
        "target_iter": "目标迭代数",
        "amp_mode": "混合精度",
        "lr_dropout": "lr_dropout",
        "pretrain_iter": "预训练迭代数",
    }

    def __init__(self, config: XSegTrainingConfig, model_dir: Path, device: torch.device):
        self.xseg_net = None
        super().__init__(config, model_dir, device)

    def build(self) -> None:
        resolution = self.config.resolution
        if self.config.face_type == "head":
            resolution = max(resolution, 384)
            self.config.resolution = resolution
        self.xseg_net = XSegNet(resolution=resolution)
        self.register_module('xseg_net', self.xseg_net)

    def forward(self, *args, **kwargs) -> dict:
        return self.xseg_net(*args, **kwargs)

    def compute_loss(self, batch_src: dict, batch_dst: dict, fw: dict) -> dict:
        raise NotImplementedError("XSegModel does not use SAEHD-style compute_loss")

    def get_preview_section_names(self) -> list[str]:
        return ["XSeg pretrain (gray recon)", "XSeg training faces",
                "XSeg src faces", "XSeg dst faces"]

    def generate_preview_data(self, src_dataset, dst_dataset,
                               src_indices: list[int],
                               dst_indices: list[int]) -> dict[str, list]:
        return {}

    def build_optimizers(self) -> None:
        lr = self.config.learning_rate
        target_iter = self.config.target_iter
        optimizer = torch.optim.RMSprop(self.xseg_net.parameters(), lr=lr, alpha=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=target_iter, eta_min=lr * 0.01)
        self.register_optimizer('xseg_opt', optimizer)
        self._scheduler = scheduler

    def try_load(self) -> None:
        super().try_load()
        if hasattr(self, '_scheduler'):
            iter_count = self._aux_state.get('iter_count', 0)
            self._scheduler.last_epoch = iter_count
