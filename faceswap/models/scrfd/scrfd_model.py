from pathlib import Path

import torch
import torch.nn as nn

from faceswap.models.base_model import BaseModel
from faceswap.models.scrfd.scrfd_arch import SCRFDNet
from faceswap.shared.logger import get_logger

_logger = get_logger("scrfd_model")


class SCRFDTrainingConfig:
    def __init__(self, **kwargs):
        self.batch_size = max(1, kwargs.get('batch_size', 8))
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.momentum = kwargs.get('momentum', 0.9)
        self.weight_decay = kwargs.get('weight_decay', 0.0005)
        self.max_epochs = max(1, kwargs.get('max_epochs', 30))
        self.input_size = kwargs.get('input_size', 640)
        self.augment = kwargs.get('augment', True)
        self.data_dir = kwargs.get('data_dir', '')
        self.warmup_iters = kwargs.get('warmup_iters', 1500)
        self.warmup_ratio = kwargs.get('warmup_ratio', 0.001)
        self.lr_step_epochs = kwargs.get('lr_step_epochs', [20, 27])
        self.pretrained_onnx = kwargs.get('pretrained_onnx', '')

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict) -> "SCRFDTrainingConfig":
        return cls(**d)


class SCRFDModel(BaseModel):
    _model_prefix = "SCRFD"
    _config_filename = "SCRFD_config.json"
    _param_labels = {
        "batch_size": "批次大小",
        "learning_rate": "学习率",
        "momentum": "动量",
        "weight_decay": "权重衰减",
        "max_epochs": "最大轮数",
        "input_size": "输入尺寸",
        "augment": "数据增强",
        "data_dir": "数据目录",
        "warmup_iters": "预热迭代数",
        "warmup_ratio": "预热比例",
        "lr_step_epochs": "学习率衰减轮次",
        "pretrained_onnx": "预训练ONNX路径",
    }

    def __init__(self, config: SCRFDTrainingConfig, model_dir: Path, device: torch.device):
        self.scrfd_net: nn.Module | None = None
        super().__init__(config, model_dir, device)

    def build(self) -> None:
        self.scrfd_net = SCRFDNet()
        self.register_module('scrfd_net', self.scrfd_net)

    def forward(self, *args, **kwargs) -> dict:
        return {'pred': self.scrfd_net(*args, **kwargs)}

    def compute_loss(self, batch_src: dict, batch_dst: dict, fw: dict) -> dict:
        raise NotImplementedError("SCRFDModel does not use SAEHD-style compute_loss")

    def get_preview_section_names(self) -> list[str]:
        return ["SCRFD predictions"]

    def generate_preview_data(self, src_dataset, dst_dataset,
                               src_indices: list[int],
                               dst_indices: list[int]) -> dict[str, list]:
        return {}

    def try_load(self) -> None:
        super().try_load()

    def export_onnx(self, path: str | Path) -> None:
        self.scrfd_net.export_onnx(str(path), input_size=self.config.input_size)
