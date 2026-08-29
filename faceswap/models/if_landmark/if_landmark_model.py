from pathlib import Path

import torch
import torch.nn as nn

from faceswap.models.base_model import BaseModel
from faceswap.models.if_landmark.if_landmark_arch import IFLandmarkNet
from faceswap.shared.logger import get_logger

_logger = get_logger("if_landmark_model")


class IFLandmarkTrainingConfig:
    def __init__(self, **kwargs):
        self.batch_size = max(1, kwargs.get('batch_size', 32))
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.momentum = kwargs.get('momentum', 0.9)
        self.weight_decay = kwargs.get('weight_decay', 0.0005)
        self.max_epochs = max(1, kwargs.get('max_epochs', 30))
        self.lr_steps = kwargs.get('lr_steps', [15, 25, 28])
        self.input_size = kwargs.get('input_size', 192)
        self.augment = kwargs.get('augment', True)
        self.data_dir = kwargs.get('data_dir', '')
        self.pretrained_onnx = kwargs.get('pretrained_onnx', '')

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict) -> "IFLandmarkTrainingConfig":
        return cls(**d)


class IFLandmarkModel(BaseModel):
    _model_prefix = "IFLandmark"
    _config_filename = "IFLandmark_config.json"
    _param_labels = {
        "batch_size": "批次大小",
        "learning_rate": "学习率",
        "momentum": "动量",
        "weight_decay": "权重衰减",
        "max_epochs": "最大轮数",
        "lr_steps": "学习率衰减轮次",
        "input_size": "输入尺寸",
        "augment": "数据增强",
        "data_dir": "数据目录",
        "pretrained_onnx": "预训练ONNX路径",
    }

    def __init__(self, config: IFLandmarkTrainingConfig, model_dir: Path, device: torch.device):
        self.if_net: nn.Module | None = None
        self._scheduler = None
        super().__init__(config, model_dir, device)

    def build(self) -> None:
        self.if_net = IFLandmarkNet()
        self.register_module('if_net', self.if_net)

    def forward(self, *args, **kwargs) -> dict:
        return {'pred': self.if_net(*args, **kwargs)}

    def compute_loss(self, batch_src: dict, batch_dst: dict, fw: dict) -> dict:
        raise NotImplementedError("IFLandmarkModel does not use SAEHD-style compute_loss")

    def get_preview_section_names(self) -> list[str]:
        return ["IF landmark predictions"]

    def generate_preview_data(self, src_dataset, dst_dataset,
                               src_indices: list[int],
                               dst_indices: list[int]) -> dict[str, list]:
        return {}

    def build_optimizers(self) -> None:
        lr = self.config.learning_rate
        optimizer = torch.optim.SGD(
            self.if_net.parameters(),
            lr=lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        self.register_optimizer('if_opt', optimizer)

        def lr_step_func(epoch: int) -> float:
            return 0.1 ** len([m for m in self.config.lr_steps if m <= epoch])

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lr_step_func)

    def on_pretrain_override(self) -> None:
        pretrained_path = self.config.pretrained_onnx
        if pretrained_path and Path(pretrained_path).exists():
            self.if_net.load_pretrained_onnx(pretrained_path)
        elif pretrained_path:
            _logger.warning(f"预训练权重文件不存在: {pretrained_path}")

    def try_load(self) -> None:
        super().try_load()
        if self._scheduler is not None:
            epoch = self._aux_state.get('iter_count', 0)
            self._scheduler.last_epoch = epoch

    def export_onnx(self, path: str | Path) -> None:
        self.if_net.export_onnx(str(path))
