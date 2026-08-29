from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from faceswap.shared.logger import get_logger

_logger = get_logger("if_landmark_arch")

_NUM_LANDMARKS = 106
_INPUT_SIZE = 192
_BN_EPS = 1e-3
_BN_MOMENTUM = 0.1


class _ConvBNPRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.prelu = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class IFLandmarkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = _ConvBNPRelu(3, 16, 3, 2, 1)
        self.conv_2_dw = _ConvBNPRelu(16, 16, 3, 1, 1, 16)
        self.conv_2 = _ConvBNPRelu(16, 32, 1, 1, 0)
        self.conv_3_dw = _ConvBNPRelu(32, 32, 3, 2, 1, 32)
        self.conv_3 = _ConvBNPRelu(32, 64, 1, 1, 0)
        self.conv_4_dw = _ConvBNPRelu(64, 64, 3, 1, 1, 64)
        self.conv_4 = _ConvBNPRelu(64, 64, 1, 1, 0)
        self.conv_5_dw = _ConvBNPRelu(64, 64, 3, 2, 1, 64)
        self.conv_5 = _ConvBNPRelu(64, 128, 1, 1, 0)
        self.conv_6_dw = _ConvBNPRelu(128, 128, 3, 1, 1, 128)
        self.conv_6 = _ConvBNPRelu(128, 128, 1, 1, 0)
        self.conv_7_dw = _ConvBNPRelu(128, 128, 3, 2, 1, 128)
        self.conv_7 = _ConvBNPRelu(128, 256, 1, 1, 0)
        self.conv_8_dw = _ConvBNPRelu(256, 256, 3, 1, 1, 256)
        self.conv_8 = _ConvBNPRelu(256, 256, 1, 1, 0)
        self.conv_9_dw = _ConvBNPRelu(256, 256, 3, 1, 1, 256)
        self.conv_9 = _ConvBNPRelu(256, 256, 1, 1, 0)
        self.conv_10_dw = _ConvBNPRelu(256, 256, 3, 1, 1, 256)
        self.conv_10 = _ConvBNPRelu(256, 256, 1, 1, 0)
        self.conv_11_dw = _ConvBNPRelu(256, 256, 3, 1, 1, 256)
        self.conv_11 = _ConvBNPRelu(256, 256, 1, 1, 0)
        self.conv_12_dw = _ConvBNPRelu(256, 256, 3, 1, 1, 256)
        self.conv_12 = _ConvBNPRelu(256, 256, 1, 1, 0)
        self.conv_13_dw = _ConvBNPRelu(256, 256, 3, 2, 1, 256)
        self.conv_13 = _ConvBNPRelu(256, 512, 1, 1, 0)
        self.conv_14_dw = _ConvBNPRelu(512, 512, 3, 1, 1, 512)
        self.conv_14 = _ConvBNPRelu(512, 512, 1, 1, 0)
        self.conv_15 = _ConvBNPRelu(512, 64, 3, 2, 1)
        self.fc1 = nn.Linear(576, 212)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2_dw(x); x = self.conv_2(x)
        x = self.conv_3_dw(x); x = self.conv_3(x)
        x = self.conv_4_dw(x); x = self.conv_4(x)
        x = self.conv_5_dw(x); x = self.conv_5(x)
        x = self.conv_6_dw(x); x = self.conv_6(x)
        x = self.conv_7_dw(x); x = self.conv_7(x)
        x = self.conv_8_dw(x); x = self.conv_8(x)
        x = self.conv_9_dw(x); x = self.conv_9(x)
        x = self.conv_10_dw(x); x = self.conv_10(x)
        x = self.conv_11_dw(x); x = self.conv_11(x)
        x = self.conv_12_dw(x); x = self.conv_12(x)
        x = self.conv_13_dw(x); x = self.conv_13(x)
        x = self.conv_14_dw(x); x = self.conv_14(x)
        x = self.conv_15(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return x

    @property
    def input_size(self) -> int:
        return _INPUT_SIZE

    @property
    def num_landmarks(self) -> int:
        return _NUM_LANDMARKS

    def load_pretrained_onnx(self, onnx_path: str | Path) -> None:
        import onnx
        m = onnx.load(str(onnx_path))
        onnx_weights = {}
        for init in m.graph.initializer:
            onnx_weights[init.name] = onnx.numpy_helper.to_array(init)

        new_state = {}
        for name, param in self.state_dict().items():
            parts = name.split('.')
            block = parts[0]
            onnx_name = self._map_to_onnx_name(block, parts)
            if onnx_name is None:
                continue
            arr = onnx_weights.get(onnx_name)
            if arr is None:
                _logger.warning(f"ONNX权重缺失: {onnx_name} (对应 {name})")
                continue
            arr = np.asarray(arr)
            if parts[1] == "prelu":
                arr = arr.squeeze()
            new_state[name] = torch.from_numpy(arr.copy()).float()

        missing, unexpected = self.load_state_dict(new_state, strict=False)
        if missing:
            _logger.warning(f"未加载的参数: {missing}")
        if unexpected:
            _logger.warning(f"多余的参数: {unexpected}")
        _logger.info(f"从ONNX加载预训练权重: {onnx_path}")

    @staticmethod
    def _map_to_onnx_name(block: str, parts: list[str]) -> str | None:
        if block == "fc1":
            if parts[1] == "weight":
                return "fc1_weight"
            if parts[1] == "bias":
                return "fc1_bias"
            return None
        ptype = parts[1]
        wtype = parts[2]
        if ptype == "conv" and wtype == "weight":
            return f"{block}_conv2d_weight"
        if ptype == "bn":
            return {
                "weight": f"{block}_batchnorm_gamma",
                "bias": f"{block}_batchnorm_beta",
                "running_mean": f"{block}_batchnorm_moving_mean",
                "running_var": f"{block}_batchnorm_moving_var",
            }.get(wtype)
        if ptype == "prelu" and wtype == "weight":
            return f"{block}_relu_gamma"
        return None

    def export_onnx(self, path: str | Path) -> None:
        self.eval()
        wrapper = _PreprocessWrapper(self).eval()
        device = next(self.parameters()).device
        dummy = (torch.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE, device=device) * 255.0)
        torch.onnx.export(
            wrapper,
            dummy,
            str(path),
            input_names=["data"],
            output_names=["fc1"],
            dynamic_axes={"data": {0: "batch"}, "fc1": {0: "batch"}},
            opset_version=18,
            dynamo=False,
        )
        _logger.info(f"ONNX exported: {path}")


class _PreprocessWrapper(nn.Module):
    def __init__(self, net: "IFLandmarkNet"):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - 127.5) / 128.0
        return self.net(x)
