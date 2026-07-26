from typing import Optional

import torch
import torch.nn as nn

from DeepFaceLab.models.base_model import BaseModel


class Quick96Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, base_ch),
            ConvBlock(base_ch, base_ch * 2, stride=2),
            ConvBlock(base_ch * 2, base_ch * 4, stride=2),
            ConvBlock(base_ch * 4, base_ch * 8, stride=2),
            ConvBlock(base_ch * 8, base_ch * 16, stride=2),
        )
        self.out_ch = base_ch * 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Quick96Decoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            UpBlock(in_ch, in_ch // 2),
            UpBlock(in_ch // 2, in_ch // 4),
            UpBlock(in_ch // 4, in_ch // 8),
            UpBlock(in_ch // 8, in_ch // 16),
        )
        self.final = nn.Sequential(
            nn.Conv2d(in_ch // 16, out_ch, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.net(x))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class Quick96Model(BaseModel):
    def __init__(self, resolution: int = 96, base_ch: int = 32):
        super().__init__(name="Quick96", resolution=resolution)
        self._base_ch = base_ch

        self._encoder = Quick96Encoder(in_ch=3, base_ch=base_ch)
        enc_out_ch = self._encoder.out_ch

        self._decoder_src = Quick96Decoder(in_ch=enc_out_ch, out_ch=3)
        self._decoder_dst = Quick96Decoder(in_ch=enc_out_ch, out_ch=3)

    def get_encoder(self) -> nn.Module:
        return self._encoder

    def get_decoder_src(self) -> nn.Module:
        return self._decoder_src

    def get_decoder_dst(self) -> nn.Module:
        return self._decoder_dst

    def get_inter(self) -> Optional[nn.Module]:
        return None

    def forward_src(self, x: torch.Tensor) -> torch.Tensor:
        return self._decoder_src(self._encoder(x))

    def forward_dst(self, x: torch.Tensor) -> torch.Tensor:
        return self._decoder_dst(self._encoder(x))

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["base_ch"] = self._base_ch
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> "Quick96Model":
        return cls(
            resolution=config.get("resolution", 96),
            base_ch=config.get("base_ch", 32),
        )
