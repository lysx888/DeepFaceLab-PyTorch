from typing import Optional

import torch
import torch.nn as nn

from DeepFaceLab.models.base_model import BaseModel
from DeepFaceLab.models.saehd_model import ConvBlock, UpBlock


class AMPEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64, down_steps: int = 4):
        super().__init__()
        layers = [ConvBlock(in_ch, base_ch)]
        ch = base_ch
        for _ in range(down_steps):
            layers.append(ConvBlock(ch, ch * 2, stride=2))
            ch *= 2
        self.net = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AMPDecoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 3, up_steps: int = 4):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(up_steps):
            next_ch = max(ch // 2, 64)
            layers.append(UpBlock(ch, next_ch))
            ch = next_ch
        self.ups = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.Conv2d(ch, out_ch, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.ups(x))


class AMPModel(BaseModel):
    def __init__(self, resolution: int = 128, base_ch: int = 64, src_src_mode: bool = False):
        super().__init__(name="AMP", resolution=resolution)
        self._base_ch = base_ch
        self._src_src_mode = src_src_mode

        down_steps = 0
        r = resolution
        while r > 4:
            r //= 2
            down_steps += 1

        self._encoder = AMPEncoder(in_ch=3, base_ch=base_ch, down_steps=down_steps)
        enc_out_ch = self._encoder.out_ch

        self._decoder_src = AMPDecoder(in_ch=enc_out_ch, out_ch=3, up_steps=down_steps)
        self._decoder_dst = AMPDecoder(in_ch=enc_out_ch, out_ch=3, up_steps=down_steps)

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
        if self._src_src_mode:
            return self._decoder_dst(self._encoder(x))
        return self._decoder_dst(self._encoder(x))

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["base_ch"] = self._base_ch
        cfg["src_src_mode"] = self._src_src_mode
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> "AMPModel":
        return cls(
            resolution=config.get("resolution", 128),
            base_ch=config.get("base_ch", 64),
            src_src_mode=config.get("src_src_mode", False),
        )
