from typing import Optional

import torch
import torch.nn as nn

from DeepFaceLab.models.base_model import BaseModel


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


class SAEHDEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64, down_steps: int = 5):
        super().__init__()
        layers = [ConvBlock(in_ch, base_ch)]
        ch = base_ch
        for _ in range(down_steps):
            layers.append(ConvBlock(ch, ch * 2, stride=2))
            ch *= 2
        self.layers = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SAEHDDecoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 3, up_steps: int = 5, use_liae: bool = False, latent_ch: int = 0):
        super().__init__()
        self.use_liae = use_liae
        self.latent_proj = None
        actual_in = in_ch
        if use_liae and latent_ch > 0:
            self.latent_proj = nn.Conv2d(latent_ch + in_ch, in_ch, 1, bias=False)
        layers = []
        ch = in_ch
        for _ in range(up_steps):
            next_ch = max(ch // 2, 64)
            layers.append(UpBlock(ch, next_ch))
            ch = next_ch
        self.ups = nn.Sequential(*layers)
        self.final_conv = nn.Sequential(
            nn.Conv2d(ch, out_ch, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_liae and self.latent_proj is not None and latent is not None:
            x = self.latent_proj(torch.cat([x, latent], dim=1))
        return self.final_conv(self.ups(x))


class SAEHDInter(nn.Module):
    def __init__(self, in_ch: int, latent_ch: int = 128):
        super().__init__()
        self.fc = nn.Linear(in_ch, latent_ch)
        self.norm = nn.LayerNorm(latent_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        pooled = x.mean(dim=[2, 3])
        out = self.norm(self.fc(pooled))
        return out.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])


class SAEHDModel(BaseModel):
    def __init__(self, resolution: int = 128, architecture: str = "df", base_ch: int = 64):
        super().__init__(name="SAEHD", resolution=resolution)
        self._architecture = architecture
        self._base_ch = base_ch
        self._use_liae = architecture == "liae"

        down_steps = 0
        r = resolution
        while r > 4:
            r //= 2
            down_steps += 1

        self._encoder = SAEHDEncoder(in_ch=3, base_ch=base_ch, down_steps=down_steps)
        enc_out_ch = self._encoder.out_ch

        self._decoder_src = SAEHDDecoder(
            in_ch=enc_out_ch, out_ch=3, up_steps=down_steps,
            use_liae=self._use_liae, latent_ch=enc_out_ch if self._use_liae else 0,
        )
        self._decoder_dst = SAEHDDecoder(
            in_ch=enc_out_ch, out_ch=3, up_steps=down_steps,
            use_liae=self._use_liae, latent_ch=enc_out_ch if self._use_liae else 0,
        )

        if self._use_liae:
            self._inter = SAEHDInter(in_ch=enc_out_ch, latent_ch=enc_out_ch)

    def get_encoder(self) -> nn.Module:
        return self._encoder

    def get_decoder_src(self) -> nn.Module:
        return self._decoder_src

    def get_decoder_dst(self) -> nn.Module:
        return self._decoder_dst

    def get_inter(self) -> Optional[nn.Module]:
        return self._inter

    def forward_src(self, x: torch.Tensor) -> torch.Tensor:
        enc = self._encoder(x)
        if self._use_liae and self._inter is not None:
            latent = self._inter(enc)
            return self._decoder_src(enc, latent)
        return self._decoder_src(enc)

    def forward_dst(self, x: torch.Tensor) -> torch.Tensor:
        enc = self._encoder(x)
        if self._use_liae and self._inter is not None:
            latent = self._inter(enc)
            return self._decoder_dst(enc, latent)
        return self._decoder_dst(enc)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["architecture"] = self._architecture
        cfg["base_ch"] = self._base_ch
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> "SAEHDModel":
        return cls(
            resolution=config.get("resolution", 128),
            architecture=config.get("architecture", "df"),
            base_ch=config.get("base_ch", 64),
        )
