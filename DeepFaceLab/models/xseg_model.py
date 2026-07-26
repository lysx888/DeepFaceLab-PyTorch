import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        x = nn.functional.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class XSegNet(nn.Module):
    def __init__(self, resolution: int = 256, base_ch: int = 32):
        super().__init__()
        self._resolution = resolution

        enc_channels = [base_ch * (2 ** i) for i in range(6)]
        enc_channels[0] = base_ch

        self.enc0 = _ConvBlock(3, enc_channels[0])
        self.enc1 = _DownBlock(enc_channels[0], enc_channels[1])
        self.enc2 = _DownBlock(enc_channels[1], enc_channels[2])
        self.enc3 = _DownBlock(enc_channels[2], enc_channels[3])
        self.enc4 = _DownBlock(enc_channels[3], enc_channels[4])

        self._extra_level = resolution >= 512
        if self._extra_level:
            self.enc5 = _DownBlock(enc_channels[4], enc_channels[5])

        if self._extra_level:
            self.dec5 = _UpBlock(enc_channels[5] + enc_channels[4], enc_channels[3])
            dec4_in = enc_channels[3] + enc_channels[3]
        else:
            dec4_in = enc_channels[4] + enc_channels[3]
        self.dec4 = _UpBlock(dec4_in, enc_channels[2])

        self.dec3 = _UpBlock(enc_channels[2] + enc_channels[2], enc_channels[1])
        self.dec2 = _UpBlock(enc_channels[1] + enc_channels[1], enc_channels[0])
        self.dec1 = _UpBlock(enc_channels[0] + enc_channels[0], enc_channels[0])

        self.final = nn.Conv2d(enc_channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        if self._extra_level:
            e5 = self.enc5(e4)
            d5 = self.dec5(e5, e4)
            d4 = self.dec4(d5, e3)
        else:
            d4 = self.dec4(e4, e3)

        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        return self.final(d1)

    @property
    def resolution(self) -> int:
        return self._resolution
