import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FRNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = nn.Parameter(torch.tensor([eps]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x / torch.sqrt(nu2 + self.eps.abs())
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class TLU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.tau = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, self.tau.view(1, -1, 1, 1))


class _BlurPool(nn.Module):
    _BINOMIAL = {1: [1.], 2: [1., 1.], 3: [1., 2., 1.], 4: [1., 3., 3., 1.],
                 5: [1., 4., 6., 4., 1.], 6: [1., 5., 10., 10., 5., 1.],
                 7: [1., 6., 15., 20., 15., 6., 1.]}

    def __init__(self, channels: int, filt_size: int = 3):
        super().__init__()
        self._filt_size = filt_size
        self._pad0 = (filt_size - 1) // 2
        self._pad1 = int(np.ceil(1.0 * (filt_size - 1) / 2))
        a = np.array(self._BINOMIAL.get(filt_size, [1.]), dtype=np.float32)
        a = a[:, None] * a[None, :]
        a = a / a.sum()
        self.register_buffer('_kernel', torch.from_numpy(a[None, None, :, :]).repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self._pad0, self._pad1, self._pad0, self._pad1), mode='constant', value=0.0)
        return F.conv2d(x, self._kernel, stride=2, groups=x.shape[1])


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.frn = FRNorm2d(out_ch)
        self.tlu = TLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tlu(self.frn(self.conv(x)))


class _UpConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 0)
        self.frn = FRNorm2d(out_ch)
        self.tlu = TLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x[:, :, :-1, :-1]
        return self.tlu(self.frn(x))


class XSegNet(nn.Module):
    def __init__(self, resolution: int = 256, base_ch: int = 32):
        super().__init__()
        if resolution % 64 != 0:
            raise ValueError(f"resolution must be a multiple of 64, got {resolution}")
        self._resolution = resolution
        self._bottleneck_size = resolution // 64
        b = base_ch

        self.conv01 = _ConvBlock(3, b)
        self.conv02 = _ConvBlock(b, b)
        self.bp0 = _BlurPool(b, filt_size=4)

        self.conv11 = _ConvBlock(b, b * 2)
        self.conv12 = _ConvBlock(b * 2, b * 2)
        self.bp1 = _BlurPool(b * 2, filt_size=3)

        self.conv21 = _ConvBlock(b * 2, b * 4)
        self.conv22 = _ConvBlock(b * 4, b * 4)
        self.bp2 = _BlurPool(b * 4, filt_size=2)

        self.conv31 = _ConvBlock(b * 4, b * 8)
        self.conv32 = _ConvBlock(b * 8, b * 8)
        self.conv33 = _ConvBlock(b * 8, b * 8)
        self.bp3 = _BlurPool(b * 8, filt_size=2)

        self.conv41 = _ConvBlock(b * 8, b * 8)
        self.conv42 = _ConvBlock(b * 8, b * 8)
        self.conv43 = _ConvBlock(b * 8, b * 8)
        self.bp4 = _BlurPool(b * 8, filt_size=2)

        self.conv51 = _ConvBlock(b * 8, b * 8)
        self.conv52 = _ConvBlock(b * 8, b * 8)
        self.conv53 = _ConvBlock(b * 8, b * 8)
        self.bp5 = _BlurPool(b * 8, filt_size=2)

        bs = self._bottleneck_size
        self.dense1 = nn.Linear(bs * bs * b * 8, 512)
        self.dense2 = nn.Linear(512, bs * bs * b * 8)

        self.up5 = _UpConvBlock(b * 8, b * 4)
        self.uconv53 = _ConvBlock(b * 12, b * 8)
        self.uconv52 = _ConvBlock(b * 8, b * 8)
        self.uconv51 = _ConvBlock(b * 8, b * 8)

        self.up4 = _UpConvBlock(b * 8, b * 4)
        self.uconv43 = _ConvBlock(b * 12, b * 8)
        self.uconv42 = _ConvBlock(b * 8, b * 8)
        self.uconv41 = _ConvBlock(b * 8, b * 8)

        self.up3 = _UpConvBlock(b * 8, b * 4)
        self.uconv33 = _ConvBlock(b * 12, b * 8)
        self.uconv32 = _ConvBlock(b * 8, b * 8)
        self.uconv31 = _ConvBlock(b * 8, b * 8)

        self.up2 = _UpConvBlock(b * 8, b * 4)
        self.uconv22 = _ConvBlock(b * 8, b * 4)
        self.uconv21 = _ConvBlock(b * 4, b * 4)

        self.up1 = _UpConvBlock(b * 4, b * 2)
        self.uconv12 = _ConvBlock(b * 4, b * 2)
        self.uconv11 = _ConvBlock(b * 2, b * 2)

        self.up0 = _UpConvBlock(b * 2, b)
        self.uconv02 = _ConvBlock(b * 2, b)
        self.uconv01 = _ConvBlock(b, b)

        self.out_conv = nn.Conv2d(b, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor, skip_enabled: bool = True) -> torch.Tensor:
        x = self.conv01(x)
        x = x0 = self.conv02(x)
        x = self.bp0(x)

        x = self.conv11(x)
        x = x1 = self.conv12(x)
        x = self.bp1(x)

        x = self.conv21(x)
        x = x2 = self.conv22(x)
        x = self.bp2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = x3 = self.conv33(x)
        x = self.bp3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = x4 = self.conv43(x)
        x = self.bp4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = x5 = self.conv53(x)
        x = self.bp5(x)

        b = x.shape[0]
        bs = self._bottleneck_size
        x = x.reshape(b, -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.reshape(b, -1, bs, bs)

        x = self.up5(x)
        s5 = x5 if skip_enabled else torch.zeros_like(x5)
        x = self.uconv53(torch.cat([x, s5], dim=1))
        x = self.uconv52(x)
        x = self.uconv51(x)

        x = self.up4(x)
        s4 = x4 if skip_enabled else torch.zeros_like(x4)
        x = self.uconv43(torch.cat([x, s4], dim=1))
        x = self.uconv42(x)
        x = self.uconv41(x)

        x = self.up3(x)
        s3 = x3 if skip_enabled else torch.zeros_like(x3)
        x = self.uconv33(torch.cat([x, s3], dim=1))
        x = self.uconv32(x)
        x = self.uconv31(x)

        x = self.up2(x)
        s2 = x2 if skip_enabled else torch.zeros_like(x2)
        x = self.uconv22(torch.cat([x, s2], dim=1))
        x = self.uconv21(x)

        x = self.up1(x)
        s1 = x1 if skip_enabled else torch.zeros_like(x1)
        x = self.uconv12(torch.cat([x, s1], dim=1))
        x = self.uconv11(x)

        x = self.up0(x)
        s0 = x0 if skip_enabled else torch.zeros_like(x0)
        x = self.uconv02(torch.cat([x, s0], dim=1))
        x = self.uconv01(x)

        return self.out_conv(x)

    @property
    def resolution(self) -> int:
        return self._resolution
