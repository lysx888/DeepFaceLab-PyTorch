"""AMP模型架构的 PyTorch 原生实现。

AMP特点：
- Encoder: 5 downscale + 2 res + pixel_norm + dense
- 两个独立Inter (inter_src, inter_dst) + morph机制
- 共享一个Decoder，4 upscale + 4 res + depth_to_space输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from faceswap.models.saehd.saehd_arch import Downscale, Upscale, ResidualBlock, pixel_norm


class AMPEncoder(nn.Module):
    def __init__(self, in_ch=3, e_ch=64, resolution=224, ae_dims=256):
        super().__init__()
        self.down1 = Downscale(in_ch, e_ch, kernel_size=5)
        self.res1 = ResidualBlock(e_ch, kernel_size=3)
        self.down2 = Downscale(e_ch, e_ch * 2, kernel_size=5)
        self.down3 = Downscale(e_ch * 2, e_ch * 4, kernel_size=5)
        self.down4 = Downscale(e_ch * 4, e_ch * 8, kernel_size=5)
        self.down5 = Downscale(e_ch * 8, e_ch * 8, kernel_size=5)
        self.res5 = ResidualBlock(e_ch * 8, kernel_size=3)
        flat_size = (resolution // 32) ** 2 * e_ch * 8
        self.dense1 = nn.Linear(flat_size, ae_dims)

    def forward(self, x):
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.res5(x)
        x = torch.flatten(x, 1)
        x = pixel_norm(x, dim=-1)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.dense1(x.float())
        return x


class AMPInter(nn.Module):
    def __init__(self, ae_dims=256, inter_dims=1024, inter_res=7):
        super().__init__()
        self.inter_res = inter_res
        self.inter_dims = inter_dims
        self.dense2 = nn.Linear(ae_dims, inter_res * inter_res * inter_dims)

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.dense2(x.float())
            x = x.view(x.shape[0], self.inter_dims, self.inter_res, self.inter_res)
        return x


class AMPDecoder(nn.Module):
    def __init__(self, inter_dims=1024, d_ch=64, d_mask_ch=22):
        super().__init__()
        self.upscale0 = Upscale(inter_dims, d_ch * 8, kernel_size=3)
        self.upscale1 = Upscale(d_ch * 8, d_ch * 8, kernel_size=3)
        self.upscale2 = Upscale(d_ch * 8, d_ch * 4, kernel_size=3)
        self.upscale3 = Upscale(d_ch * 4, d_ch * 2, kernel_size=3)

        self.res0 = ResidualBlock(d_ch * 8, kernel_size=3)
        self.res1 = ResidualBlock(d_ch * 8, kernel_size=3)
        self.res2 = ResidualBlock(d_ch * 4, kernel_size=3)
        self.res3 = ResidualBlock(d_ch * 2, kernel_size=3)

        self.out_conv = nn.Conv2d(d_ch * 2, 3, kernel_size=1)
        self.out_conv1 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)

        self.upscalem0 = Upscale(inter_dims, d_mask_ch * 8, kernel_size=3)
        self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * 8, kernel_size=3)
        self.upscalem2 = Upscale(d_mask_ch * 8, d_mask_ch * 4, kernel_size=3)
        self.upscalem3 = Upscale(d_mask_ch * 4, d_mask_ch * 2, kernel_size=3)
        self.upscalem4 = Upscale(d_mask_ch * 2, d_mask_ch * 1, kernel_size=3)
        self.out_convm = nn.Conv2d(d_mask_ch * 1, 1, kernel_size=1)

    def forward(self, z):
        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)
        x = self.upscale3(x)
        x = self.res3(x)

        x = torch.sigmoid(F.pixel_shuffle(torch.cat([
            self.out_conv(x), self.out_conv1(x), self.out_conv2(x), self.out_conv3(x)
        ], dim=1), 2))

        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        m = self.upscalem3(m)
        m = self.upscalem4(m)
        m = torch.sigmoid(self.out_convm(m))
        return x, m
