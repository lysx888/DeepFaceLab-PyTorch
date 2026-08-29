"""DFL DeepFakeArchi 的 PyTorch 原生实现，支持全部 archi opts。

opts 字符串可包含:
  'u' - pixel_norm（编码器输出归一化）
  'd' - 分辨率倍增（Inter 用 lowest_dense_res//32，Decoder 末层 pixel_shuffle×2）
  't' - 深层变体（Encoder 5 downscale+2 res，Inter 无 upscale1，Decoder 4 层）
  'c' - cos 激活（x*cos(x) 替代 leaky_relu）

pixel_shuffle 与 DFL 的 depth_to_space 通道排列完全相同，功能等价。
padding=kernel_size//2 与 DFL 的 TF 'SAME' 在 stride=2 时有1像素偏移，从零训练可吸收。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(x, alpha=0.1, use_cos=False):
    if use_cos:
        return x * torch.cos(x)
    return F.leaky_relu(x, alpha)


class Downscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, use_cos=False):
        super().__init__()
        self.use_cos = use_cos
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=2, padding=kernel_size // 2)

    def forward(self, x):
        return _activation(self.conv(x), 0.1, self.use_cos)


class DownscaleBlock(nn.Module):
    def __init__(self, in_ch, ch, n_downscales, kernel_size=5, use_cos=False):
        super().__init__()
        self.downs = nn.ModuleList()
        last_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * min(2 ** i, 8)
            self.downs.append(Downscale(last_ch, cur_ch, kernel_size, use_cos))
            last_ch = cur_ch
        self._out_ch = last_ch

    def forward(self, x):
        for down in self.downs:
            x = down(x)
        return x

    def get_out_ch(self):
        return self._out_ch


class Upscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_cos=False):
        super().__init__()
        self.use_cos = use_cos
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x):
        x = _activation(self.conv(x), 0.1, self.use_cos)
        return F.pixel_shuffle(x, 2)


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, use_cos=False):
        super().__init__()
        self.use_cos = use_cos
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, inp):
        x = _activation(self.conv1(inp), 0.2, self.use_cos)
        x = self.conv2(x)
        return _activation(inp + x, 0.2, self.use_cos)


def pixel_norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=dim, keepdim=True) + eps)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, e_ch=64, resolution=128, opts=''):
        super().__init__()
        self.resolution = resolution
        self.opts = opts
        self.use_cos = 'c' in opts
        use_t = 't' in opts

        if use_t:
            self.down1 = Downscale(in_ch, e_ch, kernel_size=5, use_cos=self.use_cos)
            self.res1 = ResidualBlock(e_ch, kernel_size=3, use_cos=self.use_cos)
            self.down2 = Downscale(e_ch, e_ch * 2, kernel_size=5, use_cos=self.use_cos)
            self.down3 = Downscale(e_ch * 2, e_ch * 4, kernel_size=5, use_cos=self.use_cos)
            self.down4 = Downscale(e_ch * 4, e_ch * 8, kernel_size=5, use_cos=self.use_cos)
            self.down5 = Downscale(e_ch * 8, e_ch * 8, kernel_size=5, use_cos=self.use_cos)
            self.res5 = ResidualBlock(e_ch * 8, kernel_size=3, use_cos=self.use_cos)
        else:
            self.down1 = DownscaleBlock(in_ch, e_ch, n_downscales=4, kernel_size=5, use_cos=self.use_cos)

        self._out_ch = e_ch * 8
        self._out_res = resolution // (32 if use_t else 16)

    def forward(self, x):
        if 't' in self.opts:
            x = self.down1(x)
            x = self.res1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.down5(x)
            x = self.res5(x)
        else:
            x = self.down1(x)
        x = torch.flatten(x, 1)
        if 'u' in self.opts:
            x = pixel_norm(x, dim=-1)
        return x

    def get_out_ch(self):
        return self._out_ch

    def get_out_res(self, res):
        return res // (32 if 't' in self.opts else 16)


class Inter(nn.Module):
    def __init__(self, in_ch, ae_ch, ae_out_ch, resolution=128, opts=''):
        super().__init__()
        self.opts = opts
        self.ae_out_ch = ae_out_ch
        use_t = 't' in self.opts
        self.lowest_dense_res = resolution // (32 if 'd' in self.opts else 16)

        self.dense1 = nn.Linear(in_ch, ae_ch)
        self.dense2 = nn.Linear(ae_ch, self.lowest_dense_res * self.lowest_dense_res * ae_out_ch)
        if not use_t:
            self.upscale1 = Upscale(ae_out_ch, ae_out_ch, use_cos='c' in self.opts)

        self._out_ch = ae_out_ch
        self._out_res = self.lowest_dense_res * (2 if not use_t else 1)

    def forward(self, inp):
        with torch.amp.autocast(device_type=inp.device.type, enabled=False):
            inp = inp.float()
            x = self.dense1(inp)
            x = self.dense2(x)
            x = x.view(x.shape[0], self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res)
        if 't' not in self.opts:
            x = self.upscale1(x)
        return x

    def get_out_ch(self):
        return self._out_ch

    def get_out_res(self):
        return self._out_res


class Decoder(nn.Module):
    def __init__(self, in_ch, d_ch, d_mask_ch, opts=''):
        super().__init__()
        self.opts = opts
        use_cos = 'c' in opts
        use_t = 't' in opts
        use_d = 'd' in opts

        if not use_t:
            self.upscale0 = Upscale(in_ch, d_ch * 8, kernel_size=3, use_cos=use_cos)
            self.upscale1 = Upscale(d_ch * 8, d_ch * 4, kernel_size=3, use_cos=use_cos)
            self.upscale2 = Upscale(d_ch * 4, d_ch * 2, kernel_size=3, use_cos=use_cos)
            self.res0 = ResidualBlock(d_ch * 8, kernel_size=3, use_cos=use_cos)
            self.res1 = ResidualBlock(d_ch * 4, kernel_size=3, use_cos=use_cos)
            self.res2 = ResidualBlock(d_ch * 2, kernel_size=3, use_cos=use_cos)
        else:
            self.upscale0 = Upscale(in_ch, d_ch * 8, kernel_size=3, use_cos=use_cos)
            self.upscale1 = Upscale(d_ch * 8, d_ch * 8, kernel_size=3, use_cos=use_cos)
            self.upscale2 = Upscale(d_ch * 8, d_ch * 4, kernel_size=3, use_cos=use_cos)
            self.upscale3 = Upscale(d_ch * 4, d_ch * 2, kernel_size=3, use_cos=use_cos)
            self.res0 = ResidualBlock(d_ch * 8, kernel_size=3, use_cos=use_cos)
            self.res1 = ResidualBlock(d_ch * 8, kernel_size=3, use_cos=use_cos)
            self.res2 = ResidualBlock(d_ch * 4, kernel_size=3, use_cos=use_cos)
            self.res3 = ResidualBlock(d_ch * 2, kernel_size=3, use_cos=use_cos)

        self.out_conv = nn.Conv2d(d_ch * 2, 3, kernel_size=1)
        if use_d:
            self.out_conv1 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)
            self.out_conv2 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)
            self.out_conv3 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)

        if not use_t:
            self.upscalem0 = Upscale(in_ch, d_mask_ch * 8, kernel_size=3, use_cos=use_cos)
            self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * 4, kernel_size=3, use_cos=use_cos)
            self.upscalem2 = Upscale(d_mask_ch * 4, d_mask_ch * 2, kernel_size=3, use_cos=use_cos)
            if use_d:
                self.upscalem3 = Upscale(d_mask_ch * 2, d_mask_ch * 1, kernel_size=3, use_cos=use_cos)
                self.out_convm = nn.Conv2d(d_mask_ch * 1, 1, kernel_size=1)
            else:
                self.out_convm = nn.Conv2d(d_mask_ch * 2, 1, kernel_size=1)
        else:
            self.upscalem0 = Upscale(in_ch, d_mask_ch * 8, kernel_size=3, use_cos=use_cos)
            self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * 8, kernel_size=3, use_cos=use_cos)
            self.upscalem2 = Upscale(d_mask_ch * 8, d_mask_ch * 4, kernel_size=3, use_cos=use_cos)
            self.upscalem3 = Upscale(d_mask_ch * 4, d_mask_ch * 2, kernel_size=3, use_cos=use_cos)
            if use_d:
                self.upscalem4 = Upscale(d_mask_ch * 2, d_mask_ch * 1, kernel_size=3, use_cos=use_cos)
                self.out_convm = nn.Conv2d(d_mask_ch * 1, 1, kernel_size=1)
            else:
                self.out_convm = nn.Conv2d(d_mask_ch * 2, 1, kernel_size=1)

    def forward(self, z):
        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)

        if 't' in self.opts:
            x = self.upscale3(x)
            x = self.res3(x)

        if 'd' in self.opts:
            x = torch.sigmoid(F.pixel_shuffle(torch.cat([
                self.out_conv(x), self.out_conv1(x), self.out_conv2(x), self.out_conv3(x)
            ], dim=1), 2))
        else:
            x = torch.sigmoid(self.out_conv(x))

        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)

        if 't' in self.opts:
            m = self.upscalem3(m)
            if 'd' in self.opts:
                m = self.upscalem4(m)
        else:
            if 'd' in self.opts:
                m = self.upscalem3(m)

        m = torch.sigmoid(self.out_convm(m))
        return x, m
