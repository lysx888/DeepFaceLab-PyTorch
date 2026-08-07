import torch
import torch.nn as nn
import torch.nn.functional as F

from faceswap.core.saehd_utils import depth_to_space, pixel_norm


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_in',
                                nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Downscale(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5,
                 use_cos_act: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_cos_act = use_cos_act
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=2, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_cos_act:
            x = x * torch.cos(x)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)
        return x


class DownscaleBlock(nn.Module):
    def __init__(self, in_ch: int, ch: int, n_downscales: int = 4,
                 kernel_size: int = 5, use_cos_act: bool = False):
        super().__init__()
        self.downs = nn.ModuleList()
        last_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * min(2 ** i, 8)
            self.downs.append(Downscale(last_ch, cur_ch, kernel_size=kernel_size,
                                        use_cos_act=use_cos_act))
            last_ch = cur_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for down in self.downs:
            x = down(x)
        return x


class Upscale(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 use_cos_act: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.use_cos_act = use_cos_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_cos_act:
            x = x * torch.cos(x)
        else:
            x = F.leaky_relu(x, negative_slope=0.1)
        x = depth_to_space(x, 2)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3,
                 use_cos_act: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.use_cos_act = use_cos_act

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv1(inp)
        x = F.leaky_relu(x, negative_slope=0.1) if not self.use_cos_act else x * torch.cos(x)
        x = self.conv2(x)
        x = inp + x
        x = F.leaky_relu(x, negative_slope=0.1) if not self.use_cos_act else x * torch.cos(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, e_ch: int = 64,
                 opts: str = '', resolution: int = 128):
        super().__init__()
        self.in_ch = in_ch
        self.e_ch = e_ch
        self.opts = opts
        self.resolution = resolution
        use_cos = 'c' in opts
        use_t = 't' in opts
        use_u = 'u' in opts

        if use_t:
            self.down1 = Downscale(in_ch, e_ch, kernel_size=5, use_cos_act=use_cos)
            self.res1 = ResidualBlock(e_ch, kernel_size=3, use_cos_act=use_cos)
            self.down2 = Downscale(e_ch, e_ch * 2, kernel_size=5, use_cos_act=use_cos)
            self.down3 = Downscale(e_ch * 2, e_ch * 4, kernel_size=5, use_cos_act=use_cos)
            self.down4 = Downscale(e_ch * 4, e_ch * 8, kernel_size=5, use_cos_act=use_cos)
            self.down5 = Downscale(e_ch * 8, e_ch * 8, kernel_size=5, use_cos_act=use_cos)
            self.res5 = ResidualBlock(e_ch * 8, kernel_size=3, use_cos_act=use_cos)
            self._use_t = True
        else:
            self.downblock = DownscaleBlock(in_ch, e_ch, n_downscales=4,
                                            kernel_size=5, use_cos_act=use_cos)
            self._use_t = False

        self._use_u = use_u
        self._n_downscales = 5 if use_t else 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_t:
            x = self.down1(x)
            x = self.res1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.down5(x)
            x = self.res5(x)
        else:
            x = self.downblock(x)

        x = x.reshape(x.shape[0], -1)

        if self._use_u:
            x = pixel_norm(x)

        return x

    def get_out_res(self, res: int) -> int:
        return res // (2 ** self._n_downscales)

    def get_out_ch(self) -> int:
        return self.e_ch * 8


class Inter(nn.Module):
    def __init__(self, in_ch: int, ae_ch: int, ae_out_ch: int,
                 resolution: int = 128, opts: str = ''):
        super().__init__()
        self.in_ch = in_ch
        self.ae_ch = ae_ch
        self.ae_out_ch = ae_out_ch
        self.opts = opts

        lowest_dense_res = resolution // (32 if 'd' in opts else 16)
        self._lowest_dense_res = lowest_dense_res
        use_t = 't' in opts
        use_cos = 'c' in opts

        self.dense1 = nn.Linear(in_ch, ae_ch)
        self.dense2 = nn.Linear(ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch)

        if not use_t:
            self.upscale1 = Upscale(ae_out_ch, ae_out_ch, kernel_size=3,
                                    use_cos_act=use_cos)
        self._use_t = use_t

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.reshape(x.shape[0], self.ae_out_ch,
                       self._lowest_dense_res, self._lowest_dense_res)

        if not self._use_t:
            x = self.upscale1(x)

        return x

    def get_out_res(self) -> int:
        return self._lowest_dense_res * 2 if not self._use_t else self._lowest_dense_res

    def get_out_ch(self) -> int:
        return self.ae_out_ch


class Decoder(nn.Module):
    def __init__(self, in_ch: int, d_ch: int = 64, d_mask_ch: int = 22,
                 resolution: int = 128, opts: str = ''):
        super().__init__()
        self.opts = opts
        use_t = 't' in opts
        use_d = 'd' in opts
        use_cos = 'c' in opts

        # Image branch
        self.upscale0 = Upscale(in_ch, d_ch * 8, kernel_size=3, use_cos_act=use_cos)
        self.res0 = ResidualBlock(d_ch * 8, kernel_size=3, use_cos_act=use_cos)

        if use_t:
            self.upscale1 = Upscale(d_ch * 8, d_ch * 8, kernel_size=3, use_cos_act=use_cos)
            self.res1 = ResidualBlock(d_ch * 8, kernel_size=3, use_cos_act=use_cos)
            self.upscale2 = Upscale(d_ch * 8, d_ch * 4, kernel_size=3, use_cos_act=use_cos)
            self.res2 = ResidualBlock(d_ch * 4, kernel_size=3, use_cos_act=use_cos)
            self.upscale3 = Upscale(d_ch * 4, d_ch * 2, kernel_size=3, use_cos_act=use_cos)
            self.res3 = ResidualBlock(d_ch * 2, kernel_size=3, use_cos_act=use_cos)
        else:
            self.upscale1 = Upscale(d_ch * 8, d_ch * 4, kernel_size=3, use_cos_act=use_cos)
            self.res1 = ResidualBlock(d_ch * 4, kernel_size=3, use_cos_act=use_cos)
            self.upscale2 = Upscale(d_ch * 4, d_ch * 2, kernel_size=3, use_cos_act=use_cos)
            self.res2 = ResidualBlock(d_ch * 2, kernel_size=3, use_cos_act=use_cos)

        self.out_conv = nn.Conv2d(d_ch * 2, 3, kernel_size=1, padding=0)

        if use_d:
            self.out_conv1 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)
            self.out_conv2 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)
            self.out_conv3 = nn.Conv2d(d_ch * 2, 3, kernel_size=3, padding=1)

        # Mask branch
        self.upscalem0 = Upscale(in_ch, d_mask_ch * 8, kernel_size=3, use_cos_act=use_cos)
        self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * (8 if use_t else 4), kernel_size=3, use_cos_act=use_cos)
        if use_t:
            self.upscalem2 = Upscale(d_mask_ch * 8, d_mask_ch * 4, kernel_size=3, use_cos_act=use_cos)
            self.upscalem3 = Upscale(d_mask_ch * 4, d_mask_ch * 2, kernel_size=3, use_cos_act=use_cos)
            if use_d:
                self.upscalem4 = Upscale(d_mask_ch * 2, d_mask_ch, kernel_size=3, use_cos_act=use_cos)
                self.out_convm = nn.Conv2d(d_mask_ch, 1, kernel_size=1, padding=0)
            else:
                self.out_convm = nn.Conv2d(d_mask_ch * 2, 1, kernel_size=1, padding=0)
        else:
            self.upscalem2 = Upscale(d_mask_ch * 4, d_mask_ch * 2, kernel_size=3, use_cos_act=use_cos)
            if use_d:
                self.upscalem3 = Upscale(d_mask_ch * 2, d_mask_ch, kernel_size=3, use_cos_act=use_cos)
                self.out_convm = nn.Conv2d(d_mask_ch, 1, kernel_size=1, padding=0)
            else:
                self.out_convm = nn.Conv2d(d_mask_ch * 2, 1, kernel_size=1, padding=0)

        self._use_t = use_t
        self._use_d = use_d

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Image branch
        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)

        if self._use_t:
            x = self.upscale3(x)
            x = self.res3(x)

        if self._use_d:
            x_pre_dts = torch.cat((self.out_conv(x),
                       self.out_conv1(x),
                       self.out_conv2(x),
                       self.out_conv3(x)), dim=1)
            x = depth_to_space(x_pre_dts, 2)
            x = torch.sigmoid(x)
        else:
            x = self.out_conv(x)
            x = torch.sigmoid(x)


        # Mask branch
        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)

        if self._use_t:
            m = self.upscalem3(m)
            if self._use_d:
                m = self.upscalem4(m)
        else:
            if self._use_d:
                m = self.upscalem3(m)

        m_pre_sig = self.out_convm(m)
        m = torch.sigmoid(m_pre_sig)

        return x, m
