import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeDiscriminator(nn.Module):
    def __init__(self, in_ch: int, code_res: int, ch: int = 256):
        super().__init__()
        n_downscales = 1 + code_res // 8

        self.convs = nn.ModuleList()
        prev_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * min(2 ** i, 8)
            ks = 4 if i == 0 else 3
            self.convs.append(nn.Conv2d(prev_ch, cur_ch, kernel_size=ks,
                                        stride=2, padding=ks // 2))
            prev_ch = cur_ch

        self.out_conv = nn.Conv2d(prev_ch, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
        return self.out_conv(x)


class UNetPatchDiscriminator(nn.Module):
    def __init__(self, patch_size: int = 32, in_ch: int = 3,
                 base_ch: int = 16):
        super().__init__()
        layers = self._find_archi(patch_size)
        level_chs = {i - 1: min(base_ch * (2 ** i), 512) for i in range(len(layers) + 1)}

        self.in_conv = nn.Conv2d(in_ch, level_chs[-1], kernel_size=1, padding=0)

        self.convs = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i, (kernel_size, strides) in enumerate(layers):
            self.convs.append(nn.Conv2d(level_chs[i - 1], level_chs[i],
                                        kernel_size=kernel_size, stride=strides,
                                        padding=kernel_size // 2))
            cat_factor = 2 if i != len(layers) - 1 else 1
            self.upconvs.insert(0, nn.ConvTranspose2d(level_chs[i] * cat_factor, level_chs[i - 1],
                                                       kernel_size=kernel_size, stride=strides,
                                                       padding=kernel_size // 2))

        self.out_conv = nn.Conv2d(level_chs[-1] * 2, 1, kernel_size=1, padding=0)
        self.center_out = nn.Conv2d(level_chs[len(layers) - 1], 1, kernel_size=1, padding=0)
        self.center_conv = nn.Conv2d(level_chs[len(layers) - 1], level_chs[len(layers) - 1],
                                     kernel_size=1, padding=0)

    @staticmethod
    def _calc_rf(layers):
        rf = 0
        ts = 1
        for i, (k, s) in enumerate(layers):
            if i == 0:
                rf = k
            else:
                rf += (k - 1) * ts
            ts *= s
        return rf

    def _find_archi(self, target_patch_size, max_layers=9):
        s = {}
        for layers_count in range(1, max_layers + 1):
            val = 1 << (layers_count - 1)
            while True:
                val -= 1
                layers = [[3, 2]]
                sum_st = 2
                for i in range(layers_count - 1):
                    st = 1 + (1 if val & (1 << i) != 0 else 0)
                    layers.append([3, st])
                    sum_st += st
                rf = self._calc_rf(layers)
                s_rf = s.get(rf, None)
                if s_rf is None:
                    s[rf] = (layers_count, sum_st, layers)
                else:
                    if layers_count < s_rf[0] or (layers_count == s_rf[0] and sum_st > s_rf[1]):
                        s[rf] = (layers_count, sum_st, layers)
                if val == 0:
                    break
        x = sorted(list(s.keys()))
        q = x[np.abs(np.array(x) - target_patch_size).argmin()]
        return s[q][2]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.leaky_relu(self.in_conv(x), 0.2)

        encs = []
        for conv in self.convs:
            encs.insert(0, x)
            x = F.leaky_relu(conv(x), 0.2)

        center_out = self.center_out(x)
        x = F.leaky_relu(self.center_conv(x), 0.2)

        for upconv, enc in zip(self.upconvs, encs):
            x = F.leaky_relu(upconv(x), 0.2)
            x = torch.cat([enc, x], dim=1)

        x = self.out_conv(x)
        return center_out, x
