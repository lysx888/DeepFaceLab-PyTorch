"""DFL 判别器的 PyTorch 原生实现。

CodeDiscriminator: true_face_power 用，判别编码向量。
UNetPatchDiscriminator: GAN 用，U-Net 结构判别图像 patch。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeDiscriminator(nn.Module):
    """对齐 DFL leras/models/CodeDiscriminator.py。"""

    def __init__(self, in_ch, code_res, ch=256):
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
        self.out_conv = nn.Conv2d(prev_ch, 1, kernel_size=1)

    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
        return self.out_conv(x)


def _calc_receptive_field_size(layers):
    """计算感受野大小，对齐 DFL UNetPatchDiscriminator.calc_receptive_field_size。"""
    rf = 0
    ts = 1
    for i, (k, s) in enumerate(layers):
        if i == 0:
            rf = k
        else:
            rf += (k - 1) * ts
        ts *= s
    return rf


def _find_archi(target_patch_size, max_layers=9):
    """寻找最佳层配置，对齐 DFL UNetPatchDiscriminator.find_archi。

    只用 3x3 conv，第一层 stride=2，后续层 stride=1 或 2。
    """
    s = {}
    for layers_count in range(1, max_layers + 1):
        val = 1 << (layers_count - 1)
        while True:
            val -= 1

            layers = []
            sum_st = 0
            layers.append([3, 2])
            sum_st += 2
            for i in range(layers_count - 1):
                st = 1 + (1 if val & (1 << i) != 0 else 0)
                layers.append([3, st])
                sum_st += st

            rf = _calc_receptive_field_size(layers)

            s_rf = s.get(rf, None)
            if s_rf is None:
                s[rf] = (layers_count, sum_st, layers)
            else:
                if layers_count < s_rf[0] or \
                   (layers_count == s_rf[0] and sum_st > s_rf[1]):
                    s[rf] = (layers_count, sum_st, layers)

            if val == 0:
                break

    x = sorted(list(s.keys()))
    q = x[np.abs(np.array(x) - target_patch_size).argmin()]
    return s[q][2]


def _conv_transpose_same(in_ch, out_ch, kernel_size, stride):
    """TF SAME 转置卷积，输出=输入×stride。

    DFL Conv2DTranspose SAME: output = input * stride。
    PyTorch: output = (input-1)*stride - 2*padding + kernel_size + output_padding。
    => output_padding = stride + 2*padding - kernel_size。
    """
    padding = (kernel_size - 1) // 2
    output_padding = stride + 2 * padding - kernel_size
    if output_padding < 0:
        output_padding = 0
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              output_padding=output_padding)


class UNetPatchDiscriminator(nn.Module):
    """对齐 DFL leras/models/PatchDiscriminator.py 的 UNetPatchDiscriminator。

    用 find_archi 动态生成层配置（只用 3x3 conv）。
    forward 返回 (center_out, x)，center_out 是中心层输出，x 是完整 U-Net 输出。
    """

    def __init__(self, patch_size, in_ch, base_ch=16):
        super().__init__()
        layers = _find_archi(patch_size)
        n = len(layers)
        level_chs = {i - 1: min(base_ch * (2 ** i), 512) for i in range(n + 1)}

        self.in_conv = nn.Conv2d(in_ch, level_chs[-1], kernel_size=1)

        self.convs = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i, (ks, s) in enumerate(layers):
            self.convs.append(nn.Conv2d(level_chs[i - 1], level_chs[i],
                                        kernel_size=ks, stride=s,
                                        padding=(ks - 1) // 2))
            mult = 2 if i != n - 1 else 1
            self.upconvs.insert(0, _conv_transpose_same(level_chs[i] * mult, level_chs[i - 1], ks, s))

        self.out_conv = nn.Conv2d(level_chs[-1] * 2, 1, kernel_size=1)
        self.center_out = nn.Conv2d(level_chs[n - 1], 1, kernel_size=1)
        self.center_conv = nn.Conv2d(level_chs[n - 1], level_chs[n - 1], kernel_size=1)

    def forward(self, x):
        x = F.leaky_relu(self.in_conv(x), 0.2)

        encs = []
        for conv in self.convs:
            encs.insert(0, x)
            x = F.leaky_relu(conv(x), 0.2)

        center_out = self.center_out(x)
        x = F.leaky_relu(self.center_conv(x), 0.2)

        for upconv, enc in zip(self.upconvs, encs):
            x = F.leaky_relu(upconv(x), 0.2)
            if x.shape[2:] != enc.shape[2:]:
                x = x[:, :, :enc.shape[2], :enc.shape[3]]
            x = torch.cat([enc, x], dim=1)

        x = self.out_conv(x)
        return center_out, x
