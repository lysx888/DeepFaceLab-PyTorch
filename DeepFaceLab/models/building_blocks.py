import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class XCosX(nn.Module):
    """x * cos(x) activation function (DFL 'c' architecture option).

    Smooth, bounded, oscillating — provides implicit regularization.
    Derivative: cos(x) - x*sin(x), well-defined everywhere.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.cos(x)


def _get_activation(use_c: bool = False) -> nn.Module:
    if use_c:
        return XCosX()
    return nn.LeakyReLU(0.1, inplace=True)


class VGGPerceptualLoss(nn.Module):
    """VGG16-based perceptual loss (LPIPS-style).

    Extracts features from conv1_2, conv2_2, conv3_3, conv4_3 layers
    and computes L1 distance in feature space.

    Input: BGR images in [0, 1] range (will be converted to RGB internally).
    """

    _VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
    _VGG_STD = torch.tensor([0.229, 0.224, 0.225])

    def __init__(self, device: torch.device = None):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except (ImportError, OSError):
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)

        self._feature_layers = [3, 8, 15, 22]
        self._vgg_blocks = nn.ModuleList()
        prev_idx = 0
        for layer_idx in self._feature_layers:
            block = nn.Sequential(*list(vgg.features.children())[prev_idx:layer_idx + 1])
            self._vgg_blocks.append(block)
            prev_idx = layer_idx + 1

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        if device is not None:
            self.to(device)
        self._mean: torch.Tensor = None
        self._std: torch.Tensor = None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # BGR -> RGB
        x = x[:, [2, 1, 0], :, :]
        if self._mean is None or self._mean.device != x.device:
            self._mean = self._VGG_MEAN.to(x.device).view(1, 3, 1, 1)
            self._std = self._VGG_STD.to(x.device).view(1, 3, 1, 1)
        return (x - self._mean) / self._std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)
        loss = torch.tensor(0.0, device=pred.device)
        x_pred = pred_n
        x_target = target_n
        for block in self._vgg_blocks:
            x_pred = block(x_pred)
            x_target = block(x_target)
            loss = loss + F.l1_loss(x_pred, x_target)
        return loss


def ca_init_weights(module: nn.Module) -> None:
    """Initialize Conv2d weights using Convolution Aware initialization.

    Based on 'Convolutional Filters That Are Trained in the Frequency Domain'
    Uses Gabor-like filters for initial weights instead of random init.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kernel_size = m.kernel_size[0]
            out_ch, in_ch = m.out_channels, m.in_channels
            weights = np.zeros((out_ch, in_ch, kernel_size, kernel_size), dtype=np.float32)
            for o in range(out_ch):
                for i in range(in_ch):
                    freq = np.random.uniform(0.1, 0.5)
                    theta = np.random.uniform(0, np.pi)
                    sigma = np.random.uniform(1.0, kernel_size / 3.0)
                    for y in range(kernel_size):
                        for x in range(kernel_size):
                            x_rot = (x - kernel_size // 2) * np.cos(theta) + (y - kernel_size // 2) * np.sin(theta)
                            y_rot = -(x - kernel_size // 2) * np.sin(theta) + (y - kernel_size // 2) * np.cos(theta)
                            gabor = np.exp(-0.5 * (x_rot**2 + y_rot**2) / sigma**2) * np.cos(2 * np.pi * freq * x_rot)
                            weights[o, i, y, x] = gabor
            std = weights.std()
            if std > 0:
                weights = weights / std * 0.1
            m.weight.data = torch.from_numpy(weights).to(m.weight.device)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Downscale(nn.Module):
    """Conv2D(kernel=5, stride=2) + activation"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, use_c: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2)
        self.act = _get_activation(use_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DownscaleBlock(nn.Module):
    """N downscale steps. Channel progression: ch*min(2^i, 8) for i=0..N-1"""

    def __init__(self, in_ch: int, ch: int, n_downscales: int = 4, kernel_size: int = 5, use_c: bool = False):
        super().__init__()
        self.downs = nn.ModuleList()
        last_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * min(2 ** i, 8)
            self.downs.append(Downscale(last_ch, cur_ch, kernel_size=kernel_size, use_c=use_c))
            last_ch = cur_ch
        self.out_ch = last_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for down in self.downs:
            x = down(x)
        return x


class Upscale(nn.Module):
    """Conv2D(in, out*4, k=3) + activation + PixelShuffle(2) = 2x upsample"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, use_c: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size, padding=kernel_size // 2)
        self.act = _get_activation(use_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(x))
        B, C, H, W = x.shape
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, H * 2, W * 2)
        return x


class ResidualBlock(nn.Module):
    """Conv -> act(0.2) -> Conv -> add input -> act(0.2)"""

    def __init__(self, ch: int, kernel_size: int = 3, use_c: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size // 2)
        self.use_c = use_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        if self.use_c:
            res = res * torch.cos(res)
        else:
            res = F.leaky_relu(res, 0.2)
        res = self.conv2(res)
        out = x + res
        if self.use_c:
            out = out * torch.cos(out)
        else:
            out = F.leaky_relu(out, 0.2)
        return out


def find_unet_disc_archi(target_patch_size: int, max_layers: int = 10) -> list[int]:
    """Find optimal stride configuration for UNetPatchDiscriminator.

    Returns list of strides (1 or 2) per layer that produces a receptive field
    closest to target_patch_size.
    """
    best_strides = [2]
    best_diff = abs(target_patch_size - 3)

    for n_layers in range(1, max_layers + 1):
        for strides in itertools.product([1, 2], repeat=n_layers):
            rf = 1
            for s in strides:
                rf = rf * s + (3 - s)
            diff = abs(target_patch_size - rf)
            if diff < best_diff:
                best_diff = diff
                best_strides = list(strides)
            if diff == 0:
                return best_strides
    return best_strides


class UNetPatchDiscriminator(nn.Module):
    """DFL's UNetPatchDiscriminator: U-Net discriminator with dual output.

    Inspired by "A U-Net Based Discriminator for Generative Adversarial Networks"
    (https://arxiv.org/abs/2002.12655)

    Returns (center_out, unet_out) for multi-scale adversarial loss.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 16, patch_size: int = 16):
        super().__init__()
        strides = find_unet_disc_archi(patch_size)
        n_layers = len(strides)

        level_chs = [min(base_ch * (2 ** i), 512) for i in range(n_layers + 1)]

        self.in_conv = nn.Conv2d(in_ch, level_chs[-1], kernel_size=1)
        self.encoder_convs = nn.ModuleList()
        for i in range(n_layers):
            self.encoder_convs.append(nn.Conv2d(
                level_chs[n_layers - i], level_chs[n_layers - i - 1],
                kernel_size=3, stride=strides[i], padding=1,
            ))

        self.center_conv = nn.Conv2d(level_chs[0], level_chs[0], kernel_size=3, padding=1)
        self.center_out_conv = nn.Conv2d(level_chs[0], 1, kernel_size=1)

        self.decoder_convs = nn.ModuleList()
        for i in range(n_layers):
            ch_in = level_chs[i] * 2
            ch_out = level_chs[i + 1]
            self.decoder_convs.append(nn.ConvTranspose2d(
                ch_in, ch_out, kernel_size=3, stride=strides[n_layers - 1 - i],
                padding=1, output_padding=strides[n_layers - 1 - i] - 1,
            ))

        self.out_conv = nn.Conv2d(level_chs[-1] * 2, 1, kernel_size=1)
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.leaky_relu(self.in_conv(x), 0.2)

        skips = [x]
        for conv in self.encoder_convs:
            x = F.leaky_relu(conv(x), 0.2)
            skips.append(x)

        center = F.leaky_relu(self.center_conv(x), 0.2)
        center_out = self.center_out_conv(center)

        for i, conv in enumerate(self.decoder_convs):
            skip = skips[self.n_layers - i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = F.leaky_relu(conv(torch.cat([x, skip], dim=1)), 0.2)

        skip0 = skips[0]
        if x.shape[2:] != skip0.shape[2:]:
            x = F.interpolate(x, size=skip0.shape[2:], mode="nearest")
        out = self.out_conv(torch.cat([x, skip0], dim=1))

        return center_out, out


class CodeDiscriminator(nn.Module):
    """DFL's CodeDiscriminator for true_face_power.

    Operates on encoder latent code (not pixels) to discriminate src vs dst codes.
    Only applicable to 'df' architecture.
    """

    def __init__(self, in_ch: int, code_res: int, base_ch: int = 256):
        super().__init__()
        n_downscales = 1 + code_res // 8
        self.convs = nn.ModuleList()
        prev_ch = in_ch
        for i in range(n_downscales):
            cur_ch = base_ch * min(2 ** i, 8)
            kernel_size = 4 if i == 0 else 3
            self.convs.append(nn.Conv2d(prev_ch, cur_ch, kernel_size, stride=2, padding=kernel_size // 2))
            prev_ch = cur_ch
        self.out_conv = nn.Conv2d(prev_ch, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
        return self.out_conv(x)
