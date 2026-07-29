import io
import itertools
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("saehd_model")


# ---------------------------------------------------------------------------
# Convolution Aware (CA) weight initialization (DFL feature)
# ---------------------------------------------------------------------------

def _ca_init_weights(module: nn.Module) -> None:
    """Initialize Conv2d weights using Convolution Aware initialization.
    
    Based on 'Convolutional Filters That Are Trained in the Frequency Domain'
    Uses Gabor-like filters for initial weights instead of random init.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kernel_size = m.kernel_size[0]
            out_ch, in_ch = m.out_channels, m.in_channels
            # Gabor-like initialization
            weights = np.zeros((out_ch, in_ch, kernel_size, kernel_size), dtype=np.float32)
            for o in range(out_ch):
                for i in range(in_ch):
                    # Random frequency and orientation
                    freq = np.random.uniform(0.1, 0.5)
                    theta = np.random.uniform(0, np.pi)
                    sigma = np.random.uniform(1.0, kernel_size / 3.0)
                    for y in range(kernel_size):
                        for x in range(kernel_size):
                            x_rot = (x - kernel_size // 2) * np.cos(theta) + (y - kernel_size // 2) * np.sin(theta)
                            y_rot = -(x - kernel_size // 2) * np.sin(theta) + (y - kernel_size // 2) * np.cos(theta)
                            gabor = np.exp(-0.5 * (x_rot**2 + y_rot**2) / sigma**2) * np.cos(2 * np.pi * freq * x_rot)
                            weights[o, i, y, x] = gabor
            # Normalize
            std = weights.std()
            if std > 0:
                weights = weights / std * 0.1
            m.weight.data = torch.from_numpy(weights).to(m.weight.device)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Building blocks (exact translation from TF original)
# ---------------------------------------------------------------------------

class Downscale(nn.Module):
    """Conv2D(kernel=5, stride=2) + LeakyReLU(0.1)"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DownscaleBlock(nn.Module):
    """4 downscale steps. Channel progression: ch*min(2^i, 8) for i=0..3"""

    def __init__(self, in_ch: int, ch: int, n_downscales: int = 4, kernel_size: int = 5):
        super().__init__()
        self.downs = nn.ModuleList()
        last_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * min(2 ** i, 8)
            self.downs.append(Downscale(last_ch, cur_ch, kernel_size=kernel_size))
            last_ch = cur_ch
        self.out_ch = last_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for down in self.downs:
            x = down(x)
        return x


class Upscale(nn.Module):
    """Conv2D(in, out*4, k=3) + LeakyReLU(0.1) + PixelShuffle(2) = 2x upsample"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size, padding=kernel_size // 2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(x))
        B, C, H, W = x.shape
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, H * 2, W * 2)
        return x


class ResidualBlock(nn.Module):
    """Conv → LeakyReLU(0.2) → Conv → add input → LeakyReLU(0.2)"""

    def __init__(self, ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv1(x)
        res = F.leaky_relu(res, 0.2)
        res = self.conv2(res)
        out = x + res
        out = F.leaky_relu(out, 0.2)
        return out


# ---------------------------------------------------------------------------
# UNetPatchDiscriminator (DFL GAN)
# ---------------------------------------------------------------------------

def _find_unet_disc_archi(target_patch_size: int, max_layers: int = 10) -> list[int]:
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
        strides = _find_unet_disc_archi(patch_size)
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
            ch_in = level_chs[i + 1] * 2  # concat with skip
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


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class SAEHDEncoder(nn.Module):
    """
    DownscaleBlock(4 steps) → flatten → optional pixel_norm

    For res=128, e_dims=64:
      3→64@128 → 64@64 → 128@32 → 256@16 → 512@8
      flatten → 32768
      pixel_norm (if 'u')
    """

    def __init__(self, in_ch: int = 3, e_dims: int = 64, n_downscales: int = 4, use_pixel_norm: bool = False):
        super().__init__()
        self.down_block = DownscaleBlock(in_ch, e_dims, n_downscales=n_downscales)
        self.out_ch = e_dims * min(2 ** (n_downscales - 1), 8)  # e_dims * 8 for 4 steps
        self.use_pixel_norm = use_pixel_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_block(x)
        x = x.flatten(1)  # (B, C*H*W)
        if self.use_pixel_norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x

    def get_out_res(self, resolution: int) -> int:
        return resolution // (2 ** 4)


# ---------------------------------------------------------------------------
# Inter (bottleneck)
# ---------------------------------------------------------------------------

class SAEHDInter(nn.Module):
    """
    Dense bottleneck: flatten → Dense(in, ae_ch) → Dense(ae_ch, res²*ae_out_ch)
    → reshape → Upscale(2x)

    For df: ae_out_ch = ae_dims
    For liae: ae_out_ch = ae_dims * 2
    """

    def __init__(self, in_features: int, ae_ch: int, ae_out_ch: int, bottleneck_res: int):
        super().__init__()
        self.ae_out_ch = ae_out_ch
        self.bottleneck_res = bottleneck_res
        self.dense1 = nn.Linear(in_features, ae_ch)
        self.dense2 = nn.Linear(ae_ch, bottleneck_res * bottleneck_res * ae_out_ch)
        self.upscale = Upscale(ae_out_ch, ae_out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.dense2(x)
        B, _ = x.shape
        x = x.reshape(B, self.ae_out_ch, self.bottleneck_res, self.bottleneck_res)
        x = self.upscale(x)  # 2x resolution
        return x

    def get_out_ch(self) -> int:
        return self.ae_out_ch

    def get_out_res(self) -> int:
        return self.bottleneck_res * 2


# ---------------------------------------------------------------------------
# Decoder (outputs face RGB + mask)
# ---------------------------------------------------------------------------

class SAEHDDecoder(nn.Module):
    """
    3 upscale stages with residual blocks + separate mask branch.
    
    With use_depth_to_space=True ('d' option):
      Final output uses 4 parallel convs + depth_to_space for 2x resolution boost
      instead of a standard Upscale stage. This is more compute-efficient.
    """

    def __init__(self, in_ch: int, d_dims: int = 64, d_mask_dims: int = 21,
                 use_depth_to_space: bool = False):
        super().__init__()
        self.use_depth_to_space = use_depth_to_space

        # Face branch
        self.upscale0 = Upscale(in_ch, d_dims * 8)
        self.res0 = ResidualBlock(d_dims * 8)
        self.upscale1 = Upscale(d_dims * 8, d_dims * 4)
        self.res1 = ResidualBlock(d_dims * 4)
        self.upscale2 = Upscale(d_dims * 4, d_dims * 2)
        self.res2 = ResidualBlock(d_dims * 2)

        if use_depth_to_space:
            self.out_conv = nn.Conv2d(d_dims * 2, 3, kernel_size=1, padding=0)
            self.out_conv1 = nn.Conv2d(d_dims * 2, 3, kernel_size=3, padding=1)
            self.out_conv2 = nn.Conv2d(d_dims * 2, 3, kernel_size=3, padding=1)
            self.out_conv3 = nn.Conv2d(d_dims * 2, 3, kernel_size=3, padding=1)
        else:
            self.out_conv = nn.Conv2d(d_dims * 2, 3, kernel_size=1, padding=0)

        # Mask branch
        self.upscalem0 = Upscale(in_ch, d_mask_dims * 8)
        self.upscalem1 = Upscale(d_mask_dims * 8, d_mask_dims * 4)
        self.upscalem2 = Upscale(d_mask_dims * 4, d_mask_dims * 2)
        if use_depth_to_space:
            self.upscalem3 = Upscale(d_mask_dims * 2, d_mask_dims * 1)
            self.out_convm = nn.Conv2d(d_mask_dims * 1, 1, kernel_size=1, padding=0)
        else:
            self.out_convm = nn.Conv2d(d_mask_dims * 2, 1, kernel_size=1, padding=0)

    def _depth_to_space(self, x: torch.Tensor, block_size: int = 2) -> torch.Tensor:
        """PixelShuffle: rearrange channels into spatial dimensions."""
        B, C, H, W = x.shape
        oc = C // (block_size * block_size)
        x = x.reshape(B, block_size, block_size, oc, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2)
        x = x.reshape(B, oc, H * block_size, W * block_size)
        return x

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Face
        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)

        if self.use_depth_to_space:
            c0 = self.out_conv(x)
            c1 = self.out_conv1(x)
            c2 = self.out_conv2(x)
            c3 = self.out_conv3(x)
            face = torch.sigmoid(self._depth_to_space(torch.cat([c0, c1, c2, c3], dim=1), 2))
        else:
            face = torch.sigmoid(self.out_conv(x))

        # Mask
        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        if self.use_depth_to_space:
            m = self.upscalem3(m)
        mask = torch.sigmoid(self.out_convm(m))

        return face, mask


# ---------------------------------------------------------------------------
# Full SAEHD Model
# ---------------------------------------------------------------------------

class SAEHDModel(nn.Module):
    """
    Full SAEHD model supporting 'df' and 'liae' architectures.

    df architecture:
        encoder → inter → decoder_src (for src reconstruction)
        encoder → inter → decoder_dst (for dst reconstruction)
        SWAP: encoder(dst) → inter → decoder_src → src face on dst structure

    liae architecture:
        encoder → inter_AB, inter_B → decoder (shared)
        src code = concat(inter_AB(enc(src)), inter_AB(enc(src)))
        dst code = concat(inter_B(enc(dst)), inter_AB(enc(dst)))
        SWAP code = concat(inter_AB(enc(dst)), inter_AB(enc(dst)))
    """

    def __init__(
        self,
        resolution: int = 128,
        architecture: str = "df",
        ae_dims: int = 256,
        e_dims: int = 64,
        d_dims: int = 64,
        d_mask_dims: int = None,
        gan_dims: int = 16,
        gan_patch_size: int = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.architecture = architecture
        self.ae_dims = ae_dims
        self.e_dims = e_dims
        self.d_dims = d_dims
        self.d_mask_dims = d_mask_dims if d_mask_dims is not None else (d_dims // 3 + d_dims // 3 % 2)
        archi_parts = architecture.split("-")
        self.archi_type = archi_parts[0]  # df or liae
        self.archi_opts = archi_parts[1] if len(archi_parts) > 1 else ""  # u/d/t/c
        self.use_liae = (self.archi_type == "liae")
        self.use_depth_to_space = "d" in self.archi_opts
        self.use_pixel_norm = "u" in self.archi_opts
        self.use_true_face = "t" in self.archi_opts  # -t: make face more like src (DFL 2021-10)

        # Calculate downscale steps: res → res/2 → ... → res/16
        n_downscales = 0
        r = resolution
        while r > 4:
            r //= 2
            n_downscales += 1
        # n_downscales = 5 for res=128 (128→64→32→16→8→4), but encoder uses 4
        # The encoder does 4 downscale steps: res → res/16
        enc_downscales = 4
        bottleneck_res = resolution // (2 ** enc_downscales)  # 8 for res=128

        # 'd' option: bottleneck is half the resolution (extra depth_to_space at output)
        if self.use_depth_to_space:
            bottleneck_res = bottleneck_res // 2

        # Encoder
        self.encoder = SAEHDEncoder(
            in_ch=3, e_dims=e_dims, n_downscales=enc_downscales,
            use_pixel_norm=self.use_pixel_norm or self.use_liae,
        )
        enc_spatial_res = resolution // (2 ** enc_downscales)
        enc_out_features = self.encoder.out_ch * (enc_spatial_res ** 2)

        if not self.use_liae:
            # df architecture: shared inter + separate decoders
            self.inter = SAEHDInter(
                in_features=enc_out_features,
                ae_ch=ae_dims,
                ae_out_ch=ae_dims,
                bottleneck_res=bottleneck_res,
            )
            inter_out_ch = ae_dims
            self.decoder_src = SAEHDDecoder(inter_out_ch, d_dims, self.d_mask_dims,
                                            use_depth_to_space=self.use_depth_to_space)
            self.decoder_dst = SAEHDDecoder(inter_out_ch, d_dims, self.d_mask_dims,
                                            use_depth_to_space=self.use_depth_to_space)
        else:
            # liae architecture: inter_AB + inter_B + shared decoder
            self.inter_AB = SAEHDInter(
                in_features=enc_out_features,
                ae_ch=ae_dims,
                ae_out_ch=ae_dims * 2,
                bottleneck_res=bottleneck_res,
            )
            self.inter_B = SAEHDInter(
                in_features=enc_out_features,
                ae_ch=ae_dims,
                ae_out_ch=ae_dims * 2,
                bottleneck_res=bottleneck_res,
            )
            # Shared decoder takes concat(inter_X, inter_Y) = 2x channels
            shared_dec_in_ch = ae_dims * 2 * 2  # concat of two inter outputs
            self.decoder = SAEHDDecoder(shared_dec_in_ch, d_dims, self.d_mask_dims,
                                        use_depth_to_space=self.use_depth_to_space)

        # -t variant: learnable src identity injection (DFL 2021-10)
        if self.use_true_face:
            if not self.use_liae:
                # df: inject into inter output before decoder_src
                bottleneck_ch = ae_dims
                self.src_identity = nn.Parameter(torch.zeros(1, bottleneck_ch,
                    bottleneck_res * 2 if self.use_depth_to_space else bottleneck_res,
                    bottleneck_res * 2 if self.use_depth_to_space else bottleneck_res))
            else:
                # liae: inject into shared decoder input
                self.src_identity = nn.Parameter(torch.zeros(1, ae_dims * 2 * 2,
                    bottleneck_res * 2 if self.use_depth_to_space else bottleneck_res,
                    bottleneck_res * 2 if self.use_depth_to_space else bottleneck_res))

        self.gan_dims = gan_dims
        self.gan_patch_size = gan_patch_size if gan_patch_size is not None else max(8, resolution // 8)
        self.discriminator = None
        self.code_discriminator = None

        # DFL uses Glorot (Xavier) uniform for bottleneck Dense layers only.
        # Conv layers before LeakyReLU use PyTorch default (Kaiming uniform)
        # which is the correct init for ReLU-family activations.
        self._init_bottleneck_weights()

    # ---- Forward methods for training ----

    def _init_bottleneck_weights(self):
        """Initialize bottleneck Dense layers with Xavier uniform (DFL behavior).
        
        Only the SAEHDInter dense layers use Glorot/Xavier uniform, matching
        how DFL's tf.keras.layers.Dense defaults work. All Conv2d layers keep
        PyTorch's Kaiming uniform default (correct for LeakyReLU).
        """
        inter_modules = []
        if hasattr(self, 'inter'):
            inter_modules.append(self.inter)
        if hasattr(self, 'inter_AB'):
            inter_modules.append(self.inter_AB)
        if hasattr(self, 'inter_B'):
            inter_modules.append(self.inter_B)
        for inter in inter_modules:
            for m in inter.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode image to bottleneck code."""
        return self.encoder(img)

    def build_discriminator(self) -> UNetPatchDiscriminator:
        """Build the UNetPatchDiscriminator for GAN training."""
        self.discriminator = UNetPatchDiscriminator(
            in_ch=3, base_ch=self.gan_dims, patch_size=self.gan_patch_size,
        )
        return self.discriminator

    def discriminate(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run discriminator, returns (center_out, unet_out)."""
        if self.discriminator is None:
            raise RuntimeError("Discriminator not built. Call build_discriminator() first.")
        return self.discriminator(img)

    def build_code_discriminator(self, code_res: int) -> CodeDiscriminator:
        """Build CodeDiscriminator for true_face_power (df architecture only)."""
        if self.use_liae:
            raise RuntimeError("CodeDiscriminator only supports df architecture")
        self.code_discriminator = CodeDiscriminator(
            in_ch=self.ae_dims, code_res=code_res, base_ch=256,
        )
        return self.code_discriminator

    def decode_src(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode through src path (df) or src code path (liae)."""
        if not self.use_liae:
            inter_out = self.inter(code)
            return self.decoder_src(inter_out)
        else:
            raise RuntimeError("Use decode_liae() for liae architecture")

    def decode_dst(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode through dst path (df) or dst code path (liae)."""
        if not self.use_liae:
            inter_out = self.inter(code)
            return self.decoder_dst(inter_out)
        else:
            raise RuntimeError("Use decode_liae() for liae architecture")

    def initialize_ca_weights(self) -> None:
        """Apply Convolution Aware weight initialization (DFL feature)."""
        _ca_init_weights(self)
        _logger.info("CA weights initialized")

    def forward_df(self, src_img: torch.Tensor, dst_img: torch.Tensor) -> dict:
        """
        Full forward pass for df architecture.
        Returns dict with all reconstructions and swap outputs.
        """
        src_code = self.encoder(src_img)
        dst_code = self.encoder(dst_img)

        src_inter = self.inter(src_code)
        dst_inter = self.inter(dst_code)

        # -t variant: inject learnable src identity into src decoder path
        if self.use_true_face:
            src_inter_for_src = src_inter + self.src_identity
        else:
            src_inter_for_src = src_inter

        # Self-reconstruction
        pred_src_src, pred_src_srcm = self.decoder_src(src_inter_for_src)
        pred_dst_dst, pred_dst_dstm = self.decoder_dst(dst_inter)

        # Swap: decoder_src with dst's code (also with identity for -t)
        if self.use_true_face:
            dst_inter_for_src = dst_inter + self.src_identity
        else:
            dst_inter_for_src = dst_inter
        pred_src_dst, pred_src_dstm = self.decoder_src(dst_inter_for_src)

        # Swap with stop_gradient on code (for style loss)
        dst_inter_detached = dst_inter.detach()
        if self.use_true_face:
            dst_inter_detached = dst_inter_detached + self.src_identity
        pred_src_dst_no_grad, _ = self.decoder_src(dst_inter_detached)

        return {
            "pred_src_src": pred_src_src,
            "pred_src_srcm": pred_src_srcm,
            "pred_dst_dst": pred_dst_dst,
            "pred_dst_dstm": pred_dst_dstm,
            "pred_src_dst": pred_src_dst,
            "pred_src_dstm": pred_src_dstm,
            "pred_src_dst_no_grad": pred_src_dst_no_grad,
        }

    def forward_liae(self, src_img: torch.Tensor, dst_img: torch.Tensor) -> dict:
        """
        Full forward pass for liae architecture.
        """
        src_enc = self.encoder(src_img)
        dst_enc = self.encoder(dst_img)

        src_inter_AB = self.inter_AB(src_enc)
        dst_inter_AB = self.inter_AB(dst_enc)
        dst_inter_B = self.inter_B(dst_enc)

        # src code = concat(inter_AB(src), inter_AB(src)) — no identity info
        src_code = torch.cat([src_inter_AB, src_inter_AB], dim=1)
        # dst code = concat(inter_B(dst), inter_AB(dst)) — has dst identity
        dst_code = torch.cat([dst_inter_B, dst_inter_AB], dim=1)
        # swap code = concat(inter_AB(dst), inter_AB(dst)) — no dst identity
        swap_code = torch.cat([dst_inter_AB, dst_inter_AB], dim=1)

        # -t variant: inject learnable src identity into src decoder path
        if self.use_true_face:
            src_code = src_code + self.src_identity
            swap_code = swap_code + self.src_identity

        # Self-reconstruction
        pred_src_src, pred_src_srcm = self.decoder(src_code)
        pred_dst_dst, pred_dst_dstm = self.decoder(dst_code)

        # Swap
        pred_src_dst, pred_src_dstm = self.decoder(swap_code)

        # Swap with stop_gradient
        swap_code_detached = swap_code.detach()
        pred_src_dst_no_grad, _ = self.decoder(swap_code_detached)

        return {
            "pred_src_src": pred_src_src,
            "pred_src_srcm": pred_src_srcm,
            "pred_dst_dst": pred_dst_dst,
            "pred_dst_dstm": pred_dst_dstm,
            "pred_src_dst": pred_src_dst,
            "pred_src_dstm": pred_src_dstm,
            "pred_src_dst_no_grad": pred_src_dst_no_grad,
        }

    def forward(self, src_img: torch.Tensor, dst_img: torch.Tensor) -> dict:
        if self.use_liae:
            return self.forward_liae(src_img, dst_img)
        return self.forward_df(src_img, dst_img)

    # ---- Inference (merge) ----

    def merge(self, dst_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference: given dst face, produce swapped face + masks.
        Returns: (swapped_face, dst_mask, swap_mask)
        """
        self.eval()
        with torch.no_grad():
            if not self.use_liae:
                dst_code = self.encoder(dst_img)
                dst_inter = self.inter(dst_code)
                # -t variant: inject src_identity for swap
                if self.use_true_face:
                    swap_inter = dst_inter + self.src_identity
                else:
                    swap_inter = dst_inter
                swapped, swap_mask = self.decoder_src(swap_inter)
                _, dst_mask = self.decoder_dst(dst_inter)
            else:
                dst_enc = self.encoder(dst_img)
                dst_inter_AB = self.inter_AB(dst_enc)
                dst_inter_B = self.inter_B(dst_enc)
                swap_code = torch.cat([dst_inter_AB, dst_inter_AB], dim=1)
                # -t variant: inject src_identity for swap
                if self.use_true_face:
                    swap_code = swap_code + self.src_identity
                dst_code = torch.cat([dst_inter_B, dst_inter_AB], dim=1)
                swapped, swap_mask = self.decoder(swap_code)
                _, dst_mask = self.decoder(dst_code)
        return swapped, dst_mask, swap_mask

    # ---- Save / Load ----

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "resolution": self.resolution,
            "architecture": self.architecture,
            "ae_dims": self.ae_dims,
            "e_dims": self.e_dims,
            "d_dims": self.d_dims,
            "d_mask_dims": self.d_mask_dims,
        }
        FileManager.atomic_write(
            model_dir / "SAEHD_config.json",
            json.dumps(config, indent=2),
        )

        components = {"encoder": self.encoder}
        if not self.use_liae:
            components["inter"] = self.inter
            components["decoder_src"] = self.decoder_src
            components["decoder_dst"] = self.decoder_dst
        else:
            components["inter_AB"] = self.inter_AB
            components["inter_B"] = self.inter_B
            components["decoder"] = self.decoder

        for name, module in components.items():
            buf = io.BytesIO()
            torch.save(module.state_dict(), buf)
            FileManager.atomic_write(model_dir / f"SAEHD_{name}.pt", buf.getvalue())

        # Save src_identity for -t variant
        if self.use_true_face and hasattr(self, 'src_identity'):
            buf = io.BytesIO()
            torch.save(self.src_identity.data, buf)
            FileManager.atomic_write(model_dir / "SAEHD_src_identity.pt", buf.getvalue())

        if self.discriminator is not None:
            buf = io.BytesIO()
            torch.save(self.discriminator.state_dict(), buf)
            FileManager.atomic_write(model_dir / "SAEHD_discriminator.pt", buf.getvalue())

        if self.code_discriminator is not None:
            buf = io.BytesIO()
            torch.save(self.code_discriminator.state_dict(), buf)
            FileManager.atomic_write(model_dir / "SAEHD_code_discriminator.pt", buf.getvalue())

        _logger.info(f"SAEHD model saved to {model_dir}")

    def load(self, model_dir: Path, device: torch.device = None) -> bool:
        model_dir = Path(model_dir)
        map_loc = device if device else "cpu"

        config_path = model_dir / "SAEHD_config.json"
        if not config_path.exists():
            return False

        components = {"encoder": self.encoder}
        if not self.use_liae:
            components["inter"] = self.inter
            components["decoder_src"] = self.decoder_src
            components["decoder_dst"] = self.decoder_dst
        else:
            components["inter_AB"] = self.inter_AB
            components["inter_B"] = self.inter_B
            components["decoder"] = self.decoder

        for name, module in components.items():
            path = model_dir / f"SAEHD_{name}.pt"
            if path.exists():
                data = open(str(path), "rb").read()
                module.load_state_dict(torch.load(io.BytesIO(data), map_location=map_loc, weights_only=True))

        disc_path = model_dir / "SAEHD_discriminator.pt"
        if disc_path.exists():
            if self.discriminator is None:
                self.build_discriminator()
            data = open(str(disc_path), "rb").read()
            self.discriminator.load_state_dict(torch.load(io.BytesIO(data), map_location=map_loc, weights_only=True))

        code_disc_path = model_dir / "SAEHD_code_discriminator.pt"
        if code_disc_path.exists() and not self.use_liae:
            code_res = self.inter.get_out_res() if not self.use_liae else 0
            if code_res > 0:
                if self.code_discriminator is None:
                    self.build_code_discriminator(code_res)
                data = open(str(code_disc_path), "rb").read()
                self.code_discriminator.load_state_dict(torch.load(io.BytesIO(data), map_location=map_loc, weights_only=True))

        _logger.info(f"SAEHD model loaded from {model_dir}")

        # Load src_identity for -t variant
        identity_path = model_dir / "SAEHD_src_identity.pt"
        if identity_path.exists() and self.use_true_face and hasattr(self, 'src_identity'):
            data = open(str(identity_path), "rb").read()
            self.src_identity.data = torch.load(io.BytesIO(data), map_location=map_loc, weights_only=True)
            _logger.info("src_identity loaded")

        return True

    @classmethod
    def from_config(cls, config: dict) -> "SAEHDModel":
        return cls(
            resolution=config.get("resolution", 128),
            architecture=config.get("architecture", "df"),
            ae_dims=config.get("ae_dims", 512),
            e_dims=config.get("e_dims", 64),
            d_dims=config.get("d_dims", 64),
            d_mask_dims=config.get("d_mask_dims", None),
            gan_dims=config.get("gan_dims", 16),
            gan_patch_size=config.get("gan_patch_size", None),
        )

    def get_param_groups(self) -> list:
        """Return parameter groups for optimizer (separate src/dst for liae)."""
        if not self.use_liae:
            return list(self.parameters())
        return list(self.parameters())
