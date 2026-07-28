import io
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("tfm_model")

_TFM_PRESETS = {
    "tiny": {"embed_dim": 24, "depths": [1, 1, 1, 1], "num_heads": [2, 4, 4, 8], "base_channels": 128, "w_dim": 256},
    "small": {"embed_dim": 48, "depths": [1, 1, 2, 1], "num_heads": [2, 4, 8, 16], "base_channels": 256, "w_dim": 256},
    "medium": {"embed_dim": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24], "base_channels": 512, "w_dim": 512},
    "large": {"embed_dim": 128, "depths": [2, 2, 6, 2], "num_heads": [4, 8, 16, 32], "base_channels": 512, "w_dim": 512},
}


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 128, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B * (H // window_size) * (W // window_size), window_size * window_size, C)
    return x


def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    nH = H // window_size
    nW = W // window_size
    B = windows.shape[0] // (nH * nW)
    x = windows.reshape(B, nH, nW, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, H, W)
    return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, shift_size: int = 0, drop_rate: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop_rate)
        self.proj_drop = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop_rate),
        )
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.register_buffer("_bias_index", relative_position_index.reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm1(x_flat)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x_nchw = x_flat.transpose(1, 2).reshape(B, C, H, W)
            x_nchw = F.pad(x_nchw, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x_nchw.shape
        else:
            Hp, Wp = H, W
            x_nchw = x_flat.transpose(1, 2).reshape(B, C, Hp, Wp)

        if self.shift_size > 0:
            shifted_x = torch.roll(x_nchw, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x_nchw

        x_windows = _window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)

        N = self.window_size * self.window_size
        qkv = self.qkv(x_windows).reshape(-1, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        rel_pos_bias = self.relative_position_bias_table[self._bias_index]
        rel_pos_bias = rel_pos_bias.reshape(N, N, -1).permute(2, 0, 1).unsqueeze(0)
        attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_out = (attn @ v).transpose(1, 2).reshape(-1, N, C)
        attn_out = self.proj_drop(self.proj(attn_out))

        attn_windows = attn_out.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = _window_reverse(attn_windows.reshape(-1, self.window_size * self.window_size, C), self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x_shifted = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x_shifted = shifted_x

        if pad_r > 0 or pad_b > 0:
            x_shifted = x_shifted[:, :, :H, :W].contiguous()

        x = shortcut + x_shifted
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, 4 * dim)
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class TFMEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Optional[list[int]] = None,
        num_heads: Optional[list[int]] = None,
        window_size: int = 8,
        drop_rate: float = 0.1,
        gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_stages = len(depths)
        self.gradient_checkpoint = gradient_checkpoint

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()

        dim = embed_dim
        for i in range(self.num_stages):
            stage_blocks = nn.ModuleList()
            for j in range(depths[i]):
                shift_size = 0 if j % 2 == 0 else window_size // 2
                stage_blocks.append(
                    SwinTransformerBlock(
                        dim=dim,
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=shift_size,
                        drop_rate=drop_rate,
                    )
                )
            self.stages.append(stage_blocks)
            if i < self.num_stages - 1:
                self.downsample.append(PatchMerging(dim))
            dim *= 2

    def _run_stage(self, stage_blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for block in stage_blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        features = {}
        dim = self.embed_dim
        for i in range(self.num_stages):
            if self.gradient_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._run_stage, self.stages[i], x, use_reentrant=False
                )
            else:
                x = self._run_stage(self.stages[i], x)
            features[f"stage{i + 1}"] = x
            if i < self.num_stages - 1:
                x = self.downsample[i](x)
                dim *= 2
        return features


class ConstInput(nn.Module):
    def __init__(self, channels: int, size: int = 4) -> None:
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, channels, size, size))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.const.expand(batch_size, -1, -1, -1)


class NoiseInjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            return x + self.weight * noise
        return x


class Upscale(nn.Module):
    """DFL-style upscale: Conv(k=3) → LeakyReLU(0.2) → PixelShuffle(2x)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size, padding=kernel_size // 2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(x))
        B, C, H, W = x.shape
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, H * 2, W * 2)
        return x


class ResBlock(nn.Module):
    """DFL-style residual block: Conv → LReLU(0.2) → Conv → add → LReLU(0.2)."""

    def __init__(self, ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.leaky_relu(self.conv1(x), 0.2)
        res = self.conv2(res)
        return F.leaky_relu(x + res, 0.2)


class SynthesisLayer(nn.Module):
    """Decoder layer: upsample → conv → style_mod → noise → activate.

    DFL df-style: NO SPADE, NO skip connections.
    Identity is in decoder WEIGHTS, not in any input signal.
    The only input is the Inter code (structure/pose, identity-free).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int = 512,
        is_up: bool = True,
    ) -> None:
        super().__init__()
        self.is_up = is_up
        if is_up:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.style_proj = nn.Linear(w_dim, 2 * out_channels)
        nn.init.zeros_(self.style_proj.weight)
        nn.init.zeros_(self.style_proj.bias)
        self.noise = NoiseInjection()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        w_style: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_up:
            x = self.upsample(x)
        x = self.conv(x)
        params = self.style_proj(w_style)
        gamma, beta = params.chunk(2, dim=1)
        x = (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) * x + beta.unsqueeze(-1).unsqueeze(-1)
        x = self.noise(x)
        x = self.activation(x)
        return x


class ToRGB(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TFMDecoder(nn.Module):
    """DFL-style CNN spatial decoder.

    Input: spatial feature map from Inter bottleneck (ae_dims @ res/16).
    Identity is in DECODER WEIGHTS — each decoder (src/dst) only sees
    one identity during training, so its weights encode that identity.

    Architecture: project → Upscale → 3×(Upscale + ResBlock) → Conv1x1 → Tanh
    Channel progression (ae_dims=256, d_dims=64, res=128):
      Input: 256@8×8 → proj→512@8×8 → Upscale→512@16×16 → Res
      → Upscale→256@32 → Res → Upscale→128@64 → Res
      → Upscale→64@128 → Res → Conv1x1→3@128 → Tanh
    """

    def __init__(
        self,
        resolution: int = 128,
        ae_dims: int = 256,
        d_dims: int = 64,
    ) -> None:
        super().__init__()
        self.resolution = resolution

        # Project ae_dims → d_dims*8 for decoder input
        self.proj = nn.Sequential(
            nn.Conv2d(ae_dims, d_dims * 8, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Initial upscale: res/16 → res/8 (e.g. 8→16 for 128px)
        self.upscale_init = Upscale(d_dims * 8, d_dims * 8)
        self.res_init = ResBlock(d_dims * 8)

        # Face branch: 3 upscale + resblock stages
        self.upscale0 = Upscale(d_dims * 8, d_dims * 8)
        self.res0 = ResBlock(d_dims * 8)
        self.upscale1 = Upscale(d_dims * 8, d_dims * 4)
        self.res1 = ResBlock(d_dims * 4)
        self.upscale2 = Upscale(d_dims * 4, d_dims * 2)
        self.res2 = ResBlock(d_dims * 2)
        self.out_conv = nn.Conv2d(d_dims * 2, 3, kernel_size=1, padding=0)

    def forward(self, spatial_feat: torch.Tensor) -> torch.Tensor:
        x = self.proj(spatial_feat)
        x = self.upscale_init(x)
        x = self.res_init(x)
        x = self.upscale0(x)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)
        x = self.out_conv(x)

        if x.shape[2] != self.resolution or x.shape[3] != self.resolution:
            x = F.interpolate(x, size=(self.resolution, self.resolution),
                              mode="bilinear", align_corners=False)
        return torch.tanh(x)



class WPlusMapper(nn.Module):
    def __init__(self, embed_dim_last: int = 768, num_layers: int = 7, w_dim: int = 512) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(embed_dim_last, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_layers * w_dim)
        self.num_layers = num_layers
        self.w_dim = w_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.reshape(x.shape[0], self.num_layers, self.w_dim)


class TFMDiscriminator(nn.Module):
    def __init__(self, in_chans: int = 3, base_ch: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        layers = []
        ch_in = in_chans
        for i in range(num_layers):
            ch_out = base_ch * (2 ** min(i, 3))
            stride = 2 if i < num_layers - 1 else 1
            layers.append(nn.Conv2d(ch_in, ch_out, 4, stride, 1))
            if i > 0:
                layers.append(nn.InstanceNorm2d(ch_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch_in = ch_out
        layers.append(nn.Conv2d(ch_in, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VGGPerceptualLoss(nn.Module):
    _VGG19_PATH = None

    def __init__(self, resize: bool = True) -> None:
        super().__init__()
        from torchvision import models
        if self._VGG19_PATH is None:
            from DeepFaceLab.setting import VGG19_MODEL_PATH
            VGGPerceptualLoss._VGG19_PATH = VGG19_MODEL_PATH
        if self._VGG19_PATH.exists():
            vgg_full = models.vgg19(weights=None)
            state = torch.load(str(self._VGG19_PATH), map_location="cpu", weights_only=True)
            vgg_full.load_state_dict(state)
            vgg = vgg_full.features
        else:
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        blocks = []
        layers = [0, 4, 9, 18, 27]
        prev = 0
        for i in layers:
            blocks.append(nn.Sequential(*list(vgg.children())[prev:i]))
            prev = i
        self.blocks = nn.ModuleList(blocks)
        for p in self.parameters():
            p.requires_grad = False
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize and (pred.shape[2] < 224 or pred.shape[3] < 224):
            pred = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
        loss = 0.0
        x_pred = pred
        x_target = target
        for block in self.blocks:
            x_pred = block(x_pred)
            x_target = block(x_target)
            loss = loss + F.l1_loss(x_pred, x_target)
        return loss


class TFMInter(nn.Module):
    """Information bottleneck between encoder and decoder (DFL-style).

    DFL's critical design: Dense(32768→256) compresses encoder output 128:1,
    forcing identity out of the latent code. Identity can only be encoded
    in decoder weights.

    Our version:
    1. Flatten encoder's deepest stage features
    2. Dense compress to ae_dims (bottleneck!)
    3. Dense expand to spatial feature map (bottleneck_res² × ae_dims)
    4. Upscale 2x → output spatial feature map for CNN decoder

    The compression ratio is the key: too small → lose structure info,
    too large → identity leaks through. DFL uses 128:1 for 128px.
    """

    def __init__(
        self,
        encoder_out_dim: int,
        ae_dims: int = 256,
        lowest_dense_res: int = 8,
    ) -> None:
        super().__init__()
        self.ae_dims = ae_dims
        self.lowest_dense_res = lowest_dense_res

        self.dense_compress = nn.Linear(encoder_out_dim, ae_dims)
        expand_dim = lowest_dense_res * lowest_dense_res * ae_dims
        self.dense_expand = nn.Linear(ae_dims, expand_dim)
        self.upscale = Upscale(ae_dims, ae_dims)

    def forward(self, encoder_features: dict[str, torch.Tensor], num_stages: int) -> torch.Tensor:
        deepest_key = f"stage{num_stages}"
        feat = encoder_features[deepest_key]
        B = feat.shape[0]
        x = feat.flatten(1)
        x = self.dense_compress(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dense_expand(x)
        x = x.reshape(B, self.ae_dims, self.lowest_dense_res, self.lowest_dense_res)
        x = self.upscale(x)  # 2x resolution: (B, ae_dims, res*2, res*2)
        return x


class TFMModel(nn.Module):
    """TFM face swap model with Swin Transformer encoder + DFL CNN decoder.

    Architecture:
    - Shared encoder (Swin Transformer) extracts multi-scale features
    - Inter bottleneck: Dense compress → ae_dims → Dense expand → spatial map → Upscale 2x
      This FORCES identity out of the latent code (DFL's key insight)
    - decoder_src / decoder_dst: DFL CNN decoder (Upscale + ResBlock)
      Identity is in DECODER WEIGHTS, not in any embedding
    - Swap: encoder(dst) → Inter → decoder_src → SRC face on DST structure

    Why this works:
    - Inter bottleneck compresses 128:1 → identity cannot survive in code
    - CNN decoder takes spatial feature map → identity naturally encoded in conv weights
    - decoder_src only trained on SRC → its weights encode SRC identity
    - decoder_src(dst_code) = SRC appearance + DST structure = face swap
    """

    def __init__(
        self,
        resolution: int = 128,
        encoder_type: str = "swin",
        embed_dim: int = 96,
        depths: Optional[list[int]] = None,
        num_heads: Optional[list[int]] = None,
        window_size: int = 8,
        ae_dims: int = 256,
        d_dims: int = 64,
        gan_power: float = 0.0,
        drop_rate: float = 0.1,
        gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.resolution = resolution
        self.encoder_type = encoder_type
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.ae_dims = ae_dims
        self.d_dims = d_dims
        self.gan_power = gan_power
        self.gradient_checkpoint = gradient_checkpoint

        self.encoder = TFMEncoder(
            img_size=resolution,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=drop_rate,
            gradient_checkpoint=gradient_checkpoint,
        )

        last_dim = embed_dim * (2 ** (len(depths) - 1))
        patches_res = resolution // 4
        for _ in range(len(depths) - 1):
            patches_res //= 2
        encoder_out_dim = last_dim * patches_res * patches_res

        self.inter = TFMInter(
            encoder_out_dim=encoder_out_dim,
            ae_dims=ae_dims,
            lowest_dense_res=patches_res,
        )

        self.decoder_src = TFMDecoder(
            resolution=resolution,
            ae_dims=ae_dims,
            d_dims=d_dims,
        )
        self.decoder_dst = TFMDecoder(
            resolution=resolution,
            ae_dims=ae_dims,
            d_dims=d_dims,
        )

        self.decoder = self.decoder_src

        self.discriminator = None
        if gan_power > 0:
            self.discriminator = TFMDiscriminator()

    def encode(self, img: torch.Tensor) -> tuple[dict, torch.Tensor]:
        encoder_features = self.encoder(img)
        w_plus = self.inter(encoder_features, len(self.depths))
        return encoder_features, w_plus

    def decode_src(self, w_plus: torch.Tensor) -> torch.Tensor:
        return self.decoder_src(w_plus)

    def decode_dst(self, w_plus: torch.Tensor) -> torch.Tensor:
        return self.decoder_dst(w_plus)

    def forward(self, img: torch.Tensor, which: str = "src") -> torch.Tensor:
        _, w_plus = self.encode(img)
        decoder = self.decoder_src if which == "src" else self.decoder_dst
        return decoder(w_plus)

    def forward_with_features(
        self,
        img: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoder_features, w_plus = self.encode(img)
        output = self.decoder_src(w_plus)
        features = {
            "encoder_features": encoder_features,
            "w_plus": w_plus,
        }
        return output, features

    def discriminate(self, img: torch.Tensor) -> torch.Tensor:
        if self.discriminator is None:
            raise RuntimeError("Discriminator not initialized (gan_power=0)")
        return self.discriminator(img)

    def get_encoder(self) -> TFMEncoder:
        return self.encoder

    def get_decoder(self) -> TFMDecoder:
        return self.decoder_src

    def get_discriminator(self) -> Optional[TFMDiscriminator]:
        return self.discriminator

    def save(self, model_dir: Path) -> None:
        from DeepFaceLab.shared.file_manager import FileManager
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        import io
        components = {
            "TFM_encoder": self.encoder,
            "TFM_inter": self.inter,
            "TFM_decoder_src": self.decoder_src,
            "TFM_decoder_dst": self.decoder_dst,
        }
        if self.discriminator is not None:
            components["TFM_discriminator"] = self.discriminator

        for name, module in components.items():
            buf = io.BytesIO()
            torch.save(module.state_dict(), buf)
            FileManager.atomic_write(model_dir / f"{name}.pt", buf.getvalue())

        import json
        config = self.get_config()
        FileManager.atomic_write(model_dir / "TFM_model_config.json", json.dumps(config, indent=2))

    def load(self, model_dir: Path, device: Optional[torch.device] = None) -> bool:
        model_dir = Path(model_dir)
        if device is None:
            device = next(self.parameters()).device

        components = {
            "TFM_encoder": self.encoder,
            "TFM_inter": self.inter,
            "TFM_decoder_src": self.decoder_src,
            "TFM_decoder_dst": self.decoder_dst,
        }
        if self.discriminator is not None:
            components["TFM_discriminator"] = self.discriminator

        for name, module in components.items():
            path = model_dir / f"{name}.pt"
            if path.exists():
                try:
                    data = open(str(path), "rb").read()
                    state = torch.load(io.BytesIO(data), map_location=device, weights_only=True)
                    module.load_state_dict(state)
                except Exception as e:
                    _logger.warning(f"Failed to load {name}: {e}")
                    return False
        return True

    def get_config(self) -> dict:
        return {
            "resolution": self.resolution,
            "encoder_type": self.encoder_type,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "ae_dims": self.ae_dims,
            "d_dims": self.d_dims,
            "gan_power": self.gan_power,
        }

    @classmethod
    def from_preset(cls, preset: str = "medium", resolution: int = 128, gan_power: float = 0.0, window_size: int = 8, ae_dims: int = 256, d_dims: int = 64, gradient_checkpoint: bool = False, **kwargs) -> "TFMModel":
        if preset not in _TFM_PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(_TFM_PRESETS.keys())}")
        cfg = _TFM_PRESETS[preset]
        return cls(
            resolution=resolution,
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            window_size=window_size,
            ae_dims=ae_dims,
            d_dims=d_dims,
            gan_power=gan_power,
            gradient_checkpoint=gradient_checkpoint,
            **kwargs,
        )
