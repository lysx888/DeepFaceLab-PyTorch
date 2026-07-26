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
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm1(x_flat)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        _, _, Hp, Wp = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        x_windows = _window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)

        N = self.window_size * self.window_size
        qkv = self.qkv(x_windows).reshape(-1, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        index = self.relative_position_index.reshape(-1)
        rel_pos_bias = self.relative_position_bias_table[index]
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
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = x.flatten(2).transpose(1, 2)
        x = self.reduction(self.norm(x))
        x = x.transpose(1, 2).reshape(B, 2 * C, H // 2, W // 2)
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        features = {}
        dim = self.embed_dim
        for i in range(self.num_stages):
            for block in self.stages[i]:
                x = block(x)
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


class AdaINModule(nn.Module):
    def __init__(self, identity_dim: int = 512, w_dim: int = 512, feature_dim: int = 512) -> None:
        super().__init__()
        self.identity_proj = nn.Sequential(nn.Linear(identity_dim, 256), nn.ReLU())
        self.style_proj = nn.Sequential(nn.Linear(w_dim, 256), nn.ReLU())
        self.modulation = nn.Linear(512, 2 * feature_dim)

    def forward(self, x: torch.Tensor, identity_embed: torch.Tensor, w_style: torch.Tensor) -> torch.Tensor:
        id_feat = self.identity_proj(identity_embed)
        style_feat = self.style_proj(w_style)
        combined = torch.cat([id_feat, style_feat], dim=1)
        params = self.modulation(combined)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + 1e-8).sqrt()
        return gamma * x_norm + beta


class SynthesisLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int = 512,
        identity_dim: int = 512,
        is_up: bool = True,
    ) -> None:
        super().__init__()
        self.is_up = is_up
        if is_up:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.adain = AdaINModule(identity_dim, w_dim, out_channels)
        self.noise = NoiseInjection()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        identity_embed: torch.Tensor,
        w_style: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_up:
            x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = x + skip
        x = self.conv(x)
        x = self.adain(x, identity_embed, w_style)
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
    def __init__(
        self,
        resolution: int = 128,
        w_dim: int = 512,
        identity_dim: int = 512,
        base_channels: int = 512,
        encoder_dims: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.w_dim = w_dim
        self.num_layers = 7 if resolution >= 256 else 6

        if encoder_dims is None:
            encoder_dims = [96, 192, 384, 768]

        self.const_input = ConstInput(base_channels, size=4)

        channels = base_channels
        self.synthesis_layers = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.to_rgb_layers.append(ToRGB(channels))

        up_layers = self.num_layers
        channel_schedule = []
        ch = channels
        for i in range(up_layers):
            channel_schedule.append(ch)
            if i >= 2 and ch > 32:
                ch = max(ch // 2, 32)

        for i in range(up_layers):
            out_ch = channel_schedule[i + 1] if i + 1 < len(channel_schedule) else channel_schedule[-1]
            in_ch = channel_schedule[i]
            self.synthesis_layers.append(
                SynthesisLayer(in_ch, out_ch, w_dim, identity_dim, is_up=True)
            )
            self.to_rgb_layers.append(ToRGB(out_ch))

            enc_idx = self.num_layers - 2 - i
            if 0 <= enc_idx < len(encoder_dims):
                self.skip_convs.append(nn.Conv2d(encoder_dims[enc_idx], in_ch, 1))
            else:
                self.skip_convs.append(None)

    def forward(
        self,
        w_plus: torch.Tensor,
        identity_embed: torch.Tensor,
        encoder_features: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B = w_plus.shape[0]
        x = self.const_input(B)

        rgb = self.to_rgb_layers[0](x)

        for i, layer in enumerate(self.synthesis_layers):
            w_idx = min(i + 1, w_plus.shape[1] - 1)
            w_style = w_plus[:, w_idx]

            skip = None
            if encoder_features is not None and i < len(self.skip_convs) and self.skip_convs[i] is not None:
                enc_idx = self.num_layers - 2 - i
                enc_key = f"stage{enc_idx + 1}"
                if enc_key in encoder_features:
                    enc_feat = encoder_features[enc_key]
                    skip = self.skip_convs[i](enc_feat)
                    target_h = x.shape[2] * 2 if layer.is_up else x.shape[2]
                    target_w = x.shape[3] * 2 if layer.is_up else x.shape[3]
                    if skip.shape[2:] != (target_h, target_w):
                        skip = F.interpolate(skip, size=(target_h, target_w), mode="bilinear", align_corners=False)

            x = layer(x, identity_embed, w_style, skip)
            rgb_up = self.to_rgb_layers[i + 1](x)
            if rgb.shape[2:] != rgb_up.shape[2:]:
                rgb = F.interpolate(rgb, size=rgb_up.shape[2:], mode="bilinear", align_corners=False)
            rgb = rgb + rgb_up

        if rgb.shape[2] != self.resolution or rgb.shape[3] != self.resolution:
            rgb = F.interpolate(rgb, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)

        return rgb


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


class TFMModel(nn.Module):
    def __init__(
        self,
        resolution: int = 128,
        encoder_type: str = "swin",
        embed_dim: int = 96,
        depths: Optional[list[int]] = None,
        num_heads: Optional[list[int]] = None,
        window_size: int = 8,
        w_dim: int = 512,
        identity_dim: int = 512,
        gan_power: float = 0.0,
        base_channels: int = 512,
        drop_rate: float = 0.1,
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
        self.w_dim = w_dim
        self.identity_dim = identity_dim
        self.gan_power = gan_power
        self.base_channels = base_channels

        self.encoder = TFMEncoder(
            img_size=resolution,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=drop_rate,
        )

        last_dim = embed_dim * (2 ** (len(depths) - 1))
        num_dec_layers = 7 if resolution >= 256 else 6

        self.wplus_mapper = WPlusMapper(
            embed_dim_last=last_dim,
            num_layers=num_dec_layers,
            w_dim=w_dim,
        )

        encoder_dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        self.decoder = TFMDecoder(
            resolution=resolution,
            w_dim=w_dim,
            identity_dim=identity_dim,
            base_channels=base_channels,
            encoder_dims=encoder_dims,
        )

        self.discriminator = None
        if gan_power > 0:
            self.discriminator = TFMDiscriminator()

    def forward(self, img: torch.Tensor, identity_embed: torch.Tensor) -> torch.Tensor:
        result, _ = self.forward_with_features(img, identity_embed)
        return result

    def encode(self, img: torch.Tensor) -> tuple[dict, torch.Tensor]:
        encoder_features = self.encoder(img)
        stage4_feat = encoder_features[f"stage{len(self.depths)}"]
        w_plus = self.wplus_mapper(stage4_feat)
        return encoder_features, w_plus

    def decode(self, w_plus: torch.Tensor, identity_embed: torch.Tensor, encoder_features: dict) -> torch.Tensor:
        return self.decoder(w_plus, identity_embed, encoder_features)

    def forward_with_features(
        self,
        img: torch.Tensor,
        identity_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoder_features = self.encoder(img)
        stage4_feat = encoder_features[f"stage{len(self.depths)}"]
        w_plus = self.wplus_mapper(stage4_feat)
        output = self.decoder(w_plus, identity_embed, encoder_features)
        features = {
            "encoder_features": encoder_features,
            "w_plus": w_plus,
            "stage4_feat": stage4_feat,
        }
        return output, features

    def discriminate(self, img: torch.Tensor) -> torch.Tensor:
        if self.discriminator is None:
            raise RuntimeError("Discriminator not initialized (gan_power=0)")
        return self.discriminator(img)

    def get_encoder(self) -> TFMEncoder:
        return self.encoder

    def get_decoder(self) -> TFMDecoder:
        return self.decoder

    def get_adain(self) -> AdaINModule:
        return self.decoder

    def get_discriminator(self) -> Optional[TFMDiscriminator]:
        return self.discriminator

    def save(self, model_dir: Path) -> None:
        from DeepFaceLab.shared.file_manager import FileManager
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        import io
        components = {
            "TFM_encoder": self.encoder,
            "TFM_decoder": self.decoder,
            "TFM_wplus_mapper": self.wplus_mapper,
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
            "TFM_decoder": self.decoder,
            "TFM_wplus_mapper": self.wplus_mapper,
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
            "w_dim": self.w_dim,
            "identity_dim": self.identity_dim,
            "gan_power": self.gan_power,
            "base_channels": self.base_channels,
        }

    @classmethod
    def from_preset(cls, preset: str = "medium", resolution: int = 128, gan_power: float = 0.0, window_size: int = 8, identity_dim: int = 512, **kwargs) -> "TFMModel":
        if preset not in _TFM_PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(_TFM_PRESETS.keys())}")
        cfg = _TFM_PRESETS[preset]
        return cls(
            resolution=resolution,
            embed_dim=cfg["embed_dim"],
            depths=cfg["depths"],
            num_heads=cfg["num_heads"],
            window_size=window_size,
            w_dim=cfg["w_dim"],
            identity_dim=identity_dim,
            gan_power=gan_power,
            base_channels=cfg["base_channels"],
            **kwargs,
        )
