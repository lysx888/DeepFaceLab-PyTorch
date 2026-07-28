import io
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("saehd_model")


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

    Face branch: Upscale→Res → Upscale→Res → Upscale→Res → Conv1x1→Sigmoid
    Mask branch: Upscale → Upscale → Upscale → Conv1x1→Sigmoid

    For res=128, d_dims=64:
      Input: 512ch @ 8x8 (from inter)
      Face: 512@16→Res→256@32→Res→128@64→Res→ 1x1conv→3@128→Sigmoid
      Mask: 21@16→21@32→21@64→ 1x1conv→1@128→Sigmoid
    """

    def __init__(self, in_ch: int, d_dims: int = 64, d_mask_dims: int = 21):
        super().__init__()
        # Face branch
        self.upscale0 = Upscale(in_ch, d_dims * 8)
        self.res0 = ResidualBlock(d_dims * 8)
        self.upscale1 = Upscale(d_dims * 8, d_dims * 4)
        self.res1 = ResidualBlock(d_dims * 4)
        self.upscale2 = Upscale(d_dims * 4, d_dims * 2)
        self.res2 = ResidualBlock(d_dims * 2)
        self.out_conv = nn.Conv2d(d_dims * 2, 3, kernel_size=1, padding=0)

        # Mask branch
        self.upscalem0 = Upscale(in_ch, d_mask_dims * 8)
        self.upscalem1 = Upscale(d_mask_dims * 8, d_mask_dims * 4)
        self.upscalem2 = Upscale(d_mask_dims * 4, d_mask_dims * 2)
        self.out_convm = nn.Conv2d(d_mask_dims * 2, 1, kernel_size=1, padding=0)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Face
        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)
        face = torch.sigmoid(self.out_conv(x))

        # Mask
        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
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
    ):
        super().__init__()
        self.resolution = resolution
        self.architecture = architecture
        self.ae_dims = ae_dims
        self.e_dims = e_dims
        self.d_dims = d_dims
        self.d_mask_dims = d_mask_dims if d_mask_dims is not None else (d_dims // 3 + d_dims // 3 % 2)
        self.use_liae = (architecture == "liae")

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

        # Encoder
        self.encoder = SAEHDEncoder(
            in_ch=3, e_dims=e_dims, n_downscales=enc_downscales,
            use_pixel_norm=self.use_liae,  # 'u' option equivalent for liae
        )
        enc_out_features = self.encoder.out_ch * (bottleneck_res ** 2)

        if not self.use_liae:
            # df architecture: shared inter + separate decoders
            self.inter = SAEHDInter(
                in_features=enc_out_features,
                ae_ch=ae_dims,
                ae_out_ch=ae_dims,
                bottleneck_res=bottleneck_res,
            )
            inter_out_ch = ae_dims
            self.decoder_src = SAEHDDecoder(inter_out_ch, d_dims, self.d_mask_dims)
            self.decoder_dst = SAEHDDecoder(inter_out_ch, d_dims, self.d_mask_dims)
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
            self.decoder = SAEHDDecoder(shared_dec_in_ch, d_dims, self.d_mask_dims)

    # ---- Forward methods for training ----

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode image to bottleneck code."""
        return self.encoder(img)

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

    def forward_df(self, src_img: torch.Tensor, dst_img: torch.Tensor) -> dict:
        """
        Full forward pass for df architecture.
        Returns dict with all reconstructions and swap outputs.
        """
        src_code = self.encoder(src_img)
        dst_code = self.encoder(dst_img)

        src_inter = self.inter(src_code)
        dst_inter = self.inter(dst_code)

        # Self-reconstruction
        pred_src_src, pred_src_srcm = self.decoder_src(src_inter)
        pred_dst_dst, pred_dst_dstm = self.decoder_dst(dst_inter)

        # Swap: decoder_src with dst's code
        pred_src_dst, pred_src_dstm = self.decoder_src(dst_inter)

        # Swap with stop_gradient on code (for style loss)
        dst_inter_detached = dst_inter.detach()
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
                swapped, swap_mask = self.decoder_src(dst_inter)
                _, dst_mask = self.decoder_dst(dst_inter)
            else:
                dst_enc = self.encoder(dst_img)
                dst_inter_AB = self.inter_AB(dst_enc)
                dst_inter_B = self.inter_B(dst_enc)
                swap_code = torch.cat([dst_inter_AB, dst_inter_AB], dim=1)
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

        _logger.info(f"SAEHD model loaded from {model_dir}")
        return True

    @classmethod
    def from_config(cls, config: dict) -> "SAEHDModel":
        return cls(
            resolution=config.get("resolution", 128),
            architecture=config.get("architecture", "df"),
            ae_dims=config.get("ae_dims", 256),
            e_dims=config.get("e_dims", 64),
            d_dims=config.get("d_dims", 64),
            d_mask_dims=config.get("d_mask_dims", None),
        )

    def get_param_groups(self) -> list:
        """Return parameter groups for optimizer (separate src/dst for liae)."""
        if not self.use_liae:
            return list(self.parameters())
        return list(self.parameters())
