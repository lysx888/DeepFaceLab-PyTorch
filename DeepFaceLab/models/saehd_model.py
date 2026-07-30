import io
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from DeepFaceLab.models.building_blocks import (
    Downscale,
    DownscaleBlock,
    Upscale,
    ResidualBlock,
    UNetPatchDiscriminator,
    CodeDiscriminator,
    VGGPerceptualLoss,
    ca_init_weights,
)
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("saehd_model")


class SAEHDEncoder(nn.Module):
    """
    DFL encoder with optional 't' architecture variant.
    
    Default (4 steps): DownscaleBlock(4) -> flatten -> optional pixel_norm
      3->e@res -> 2e@res/2 -> 4e@res/4 -> 8e@res/8 -> 8e@res/16
    
    't' variant (5 steps + ResBlocks):
      down1(in->e) -> res1 -> down2(e->2e) -> down3(2e->4e) -> down4(4e->8e) -> down5(8e->8e) -> res5
      Output res/32
    """

    def __init__(self, in_ch: int = 3, e_dims: int = 64, use_t: bool = False,
                 use_pixel_norm: bool = False, use_c: bool = False):
        super().__init__()
        self.use_t = use_t
        self.use_pixel_norm = use_pixel_norm

        if use_t:
            self.down1 = Downscale(in_ch, e_dims, kernel_size=5, use_c=use_c)
            self.res1 = ResidualBlock(e_dims, use_c=use_c)
            self.down2 = Downscale(e_dims, e_dims * 2, kernel_size=5, use_c=use_c)
            self.down3 = Downscale(e_dims * 2, e_dims * 4, kernel_size=5, use_c=use_c)
            self.down4 = Downscale(e_dims * 4, e_dims * 8, kernel_size=5, use_c=use_c)
            self.down5 = Downscale(e_dims * 8, e_dims * 8, kernel_size=5, use_c=use_c)
            self.res5 = ResidualBlock(e_dims * 8, use_c=use_c)
            self.out_ch = e_dims * 8
        else:
            self.down_block = DownscaleBlock(in_ch, e_dims, n_downscales=4, use_c=use_c)
            self.out_ch = e_dims * min(2 ** 3, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_t:
            x = self.down1(x)
            x = self.res1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.down5(x)
            x = self.res5(x)
        else:
            x = self.down_block(x)
        x = x.flatten(1)
        if self.use_pixel_norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return x

    def get_out_res(self, resolution: int) -> int:
        return resolution // (2 ** 5) if self.use_t else resolution // (2 ** 4)


class SAEHDInter(nn.Module):
    """
    Dense bottleneck: flatten -> Dense(in, ae_ch) -> Dense(ae_ch, res^2*ae_out_ch)
    -> reshape -> optional Upscale(2x)
    
    't' variant: skip Upscale (encoder already downsampled to res/32)
    """

    def __init__(self, in_features: int, ae_ch: int, ae_out_ch: int,
                 bottleneck_res: int, use_t: bool = False, use_c: bool = False):
        super().__init__()
        self.ae_out_ch = ae_out_ch
        self.bottleneck_res = bottleneck_res
        self.use_t = use_t
        self.dense1 = nn.Linear(in_features, ae_ch)
        self.dense2 = nn.Linear(ae_ch, bottleneck_res * bottleneck_res * ae_out_ch)
        if not use_t:
            self.upscale = Upscale(ae_out_ch, ae_out_ch, use_c=use_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.dense2(x)
        B, _ = x.shape
        x = x.reshape(B, self.ae_out_ch, self.bottleneck_res, self.bottleneck_res)
        if not self.use_t:
            x = self.upscale(x)
        return x

    def get_out_ch(self) -> int:
        return self.ae_out_ch

    def get_out_res(self) -> int:
        return self.bottleneck_res if self.use_t else self.bottleneck_res * 2


class SAEHDDecoder(nn.Module):
    """
    DFL decoder with optional 't' architecture variant.
    
    Default: 3 upscale stages with residual blocks + separate mask branch.
    't' variant: 4 upscale stages, 2nd stage keeps d*8 channels.
    
    With use_depth_to_space=True ('d' option):
      Final output uses 4 parallel convs + depth_to_space for 2x resolution boost.
    """

    def __init__(self, in_ch: int, d_dims: int = 64, d_mask_dims: int = 21,
                 use_depth_to_space: bool = False, use_t: bool = False, use_c: bool = False):
        super().__init__()
        self.use_depth_to_space = use_depth_to_space
        self.use_t = use_t

        self.upscale0 = Upscale(in_ch, d_dims * 8, use_c=use_c)
        self.res0 = ResidualBlock(d_dims * 8, use_c=use_c)

        if use_t:
            self.upscale1 = Upscale(d_dims * 8, d_dims * 8, use_c=use_c)
            self.res1 = ResidualBlock(d_dims * 8, use_c=use_c)
            self.upscale2 = Upscale(d_dims * 8, d_dims * 4, use_c=use_c)
            self.res2 = ResidualBlock(d_dims * 4, use_c=use_c)
            self.upscale3 = Upscale(d_dims * 4, d_dims * 2, use_c=use_c)
            self.res3 = ResidualBlock(d_dims * 2, use_c=use_c)
        else:
            self.upscale1 = Upscale(d_dims * 8, d_dims * 4, use_c=use_c)
            self.res1 = ResidualBlock(d_dims * 4, use_c=use_c)
            self.upscale2 = Upscale(d_dims * 4, d_dims * 2, use_c=use_c)
            self.res2 = ResidualBlock(d_dims * 2, use_c=use_c)

        if use_depth_to_space:
            self.out_conv = nn.Conv2d(d_dims * 2, 3, kernel_size=1, padding=0)
            self.out_conv1 = nn.Conv2d(d_dims * 2, 3, kernel_size=3, padding=1)
            self.out_conv2 = nn.Conv2d(d_dims * 2, 3, kernel_size=3, padding=1)
            self.out_conv3 = nn.Conv2d(d_dims * 2, 3, kernel_size=3, padding=1)
        else:
            self.out_conv = nn.Conv2d(d_dims * 2, 3, kernel_size=1, padding=0)

        self.upscalem0 = Upscale(in_ch, d_mask_dims * 8, use_c=use_c)
        self.upscalem1 = Upscale(d_mask_dims * 8, d_mask_dims * 4, use_c=use_c)
        self.upscalem2 = Upscale(d_mask_dims * 4, d_mask_dims * 2, use_c=use_c)

        if use_t:
            self.upscalem3 = Upscale(d_mask_dims * 2, d_mask_dims * 1, use_c=use_c)
            if use_depth_to_space:
                self.upscalem4 = Upscale(d_mask_dims * 1, d_mask_dims * 1, use_c=use_c)
                self.out_convm = nn.Conv2d(d_mask_dims * 1, 1, kernel_size=1, padding=0)
            else:
                self.out_convm = nn.Conv2d(d_mask_dims * 1, 1, kernel_size=1, padding=0)
        else:
            if use_depth_to_space:
                self.upscalem3 = Upscale(d_mask_dims * 2, d_mask_dims * 1, use_c=use_c)
                self.out_convm = nn.Conv2d(d_mask_dims * 1, 1, kernel_size=1, padding=0)
            else:
                self.out_convm = nn.Conv2d(d_mask_dims * 2, 1, kernel_size=1, padding=0)

    def _depth_to_space(self, x: torch.Tensor, block_size: int = 2) -> torch.Tensor:
        B, C, H, W = x.shape
        oc = C // (block_size * block_size)
        x = x.reshape(B, block_size, block_size, oc, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2)
        x = x.reshape(B, oc, H * block_size, W * block_size)
        return x

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)
        if self.use_t:
            x = self.upscale3(x)
            x = self.res3(x)

        if self.use_depth_to_space:
            c0 = self.out_conv(x)
            c1 = self.out_conv1(x)
            c2 = self.out_conv2(x)
            c3 = self.out_conv3(x)
            face = torch.sigmoid(self._depth_to_space(torch.cat([c0, c1, c2, c3], dim=1), 2))
        else:
            face = torch.sigmoid(self.out_conv(x))

        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        if self.use_t:
            m = self.upscalem3(m)
            if self.use_depth_to_space:
                m = self.upscalem4(m)
        else:
            if self.use_depth_to_space:
                m = self.upscalem3(m)
        mask = torch.sigmoid(self.out_convm(m))

        return face, mask


class SAEHDModel(nn.Module):
    """
    Full SAEHD model supporting 'df' and 'liae' architectures.

    df architecture:
        encoder -> inter -> decoder_src (for src reconstruction)
        encoder -> inter -> decoder_dst (for dst reconstruction)
        SWAP: encoder(dst) -> inter -> decoder_src -> src face on dst structure

    liae architecture:
        encoder -> inter_AB, inter_B -> decoder (shared)
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
        self.archi_type = archi_parts[0]
        self.archi_opts = archi_parts[1] if len(archi_parts) > 1 else ""
        self.use_liae = (self.archi_type == "liae")
        self.use_depth_to_space = "d" in self.archi_opts
        self.use_pixel_norm = "u" in self.archi_opts
        self.use_t = "t" in self.archi_opts
        self.use_c = "c" in self.archi_opts

        n_downscales = 0
        r = resolution
        while r > 4:
            r //= 2
            n_downscales += 1
        enc_downscales = 5 if self.use_t else 4
        bottleneck_res = resolution // (32 if self.use_depth_to_space else 16)

        self.encoder = SAEHDEncoder(
            in_ch=3, e_dims=e_dims, use_t=self.use_t,
            use_pixel_norm=self.use_pixel_norm,
            use_c=self.use_c,
        )
        enc_spatial_res = resolution // (2 ** enc_downscales)
        enc_out_features = self.encoder.out_ch * (enc_spatial_res ** 2)

        if not self.use_liae:
            self.inter = SAEHDInter(
                in_features=enc_out_features,
                ae_ch=ae_dims,
                ae_out_ch=ae_dims,
                bottleneck_res=bottleneck_res,
                use_t=self.use_t,
                use_c=self.use_c,
            )
            inter_out_ch = ae_dims
            self.decoder_src = SAEHDDecoder(inter_out_ch, d_dims, self.d_mask_dims,
                                            use_depth_to_space=self.use_depth_to_space,
                                            use_t=self.use_t, use_c=self.use_c)
            self.decoder_dst = SAEHDDecoder(inter_out_ch, d_dims, self.d_mask_dims,
                                            use_depth_to_space=self.use_depth_to_space,
                                            use_t=self.use_t, use_c=self.use_c)
        else:
            self.inter_AB = SAEHDInter(
                in_features=enc_out_features,
                ae_ch=ae_dims,
                ae_out_ch=ae_dims * 2,
                bottleneck_res=bottleneck_res,
                use_t=self.use_t,
                use_c=self.use_c,
            )
            self.inter_B = SAEHDInter(
                in_features=enc_out_features,
                ae_ch=ae_dims,
                ae_out_ch=ae_dims * 2,
                bottleneck_res=bottleneck_res,
                use_t=self.use_t,
                use_c=self.use_c,
            )
            shared_dec_in_ch = ae_dims * 2 * 2
            self.decoder = SAEHDDecoder(shared_dec_in_ch, d_dims, self.d_mask_dims,
                                        use_depth_to_space=self.use_depth_to_space,
                                        use_t=self.use_t, use_c=self.use_c)

        self.gan_dims = gan_dims
        self.gan_patch_size = gan_patch_size if gan_patch_size is not None else max(8, resolution // 8)
        self.discriminator = None
        self.code_discriminator = None

        self._init_bottleneck_weights()

    def _init_bottleneck_weights(self):
        """Initialize bottleneck Dense layers with Xavier uniform (DFL behavior)."""
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
        return self.encoder(img)

    def build_discriminator(self) -> UNetPatchDiscriminator:
        self.discriminator = UNetPatchDiscriminator(
            in_ch=3, base_ch=self.gan_dims, patch_size=self.gan_patch_size,
        )
        return self.discriminator

    def discriminate(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.discriminator is None:
            raise RuntimeError("Discriminator not built. Call build_discriminator() first.")
        return self.discriminator(img)

    def build_code_discriminator(self, code_res: int) -> CodeDiscriminator:
        if self.use_liae:
            raise RuntimeError("CodeDiscriminator only supports df architecture")
        self.code_discriminator = CodeDiscriminator(
            in_ch=self.ae_dims, code_res=code_res, base_ch=256,
        )
        return self.code_discriminator

    def decode_src(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_liae:
            inter_out = self.inter(code)
            return self.decoder_src(inter_out)
        else:
            raise RuntimeError("Use decode_liae() for liae architecture")

    def decode_dst(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_liae:
            inter_out = self.inter(code)
            return self.decoder_dst(inter_out)
        else:
            raise RuntimeError("Use decode_liae() for liae architecture")

    def initialize_ca_weights(self) -> None:
        ca_init_weights(self)
        _logger.info("CA weights initialized")

    def forward_df(self, src_img: torch.Tensor, dst_img: torch.Tensor) -> dict:
        src_code = self.encoder(src_img)
        dst_code = self.encoder(dst_img)

        src_inter = self.inter(src_code)
        dst_inter = self.inter(dst_code)

        pred_src_src, pred_src_srcm = self.decoder_src(src_inter)
        pred_dst_dst, pred_dst_dstm = self.decoder_dst(dst_inter)
        pred_src_dst, pred_src_dstm = self.decoder_src(dst_inter)

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
            "pred_src_src_masked": pred_src_src * pred_src_srcm + src_img * (1 - pred_src_srcm),
            "pred_dst_dst_masked": pred_dst_dst * pred_dst_dstm + dst_img * (1 - pred_dst_dstm),
            "pred_src_dst_masked": pred_src_dst * pred_src_dstm + dst_img * (1 - pred_src_dstm),
        }

    def forward_liae(self, src_img: torch.Tensor, dst_img: torch.Tensor) -> dict:
        src_enc = self.encoder(src_img)
        dst_enc = self.encoder(dst_img)

        src_inter_AB = self.inter_AB(src_enc)
        dst_inter_AB = self.inter_AB(dst_enc)
        dst_inter_B = self.inter_B(dst_enc)

        src_code = torch.cat([src_inter_AB, src_inter_AB], dim=1)
        dst_code = torch.cat([dst_inter_B, dst_inter_AB], dim=1)
        swap_code = torch.cat([dst_inter_AB, dst_inter_AB], dim=1)

        pred_src_src, pred_src_srcm = self.decoder(src_code)
        pred_dst_dst, pred_dst_dstm = self.decoder(dst_code)
        pred_src_dst, pred_src_dstm = self.decoder(swap_code)

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
            "pred_src_src_masked": pred_src_src * pred_src_srcm + src_img * (1 - pred_src_srcm),
            "pred_dst_dst_masked": pred_dst_dst * pred_dst_dstm + dst_img * (1 - pred_dst_dstm),
            "pred_src_dst_masked": pred_src_dst * pred_src_dstm + dst_img * (1 - pred_src_dstm),
        }

    def forward(self, src_img: torch.Tensor, dst_img: torch.Tensor) -> dict:
        if self.use_liae:
            return self.forward_liae(src_img, dst_img)
        return self.forward_df(src_img, dst_img)

    def merge(self, dst_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            if not self.use_liae:
                dst_code = self.encoder(dst_img)
                dst_inter = self.inter(dst_code)
                swapped, swap_mask = self.decoder_src(dst_inter)
                swapped = swapped * swap_mask
                _, dst_mask = self.decoder_dst(dst_inter)
            else:
                dst_enc = self.encoder(dst_img)
                dst_inter_AB = self.inter_AB(dst_enc)
                dst_inter_B = self.inter_B(dst_enc)
                swap_code = torch.cat([dst_inter_AB, dst_inter_AB], dim=1)
                dst_code = torch.cat([dst_inter_B, dst_inter_AB], dim=1)
                swapped, swap_mask = self.decoder(swap_code)
                swapped = swapped * swap_mask
                _, dst_mask = self.decoder(dst_code)
        return swapped, dst_mask, swap_mask

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
        if not self.use_liae:
            return list(self.parameters())
        return list(self.parameters())
