"""Quick96模型：快速训练模型，全部参数硬编码。

分辨率96，使用DeepFakeArchi(opts='ud')架构，RMSprop优化器。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from faceswap.models.base_model import BaseModel
from faceswap.models.saehd.saehd_arch import Encoder, Inter, Decoder
from faceswap.models.saehd.losses import dssim, blur_mask, dssim_filter_sizes
from faceswap.models.saehd.optimizers import RMSprop
from faceswap.shared.logger import get_logger

_logger = get_logger("quick96_model")

_RESOLUTION = 96
_AE_DIMS = 128
_E_DIMS = 64
_D_DIMS = 64
_D_MASK_DIMS = 16


class Quick96Model(BaseModel):
    _model_prefix = "quick96"
    _param_labels = {}

    def build(self) -> None:
        self.resolution = _RESOLUTION
        self.ae_dims = _AE_DIMS
        self.e_dims = _E_DIMS
        self.d_dims = _D_DIMS
        self.d_mask_dims = _D_MASK_DIMS
        self.masked_training = True

        opts = 'ud'
        self.encoder = Encoder(in_ch=3, e_ch=self.e_dims,
                                resolution=self.resolution, opts=opts)
        self.register_module('encoder', self.encoder)

        encoder_out_flat = self.encoder.get_out_ch() * self.encoder.get_out_res(self.resolution) ** 2

        self.inter = Inter(in_ch=encoder_out_flat, ae_ch=self.ae_dims,
                            ae_out_ch=self.ae_dims, resolution=self.resolution, opts=opts)
        self.register_module('inter', self.inter)
        inter_out_ch = self.inter.get_out_ch()

        self.decoder_src = Decoder(in_ch=inter_out_ch, d_ch=self.d_dims,
                                    d_mask_ch=self.d_mask_dims, opts=opts)
        self.decoder_dst = Decoder(in_ch=inter_out_ch, d_ch=self.d_dims,
                                    d_mask_ch=self.d_mask_dims, opts=opts)
        self.register_module('decoder_src', self.decoder_src)
        self.register_module('decoder_dst', self.decoder_dst)

    def build_optimizers(self) -> None:
        gen_params = []
        for module in self._modules_dict.values():
            gen_params.extend(p for p in module.parameters() if p.requires_grad)
        opt = RMSprop(gen_params, lr=2e-4, lr_dropout=0.3)
        self.register_optimizer('src_dst_opt', opt)

    def forward(self, warped_src, warped_dst) -> dict:
        src_code = self.inter(self.encoder(warped_src))
        dst_code = self.inter(self.encoder(warped_dst))

        pred_src_src, pred_src_srcm = self.decoder_src(src_code)
        pred_dst_dst, pred_dst_dstm = self.decoder_dst(dst_code)
        pred_src_dst, pred_src_dstm = self.decoder_src(dst_code)

        return {
            'pred_src_src': pred_src_src, 'pred_src_srcm': pred_src_srcm,
            'pred_dst_dst': pred_dst_dst, 'pred_dst_dstm': pred_dst_dstm,
            'pred_src_dst': pred_src_dst, 'pred_src_dstm': pred_src_dstm,
        }

    def compute_loss(self, batch_src: dict, batch_dst: dict, fw: dict) -> dict:
        res = self.resolution
        blur_sigma = max(1, res // 32)

        target_src = batch_src['target_image']
        target_srcm = batch_src['target_mask']
        target_dst = batch_dst['target_image']
        target_dstm = batch_dst['target_mask']

        pred_src_src = fw['pred_src_src']
        pred_src_srcm = fw['pred_src_srcm']
        pred_dst_dst = fw['pred_dst_dst']
        pred_dst_dstm = fw['pred_dst_dstm']

        target_srcm_blur = blur_mask(target_srcm, blur_sigma)
        target_dstm_blur = blur_mask(target_dstm, blur_sigma)

        target_src_masked = target_src * target_srcm_blur
        target_dst_masked = target_dst * target_dstm_blur
        pred_src_src_masked = pred_src_src * target_srcm_blur
        pred_dst_dst_masked = pred_dst_dst * target_dstm_blur

        fs1, _ = dssim_filter_sizes(res)

        src_loss = 10 * dssim(target_src_masked, pred_src_src_masked,
                               max_val=1.0, filter_size=fs1).mean()
        src_loss = src_loss + 10 * F.mse_loss(target_src_masked, pred_src_src_masked)
        src_loss = src_loss + 10 * F.mse_loss(target_srcm, pred_src_srcm)

        dst_loss = 10 * dssim(target_dst_masked, pred_dst_dst_masked,
                               max_val=1.0, filter_size=fs1).mean()
        dst_loss = dst_loss + 10 * F.mse_loss(target_dst_masked, pred_dst_dst_masked)
        dst_loss = dst_loss + 10 * F.mse_loss(target_dstm, pred_dst_dstm)

        g_loss = src_loss + dst_loss
        return {
            'src_loss': src_loss.item(),
            'dst_loss': dst_loss.item(),
            'G_loss': g_loss,
            'D_gan_loss': None,
        }

    def get_preview_section_names(self) -> list[str]:
        return ['Quick96', 'Quick96 masked']

    _preview_section_name = "Quick96"

    def _predict_src(self, warped):
        code = self.inter(self.encoder(warped))
        return self.decoder_src(code)

    def _predict_dst(self, warped):
        code = self.inter(self.encoder(warped))
        return self.decoder_dst(code)
