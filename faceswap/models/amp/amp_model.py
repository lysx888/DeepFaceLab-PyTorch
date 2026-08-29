import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from faceswap.models.base_model import BaseModel
from faceswap.models.amp.amp_arch import AMPEncoder, AMPInter, AMPDecoder
from faceswap.models.saehd.losses import (
    dssim, total_variation_mse,
    apply_blur_out_mask, blur_mask, dssim_filter_sizes, gan_discriminator_loss,
    gan_discriminator_loss_dual,
)
from faceswap.models.saehd.optimizers import AdaBelief
from faceswap.models.saehd.discriminators import UNetPatchDiscriminator
from faceswap.shared.logger import get_logger

_logger = get_logger("amp_model")


class AMPModel(BaseModel):
    _model_prefix = "amp"
    _param_labels = {
        "resolution": "分辨率", "face_type": "人脸类型",
        "ae_dims": "编码维度", "inter_dims": "Inter维度",
        "e_dims": "编码器通道", "d_dims": "解码器通道",
        "d_mask_dims": "遮罩解码器通道", "morph_factor": "Morph因子",
        "uniform_yaw": "均匀yaw采样", "blur_out_mask": "模糊遮罩外区域",
        "lr_dropout": "学习率dropout", "random_warp": "随机变形",
        "gan_power": "GAN强度", "gan_patch_size": "GAN patch大小", "gan_dims": "GAN通道数",
        "ct_mode": "颜色迁移模式", "clipgrad": "梯度裁剪",
        "pretrain": "预训练",
    }

    def build(self) -> None:
        c = self.config
        self.resolution = c.resolution
        self.face_type = c.face_type
        self.ae_dims = c.ae_dims
        self.inter_dims = c.inter_dims
        self.e_dims = c.e_dims
        self.d_dims = c.d_dims
        self.d_mask_dims = c.d_mask_dims
        self.morph_factor = c.morph_factor
        self.blur_out_mask = getattr(c, 'blur_out_mask', False)
        self.pretrain = getattr(c, 'pretrain', False)
        self.gan_power = getattr(c, 'gan_power', 0.0) if not self.pretrain else 0.0

        inter_res = self.resolution // 32
        self.inter_res = inter_res

        self.encoder = AMPEncoder(in_ch=3, e_ch=self.e_dims,
                                   resolution=self.resolution, ae_dims=self.ae_dims)
        self.register_module('encoder', self.encoder)

        self.inter_src = AMPInter(ae_dims=self.ae_dims, inter_dims=self.inter_dims,
                                   inter_res=inter_res)
        self.inter_dst = AMPInter(ae_dims=self.ae_dims, inter_dims=self.inter_dims,
                                   inter_res=inter_res)
        self.register_module('inter_src', self.inter_src)
        self.register_module('inter_dst', self.inter_dst)

        self.decoder = AMPDecoder(inter_dims=self.inter_dims, d_ch=self.d_dims,
                                   d_mask_ch=self.d_mask_dims)
        self.register_module('decoder', self.decoder)

        if self.gan_power != 0:
            self.D_src = UNetPatchDiscriminator(
                patch_size=getattr(c, 'gan_patch_size', self.resolution // 8),
                in_ch=3, base_ch=getattr(c, 'gan_dims', 16))
            self.register_module('D_src', self.D_src)

    def build_optimizers(self) -> None:
        c = self.config
        lr = getattr(c, 'lr', 5e-5)
        clipnorm = 1.0 if getattr(c, 'clipgrad', False) else 0.0

        lr_dropout_cfg = getattr(c, 'lr_dropout', 'n')
        if lr_dropout_cfg in ('y', 'cpu'):
            lr_dropout = 0.3
            lr_cos = 500
        else:
            lr_dropout = 1.0
            lr_cos = 0

        gen_params = []
        _non_train_names = {'D_src', 'inter_src', 'inter_dst'}
        for name, module in self._modules_dict.items():
            if name in _non_train_names:
                continue
            gen_params.extend(p for p in module.parameters() if p.requires_grad)

        opt = AdaBelief(gen_params, lr=lr, lr_dropout=lr_dropout,
                        lr_cos=lr_cos, clipnorm=clipnorm)
        self.register_optimizer('src_dst_opt', opt)

        if hasattr(self, 'D_src'):
            d_params = list(self.D_src.parameters())
            d_opt = AdaBelief(d_params, lr=lr, lr_dropout=lr_dropout,
                              lr_cos=lr_cos, clipnorm=clipnorm)
            self.register_optimizer('D_src_opt', d_opt)

    def forward(self, warped_src, warped_dst, morph_value=1.0) -> dict:
        src_enc = self.encoder(warped_src)
        dst_enc = self.encoder(warped_dst)

        src_inter_src = self.inter_src(src_enc)
        src_inter_dst = self.inter_dst(src_enc)
        dst_inter_src = self.inter_src(dst_enc)
        dst_inter_dst = self.inter_dst(dst_enc)

        if self.training_mode:
            bs = src_inter_src.shape[0]
            inter_dims_bin = int(self.inter_dims * self.morph_factor)
            perm = torch.argsort(torch.rand(bs, self.inter_dims, device=src_inter_src.device), dim=1)
            rnd = torch.zeros(bs, self.inter_dims, device=src_inter_src.device)
            rnd.scatter_(1, perm[:, :inter_dims_bin], 1.0)
            rnd = rnd[:, :, None, None]
            src_code = src_inter_src * rnd + src_inter_dst * (1 - rnd)
            dst_code = dst_inter_dst
        else:
            inter_dims_slice = int(self.inter_dims * morph_value)
            src_dst_code = torch.cat([
                dst_inter_src[:, :inter_dims_slice],
                dst_inter_dst[:, inter_dims_slice:]
            ], dim=1)
            src_code = src_inter_src
            dst_code = dst_inter_dst

        pred_src_src, pred_src_srcm = self.decoder(src_code)
        pred_dst_dst, pred_dst_dstm = self.decoder(dst_code)

        if self.training_mode:
            pred_src_dst, pred_src_dstm = pred_src_src, pred_src_srcm
        else:
            pred_src_dst, pred_src_dstm = self.decoder(src_dst_code)

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
        target_srcm_em = batch_src.get('target_em_mask')
        if target_srcm_em is None:
            target_srcm_em = torch.zeros_like(target_srcm)
        target_dst = batch_dst['target_image']
        target_dstm = batch_dst['target_mask']
        target_dstm_em = batch_dst.get('target_em_mask')
        if target_dstm_em is None:
            target_dstm_em = torch.zeros_like(target_dstm)

        pred_src_src = fw['pred_src_src']
        pred_src_srcm = fw['pred_src_srcm']
        pred_dst_dst = fw['pred_dst_dst']
        pred_dst_dstm = fw['pred_dst_dstm']

        if self.blur_out_mask:
            target_src = apply_blur_out_mask(target_src, target_srcm, res)
            target_dst = apply_blur_out_mask(target_dst, target_dstm, res)

        target_srcm_blur = blur_mask(target_srcm, blur_sigma)
        target_srcm_anti_blur = 1.0 - target_srcm_blur

        target_dstm_blur = blur_mask(target_dstm, blur_sigma)

        target_src_masked = target_src * target_srcm_blur
        target_dst_masked = target_dst * target_dstm_blur
        pred_src_src_masked = pred_src_src * target_srcm_blur
        pred_dst_dst_masked = pred_dst_dst * target_dstm_blur

        target_src_anti_masked = target_src * target_srcm_anti_blur
        pred_src_src_anti_masked = pred_src_src * target_srcm_anti_blur
        target_dst_anti_masked = target_dst * (1 - target_dstm_blur)
        pred_dst_dst_anti_masked = pred_dst_dst * (1 - target_dstm_blur)

        fs1, fs2 = dssim_filter_sizes(res)

        src_loss = torch.tensor(0.0, device=target_src.device)
        src_loss = src_loss + 5 * dssim(target_src_masked, pred_src_src_masked,
                                         max_val=1.0, filter_size=fs1).mean()
        src_loss = src_loss + 5 * dssim(target_src_masked, pred_src_src_masked,
                                         max_val=1.0, filter_size=fs2).mean()
        src_loss = src_loss + 10 * F.mse_loss(target_src_masked, pred_src_src_masked)
        src_loss = src_loss + 300 * F.l1_loss(target_src * target_srcm_em,
                                               pred_src_src * target_srcm_em)
        src_loss = src_loss + 10 * F.mse_loss(target_srcm, pred_src_srcm)

        dst_loss = torch.tensor(0.0, device=target_dst.device)
        dst_loss = dst_loss + 5 * dssim(target_dst_masked, pred_dst_dst_masked,
                                         max_val=1.0, filter_size=fs1).mean()
        dst_loss = dst_loss + 5 * dssim(target_dst_masked, pred_dst_dst_masked,
                                         max_val=1.0, filter_size=fs2).mean()
        dst_loss = dst_loss + 10 * F.mse_loss(target_dst_masked, pred_dst_dst_masked)
        dst_loss = dst_loss + 300 * F.l1_loss(target_dst * target_dstm_em,
                                               pred_dst_dst * target_dstm_em)
        dst_loss = dst_loss + 10 * F.mse_loss(target_dstm, pred_dst_dstm)

        g_loss = src_loss + dst_loss
        g_loss = g_loss + 0.1 * F.mse_loss(pred_dst_dst_anti_masked, target_dst_anti_masked)
        g_loss = g_loss + 0.000001 * total_variation_mse(pred_dst_dst_anti_masked).mean()

        d_gan_loss = None
        if self.gan_power != 0:
            g_gan, d_gan_loss = gan_discriminator_loss_dual(
                self.D_src, pred_src_src_masked, target_src_masked,
                pred_dst_dst_masked, target_dst_masked)
            g_loss = g_loss + self.gan_power * g_gan

            g_loss = g_loss + 0.000001 * total_variation_mse(pred_src_src).mean()
            g_loss = g_loss + 0.02 * F.mse_loss(pred_src_src_anti_masked, target_src_anti_masked)

        return {
            'src_loss': src_loss.item(),
            'dst_loss': dst_loss.item(),
            'G_loss': g_loss,
            'D_gan_loss': d_gan_loss,
        }

    def get_preview_section_names(self) -> list[str]:
        return ['AMP morph 1.0', 'AMP morph list', 'AMP morph list masked']

    _preview_section_name = "AMP morph 1.0"

    def _preview_predict_src(self, warped: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._predict(warped)

    def _preview_predict_dst(self, warped: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            enc = self.encoder(warped)
            inter_dst = self.inter_dst(enc)
            return self.decoder(inter_dst)

    def _predict(self, warped, morph_value=1.0):
        with torch.no_grad():
            enc = self.encoder(warped)
            inter_src = self.inter_src(enc)
            inter_dst = self.inter_dst(enc)
            slice_n = int(self.inter_dims * morph_value)
            code = torch.cat([inter_src[:, :slice_n], inter_dst[:, slice_n:]], dim=1)
            return self.decoder(code)

    def generate_preview_data(self, src_dataset, dst_dataset,
                               src_indices: list[int],
                               dst_indices: list[int]) -> dict[str, list]:
        result = {name: [] for name in self.get_preview_section_names()}
        n = min(len(src_indices), len(dst_indices), 4)
        morph_values = [0.25, 0.50, 0.65, 0.75, 1.00]
        i = random.randint(0, n - 1)
        self._set_eval()
        with torch.no_grad():
            s_sample = src_dataset[src_indices[i]]
            d_sample = dst_dataset[dst_indices[i]]
            S = s_sample['target_image'].unsqueeze(0).to(self.device)
            D = d_sample['target_image'].unsqueeze(0).to(self.device)

            enc_s = self.encoder(S)
            SS, _ = self.decoder(self.inter_src(enc_s))

            enc_d = self.encoder(D)
            DD, DDM = self.decoder(self.inter_dst(enc_d))
            d_inter_src = self.inter_src(enc_d)
            d_inter_dst = self.inter_dst(enc_d)

            SDs = []
            SDMs = []
            for mv in morph_values:
                slice_n = int(self.inter_dims * mv)
                code = torch.cat([d_inter_src[:, :slice_n], d_inter_dst[:, slice_n:]], dim=1)
                sd, sdm = self.decoder(code)
                SDs.append(sd)
                SDMs.append(sdm)

            S0, SS0, D0, DD0 = S[0], SS[0], D[0], DD[0]
            DDM0 = DDM[0]
            SD0s = [sd[0] for sd in SDs]
            SDM0s = [sdm[0] for sdm in SDMs]

            result['AMP morph 1.0'].append([S0, D0, DD0 * DDM0])
            result['AMP morph 1.0'].append([SS0, DD0, SD0s[-1]])

            result['AMP morph list'].append([DD0, SD0s[0], SD0s[1]])
            result['AMP morph list'].append([SD0s[2], SD0s[3], SD0s[4]])

            result['AMP morph list masked'].append(
                [DD0, SD0s[0] * DDM0 * SDM0s[0], SD0s[1] * DDM0 * SDM0s[1]])
            result['AMP morph list masked'].append(
                [SD0s[2] * DDM0 * SDM0s[2], SD0s[3] * DDM0 * SDM0s[3], SD0s[4] * DDM0 * SDM0s[4]])
        self._set_train()
        return result

    def get_merge_face(self, warped_dst: torch.Tensor, morph_value: float = 1.0) -> tuple:
        self._set_eval()
        with torch.no_grad():
            enc = self.encoder(warped_dst)
            inter_src = self.inter_src(enc)
            inter_dst = self.inter_dst(enc)
            slice_n = int(self.inter_dims * morph_value)
            src_dst_code = torch.cat([inter_src[:, :slice_n], inter_dst[:, slice_n:]], dim=1)
            pred, pred_mask = self.decoder(src_dst_code)
            _, dst_mask = self.decoder(inter_dst)
        self._set_train()
        return pred, pred_mask, dst_mask
