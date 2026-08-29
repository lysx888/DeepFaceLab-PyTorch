import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from faceswap.models.base_model import BaseModel
from faceswap.models.saehd.saehd_arch import Encoder, Inter, Decoder
from faceswap.models.saehd.losses import (
    dssim, style_loss, total_variation_mse, dloss,
    apply_blur_out_mask, blur_mask, dssim_filter_sizes, gan_discriminator_loss,
)
from faceswap.models.saehd.optimizers import AdaBelief, RMSprop
from faceswap.models.saehd.discriminators import CodeDiscriminator, UNetPatchDiscriminator
from faceswap.shared.logger import get_logger

_logger = get_logger("saehd_model")


class SAEHDModel(BaseModel):
    _model_prefix = "saehd"
    _param_labels = {
        "resolution": "分辨率", "face_type": "人脸类型", "archi": "架构",
        "ae_dims": "编码维度", "e_dims": "编码器通道", "d_dims": "解码器通道",
        "d_mask_dims": "遮罩解码器通道", "masked_training": "遮罩训练",
        "eyes_mouth_prio": "眼嘴优先", "uniform_yaw": "均匀yaw采样",
        "blur_out_mask": "模糊遮罩外区域", "adabelief": "AdaBelief优化器",
        "lr_dropout": "学习率dropout", "random_warp": "随机变形",
        "random_hsv_power": "随机HSV强度", "true_face_power": "真脸判别强度",
        "face_style_power": "人脸风格强度", "bg_style_power": "背景风格强度",
        "gan_power": "GAN强度", "gan_patch_size": "GAN patch大小", "gan_dims": "GAN通道数",
        "ct_mode": "颜色迁移模式", "clipgrad": "梯度裁剪", "pretrain": "预训练",
        "multiscale_loss_power": "多尺度损失强度",
        "visibility_loss_power": "可见性损失强度",
    }

    def build(self) -> None:
        c = self.config
        self.resolution = c.resolution
        self.face_type = c.face_type
        self.pretrain = c.pretrain

        archi_split = c.archi.split('-')
        if len(archi_split) == 2:
            archi_type, archi_opts = archi_split
        else:
            archi_type, archi_opts = archi_split[0], ''
        self._is_df = 'df' in archi_type
        self._archi_opts = archi_opts

        if ('d' in archi_opts or 't' in archi_opts) and self.resolution % 32 != 0:
            raise ValueError(
                f"resolution={self.resolution} 不是32的倍数。"
                f"archi含'd'或't'后缀时，分辨率必须是32的倍数，否则维度不匹配。"
            )

        self.ae_dims = c.ae_dims
        self.e_dims = c.e_dims
        self.d_dims = c.d_dims
        self.d_mask_dims = c.d_mask_dims
        self.masked_training = c.masked_training
        self.eyes_mouth_prio = c.eyes_mouth_prio
        self.blur_out_mask = getattr(c, 'blur_out_mask', False)
        self.multiscale_loss_power = getattr(c, 'multiscale_loss_power', 0.0)
        self.visibility_loss_power = getattr(c, 'visibility_loss_power', 0.0)

        self.face_style_power = getattr(c, 'face_style_power', 0.0)
        self.bg_style_power = getattr(c, 'bg_style_power', 0.0)
        self.gan_power = getattr(c, 'gan_power', 0.0) if not self.pretrain else 0.0
        self.true_face_power = getattr(c, 'true_face_power', 0.0) if (self._is_df and not self.pretrain) else 0.0
        if self.pretrain:
            self.face_style_power = 0.0
            self.bg_style_power = 0.0

        opts = self._archi_opts
        self.encoder = Encoder(in_ch=3, e_ch=self.e_dims, resolution=self.resolution, opts=opts)
        self.register_module('encoder', self.encoder)

        encoder_out_flat = self.encoder.get_out_ch() * self.encoder.get_out_res(self.resolution) ** 2

        if self._is_df:
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
        else:
            self.inter_AB = Inter(in_ch=encoder_out_flat, ae_ch=self.ae_dims,
                                  ae_out_ch=self.ae_dims * 2, resolution=self.resolution, opts=opts)
            self.inter_B = Inter(in_ch=encoder_out_flat, ae_ch=self.ae_dims,
                                 ae_out_ch=self.ae_dims * 2, resolution=self.resolution, opts=opts)
            self.register_module('inter_AB', self.inter_AB)
            self.register_module('inter_B', self.inter_B)
            inter_out_ch = self.inter_AB.get_out_ch() * 2
            self.decoder = Decoder(in_ch=inter_out_ch, d_ch=self.d_dims,
                                   d_mask_ch=self.d_mask_dims, opts=opts)
            self.register_module('decoder', self.decoder)

        if self.true_face_power != 0 and self._is_df:
            code_res = self.inter.get_out_res()
            self.code_discriminator = CodeDiscriminator(self.ae_dims, code_res)
            self.register_module('code_discriminator', self.code_discriminator)

        if self.gan_power != 0:
            self.D_src = UNetPatchDiscriminator(
                patch_size=getattr(c, 'gan_patch_size', self.resolution // 8),
                in_ch=3, base_ch=getattr(c, 'gan_dims', 16))
            self.register_module('D_src', self.D_src)

    def build_optimizers(self) -> None:
        c = self.config
        lr = getattr(c, 'lr', 5e-5)
        clipnorm = 0.0

        lr_dropout_cfg = getattr(c, 'lr_dropout', 'n')
        if lr_dropout_cfg in ('y', 'cpu') and not self.pretrain:
            lr_dropout = 0.3
            lr_cos = 500
        else:
            lr_dropout = 1.0
            lr_cos = 0

        _disc_names = {'code_discriminator', 'D_src'}
        gen_params = []
        for name, module in self._modules_dict.items():
            if name in _disc_names:
                continue
            gen_params.extend(p for p in module.parameters() if p.requires_grad)
        opt_kwargs = dict(lr=lr, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm)

        opt_name = str(getattr(c, 'adabelief', 'adabelief')).lower()
        if opt_name in ('adabelief', 'true', '1', 'yes'):
            optimizer = AdaBelief(gen_params, **opt_kwargs)
        else:
            optimizer = RMSprop(gen_params, **opt_kwargs)
        self.register_optimizer('src_dst_opt', optimizer)

        d_params = []
        if hasattr(self, 'code_discriminator'):
            d_params += list(self.code_discriminator.parameters())
        if hasattr(self, 'D_src'):
            d_params += list(self.D_src.parameters())
        if d_params:
            if opt_name in ('adabelief', 'true', '1', 'yes'):
                d_opt = AdaBelief(d_params, **opt_kwargs)
            else:
                d_opt = RMSprop(d_params, **opt_kwargs)
            self.register_optimizer('D_src_opt', d_opt)

    def apply_freeze(self) -> None:
        c = self.config
        random_warp = getattr(c, 'random_warp', True)
        if not self._is_df and not random_warp:
            for p in self.inter_AB.parameters():
                p.requires_grad = False
            _logger.info("LIAE + random_warp=False: inter_AB frozen (半冻结微调模式)")

    def try_load(self) -> None:
        config_path = self.model_dir / "training_config.json"
        saved_pretrain = None
        if config_path.exists():
            try:
                import json
                saved_cfg = json.loads(config_path.read_text(encoding='utf-8'))
                saved_pretrain = saved_cfg.get('pretrain', None)
            except Exception:
                pass

        super().try_load()

        if saved_pretrain is True and not self.pretrain:
            self._reinit_inter_on_pretrain_disable()

    def _reinit_inter_on_pretrain_disable(self) -> None:
        import torch.nn.init as init
        targets = [self.inter] if self._is_df else [self.inter_AB, self.inter_B]
        for module in targets:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    init.kaiming_uniform_(m.weight, a=0, nonlinearity='leaky_relu')
                    if m.bias is not None:
                        init.zeros_(m.bias)
        names = "inter" if self._is_df else "inter_AB + inter_B"
        _logger.info(f"Pretrain disabled: reinitialized {names}")

    def _multiscale_loss(self, target: torch.Tensor, pred: torch.Tensor,
                         res: int) -> torch.Tensor:
        if self.multiscale_loss_power <= 0:
            return torch.tensor(0.0, device=target.device)
        pool_k = max(1, res // 64)
        if pool_k <= 1:
            return torch.tensor(0.0, device=target.device)
        low_t = F.avg_pool2d(target, kernel_size=pool_k)
        low_p = F.avg_pool2d(pred, kernel_size=pool_k)
        return self.multiscale_loss_power * F.mse_loss(low_t, low_p)

    def forward(self, warped_src, warped_dst) -> dict:
        need_style_grad = self.face_style_power != 0 and not self.pretrain

        if self._is_df:
            src_code = self.inter(self.encoder(warped_src))
            dst_code = self.inter(self.encoder(warped_dst))
            pred_src_src, pred_src_srcm = self.decoder_src(src_code)
            pred_dst_dst, pred_dst_dstm = self.decoder_dst(dst_code)
            pred_src_dst, pred_src_dstm = self.decoder_src(dst_code)

            if need_style_grad:
                pred_src_dst_for_style, _ = self.decoder_src(dst_code.detach())
            else:
                pred_src_dst_for_style = pred_src_dst.detach()
        else:
            src_enc = self.encoder(warped_src)
            src_inter_AB = self.inter_AB(src_enc)
            src_code = torch.cat([src_inter_AB, src_inter_AB], dim=1)

            dst_enc = self.encoder(warped_dst)
            dst_inter_B = self.inter_B(dst_enc)
            dst_inter_AB = self.inter_AB(dst_enc)
            dst_code = torch.cat([dst_inter_B, dst_inter_AB], dim=1)
            src_dst_code = torch.cat([dst_inter_AB, dst_inter_AB], dim=1)

            pred_src_src, pred_src_srcm = self.decoder(src_code)
            pred_dst_dst, pred_dst_dstm = self.decoder(dst_code)
            pred_src_dst, pred_src_dstm = self.decoder(src_dst_code)

            if need_style_grad:
                pred_src_dst_for_style, _ = self.decoder(src_dst_code.detach())
            else:
                pred_src_dst_for_style = pred_src_dst.detach()

        return {
            'pred_src_src': pred_src_src, 'pred_src_srcm': pred_src_srcm,
            'pred_dst_dst': pred_dst_dst, 'pred_dst_dstm': pred_dst_dstm,
            'pred_src_dst': pred_src_dst, 'pred_src_dstm': pred_src_dstm,
            'pred_src_dst_for_style': pred_src_dst_for_style,
            'src_code': src_code, 'dst_code': dst_code,
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
        target_src_vis = batch_src.get('target_vis_mask')
        target_dst_vis = batch_dst.get('target_vis_mask')

        pred_src_src = fw['pred_src_src']
        pred_src_srcm = fw['pred_src_srcm']
        pred_dst_dst = fw['pred_dst_dst']
        pred_dst_dstm = fw['pred_dst_dstm']
        pred_src_dst = fw['pred_src_dst']
        pred_src_dstm = fw['pred_src_dstm']
        pred_src_dst_for_style = fw['pred_src_dst_for_style']

        if self.blur_out_mask:
            target_src = apply_blur_out_mask(target_src, target_srcm, res)
            target_dst = apply_blur_out_mask(target_dst, target_dstm, res)

        target_srcm_blur = blur_mask(target_srcm, blur_sigma)
        target_srcm_anti_blur = 1.0 - target_srcm_blur

        target_dstm_blur = blur_mask(target_dstm, blur_sigma)

        style_mask_blur = torch.clamp(target_srcm_blur, 0, 1.0).detach()
        style_mask_anti_blur = 1.0 - style_mask_blur

        target_dst_masked = target_dst * target_dstm_blur

        if self.masked_training:
            target_src_masked_opt = target_src * target_srcm_blur
            target_dst_masked_opt = target_dst_masked
            pred_src_src_masked_opt = pred_src_src * target_srcm_blur
            pred_dst_dst_masked_opt = pred_dst_dst * target_dstm_blur
        else:
            target_src_masked_opt = target_src
            target_dst_masked_opt = target_dst
            pred_src_src_masked_opt = pred_src_src
            pred_dst_dst_masked_opt = pred_dst_dst

        target_src_anti_masked = target_src * target_srcm_anti_blur
        pred_src_src_anti_masked = pred_src_src * target_srcm_anti_blur

        fs1, fs2 = dssim_filter_sizes(res)

        src_loss = torch.tensor(0.0, device=target_src.device)
        if res < 256:
            src_loss = src_loss + 10 * dssim(target_src_masked_opt, pred_src_src_masked_opt,
                                             max_val=1.0, filter_size=fs1).mean()
        else:
            src_loss = src_loss + 5 * dssim(target_src_masked_opt, pred_src_src_masked_opt,
                                            max_val=1.0, filter_size=fs1).mean()
            src_loss = src_loss + 5 * dssim(target_src_masked_opt, pred_src_src_masked_opt,
                                            max_val=1.0, filter_size=fs2).mean()
        src_loss = src_loss + 10 * F.mse_loss(target_src_masked_opt, pred_src_src_masked_opt)

        src_loss = src_loss + self._multiscale_loss(target_src_masked_opt, pred_src_src_masked_opt, res)

        if self.visibility_loss_power > 0 and target_src_vis is not None:
            src_vis_diff = (target_src_masked_opt - pred_src_src_masked_opt) ** 2
            src_loss = src_loss + self.visibility_loss_power * (src_vis_diff * target_src_vis).mean()

        if self.eyes_mouth_prio:
            src_loss = src_loss + 300 * F.l1_loss(target_src * target_srcm_em,
                                                   pred_src_src * target_srcm_em)

        src_loss = src_loss + 10 * F.mse_loss(target_srcm, pred_src_srcm)

        face_style_power = self.face_style_power / 100.0
        if face_style_power != 0 and not self.pretrain:
            src_loss = src_loss + style_loss(
                pred_src_dst_for_style * pred_src_dstm.detach(),
                (pred_dst_dst * pred_dst_dstm).detach(),
                gaussian_blur_radius=res // 8,
                loss_weight=10000 * face_style_power
            ).mean()

        bg_style_power = self.bg_style_power / 100.0
        if bg_style_power != 0 and not self.pretrain:
            target_dst_style_anti_masked = target_dst * style_mask_anti_blur
            psd_style_anti_masked = pred_src_dst * style_mask_anti_blur
            src_loss = src_loss + 10 * bg_style_power * dssim(
                psd_style_anti_masked, target_dst_style_anti_masked,
                max_val=1.0, filter_size=fs1).mean()
            src_loss = src_loss + 10 * bg_style_power * F.mse_loss(
                psd_style_anti_masked, target_dst_style_anti_masked)

        dst_loss = torch.tensor(0.0, device=target_dst.device)
        if res < 256:
            dst_loss = dst_loss + 10 * dssim(target_dst_masked_opt, pred_dst_dst_masked_opt,
                                             max_val=1.0, filter_size=fs1).mean()
        else:
            dst_loss = dst_loss + 5 * dssim(target_dst_masked_opt, pred_dst_dst_masked_opt,
                                            max_val=1.0, filter_size=fs1).mean()
            dst_loss = dst_loss + 5 * dssim(target_dst_masked_opt, pred_dst_dst_masked_opt,
                                            max_val=1.0, filter_size=fs2).mean()
        dst_loss = dst_loss + 10 * F.mse_loss(target_dst_masked_opt, pred_dst_dst_masked_opt)

        dst_loss = dst_loss + self._multiscale_loss(target_dst_masked_opt, pred_dst_dst_masked_opt, res)

        if self.visibility_loss_power > 0 and target_dst_vis is not None:
            dst_vis_diff = (target_dst_masked_opt - pred_dst_dst_masked_opt) ** 2
            dst_loss = dst_loss + self.visibility_loss_power * (dst_vis_diff * target_dst_vis).mean()

        if self.eyes_mouth_prio:
            dst_loss = dst_loss + 300 * F.l1_loss(target_dst * target_dstm_em,
                                                   pred_dst_dst * target_dstm_em)

        dst_loss = dst_loss + 10 * F.mse_loss(target_dstm, pred_dst_dstm)

        g_loss = src_loss + dst_loss
        d_gan_loss = None

        if self.true_face_power != 0 and self._is_df:
            src_code = fw['src_code']
            dst_code = fw['dst_code']
            src_code_d = self.code_discriminator(src_code)
            dst_code_d = self.code_discriminator(dst_code)
            ones = torch.ones_like(src_code_d)
            zeros = torch.zeros_like(src_code_d)
            g_loss = g_loss + self.true_face_power * dloss(ones, src_code_d)
            if not self.pretrain:
                src_code_d_d = self.code_discriminator(src_code.detach())
                dst_code_d_d = self.code_discriminator(dst_code.detach())
                d_gan_loss = dloss(torch.ones_like(dst_code_d_d), dst_code_d_d) + dloss(zeros, src_code_d_d)
                d_gan_loss = d_gan_loss * 0.5

        if self.gan_power != 0:
            g_gan, d_loss = gan_discriminator_loss(self.D_src, pred_src_src_masked_opt, target_src_masked_opt)
            g_loss = g_loss + self.gan_power * g_gan
            d_gan_loss = d_loss if d_gan_loss is None else d_gan_loss + d_loss

            if self.masked_training:
                g_loss = g_loss + 0.000001 * total_variation_mse(pred_src_src).mean()
                g_loss = g_loss + 0.02 * F.mse_loss(pred_src_src_anti_masked, target_src_anti_masked)

        return {
            'src_loss': src_loss.item(),
            'dst_loss': dst_loss.item(),
            'G_loss': g_loss,
            'D_gan_loss': d_gan_loss,
        }

    def get_preview_section_names(self) -> list[str]:
        if self.resolution <= 256:
            return ['SAEHD', 'SAEHD masked']
        else:
            return ['SAEHD src-src', 'SAEHD dst-dst', 'SAEHD pred']

    _preview_section_name = "SAEHD"

    def _append_preview(self, result: dict[str, list], S, SS, D, DD, SD, Sm, Dm, DDm, SDm) -> None:
        if self.resolution <= 256:
            result['SAEHD'].append([S, SS, D, DD, SD])
            result['SAEHD masked'].append([S * Sm, SS, D * Dm, DD * DDm, SD * SDm])
        else:
            result['SAEHD src-src'].append([S, SS])
            result['SAEHD dst-dst'].append([D, DD])
            result['SAEHD pred'].append([D, SD])

    def _predict_src(self, warped):
        if self._is_df:
            code = self.inter(self.encoder(warped))
            return self.decoder_src(code)
        else:
            enc = self.encoder(warped)
            inter_AB = self.inter_AB(enc)
            code = torch.cat([inter_AB, inter_AB], dim=1)
            return self.decoder(code)

    def _predict_dst(self, warped):
        if self._is_df:
            code = self.inter(self.encoder(warped))
            return self.decoder_dst(code)
        else:
            enc = self.encoder(warped)
            inter_B = self.inter_B(enc)
            inter_AB = self.inter_AB(enc)
            code = torch.cat([inter_B, inter_AB], dim=1)
            return self.decoder(code)
