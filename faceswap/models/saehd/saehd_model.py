import torch
import torch.nn as nn
import torch.nn.functional as F

from faceswap.models.base_model import BaseModel
from faceswap.models.saehd.saehd_arch import Encoder, Inter, Decoder, _init_weights
from faceswap.models.saehd.discriminators import CodeDiscriminator, UNetPatchDiscriminator
from faceswap.models.saehd.losses import dssim, style_loss, VGGFeatureExtractor
from faceswap.core.saehd_utils import gaussian_blur, total_variation_mse
from faceswap.shared.logger import get_logger

_logger = get_logger("saehd_model")


class SAEHDModel(BaseModel):
    _model_prefix = "SAEHD"
    _config_filename = "SAEHD_training_config.json"
    _param_labels = {
        "resolution": "分辨率",
        "face_type": "人脸类型",
        "batch_size": "批次大小",
        "optimizer": "优化器",
        "lr": "学习率",
        "lr_dropout": "学习率衰减",
        "lr_cos": "余弦退火周期",
        "random_warp": "随机变形",
        "random_src_flip": "源随机翻转",
        "random_dst_flip": "目标随机翻转",
        "random_hsv_power": "随机色调偏移",
        "ct_mode": "颜色迁移",
        "clipgrad": "梯度裁剪",
        "pretrain": "预训练模式",
        "amp_mode": "混合精度",
        "gradient_checkpointing": "梯度检查点",
        "target_iter": "目标迭代数",
        "backup_interval": "保存间隔",
        "archi": "AE架构",
        "ae_dims": "自编码器维度",
        "e_dims": "编码器维度",
        "d_dims": "解码器维度",
        "d_mask_dims": "遮罩解码器维度",
        "freeze_encoder": "冻结编码器",
        "freeze_inter": "冻结中间层(DF)",
        "freeze_inter_AB": "冻结Inter_AB(LIAE)",
        "freeze_inter_B": "冻结Inter_B(LIAE)",
        "freeze_decoder_mask": "冻结遮罩解码器",
        "masked_training": "遮罩训练",
        "eyes_mouth_prio": "眼嘴优先",
        "uniform_yaw": "均匀偏航采样",
        "true_face_power": "真实人脸强度",
        "gan_power": "GAN强度",
        "gan_patch_size": "GAN块大小",
        "gan_dims": "GAN维度",
        "vgg_perceptual_power": "VGG感知损失",
        "enable_torch_compile": "torch.compile加速",
        "use_ms_ssim": "多尺度SSIM",

        "adaptive_mask_dilation": "自适应Mask膨胀",
        "mask_dilation_sigma": "膨胀模糊sigma",
        "mask_dilation_radius": "膨胀半径",
        "ramp_start_ratio": "渐进GAN起始比",
        "smart_stop_enabled": "智能停止检测",
        "smart_stop_window": "收敛窗口",
        "smart_stop_threshold": "收敛阈值",
    }

    def __init__(self, config, model_dir, device):
        self.vgg_extractor = None
        self._lr_masks = None
        self._lr_dropout_rate = 1.0
        super().__init__(config, model_dir, device)

    def build(self) -> None:
        c = self.config
        opts = c.archi_opts

        has_d = 'd' in c.archi_opts
        divisor = 32 if has_d else 16
        if c.resolution < 64 or c.resolution % divisor != 0:
            raise ValueError(f"resolution={c.resolution} must be >= 64 and a multiple of {divisor}. "
                             f"Example valid values: 64, 128, 192, 256, 384, 512")

        self.encoder = Encoder(in_ch=3, e_ch=c.e_dims, opts=c.archi,
                               resolution=c.resolution)
        encoder_out_ch = self.encoder.get_out_ch()
        encoder_out_res = self.encoder.get_out_res(c.resolution)
        encoder_flat_ch = encoder_out_ch * (encoder_out_res ** 2)

        if c.archi_type == 'df':
            self.inter = Inter(in_ch=encoder_flat_ch, ae_ch=c.ae_dims,
                               ae_out_ch=c.ae_dims, resolution=c.resolution,
                               opts=c.archi)
            inter_out_ch = self.inter.get_out_ch()
            self.decoder_src = Decoder(in_ch=inter_out_ch, d_ch=c.d_dims,
                                       d_mask_ch=c.d_mask_dims,
                                       resolution=c.resolution, opts=c.archi)
            self.decoder_dst = Decoder(in_ch=inter_out_ch, d_ch=c.d_dims,
                                       d_mask_ch=c.d_mask_dims,
                                       resolution=c.resolution, opts=c.archi)
            self.inter_AB = None
            self.inter_B = None
            self.decoder = None
        else:
            self.inter_AB = Inter(in_ch=encoder_flat_ch, ae_ch=c.ae_dims,
                                  ae_out_ch=c.ae_dims * 2, resolution=c.resolution,
                                  opts=c.archi)
            self.inter_B = Inter(in_ch=encoder_flat_ch, ae_ch=c.ae_dims,
                                 ae_out_ch=c.ae_dims * 2, resolution=c.resolution,
                                 opts=c.archi)
            inter_out_ch = self.inter_AB.get_out_ch()
            inters_out_ch = inter_out_ch * 2
            self.decoder = Decoder(in_ch=inters_out_ch, d_ch=c.d_dims,
                                   d_mask_ch=c.d_mask_dims,
                                   resolution=c.resolution, opts=c.archi)
            self.inter = None
            self.decoder_src = None
            self.decoder_dst = None

        self.code_discriminator = None
        self.D_src = None

        if c.true_face_power != 0.0 and c.archi_type == 'df':
            code_res = self.inter.get_out_res()
            code_ch = self.inter.get_out_ch()
            self.code_discriminator = CodeDiscriminator(in_ch=code_ch, code_res=code_res)

        if c.gan_power != 0.0:
            self.D_src = UNetPatchDiscriminator(
                patch_size=c.gan_patch_size, in_ch=3, base_ch=c.gan_dims)

        self.register_module('encoder', self.encoder)
        if self.inter is not None:
            self.register_module('inter', self.inter)
        if self.inter_AB is not None:
            self.register_module('inter_AB', self.inter_AB)
        if self.inter_B is not None:
            self.register_module('inter_B', self.inter_B)
        if self.decoder is not None:
            self.register_module('decoder', self.decoder)
        if self.decoder_src is not None:
            self.register_module('decoder_src', self.decoder_src)
        if self.decoder_dst is not None:
            self.register_module('decoder_dst', self.decoder_dst)
        if self.code_discriminator is not None:
            self.register_module('code_discriminator', self.code_discriminator)
        if self.D_src is not None:
            self.register_module('D_src', self.D_src)

        if self.config.vgg_perceptual_power > 0.0:
            try:
                self.vgg_extractor = VGGFeatureExtractor(norm_mode='face').to(self.device)
            except Exception as e:
                _logger.warning(f"VGG16 init failed ({e}), perceptual loss disabled")

    def eval(self):
        for m in self._modules_dict.values():
            m.eval()
        return self

    def train(self, mode=True):
        for m in self._modules_dict.values():
            m.train(mode)
        return self

    def _apply_init(self) -> None:
        for module in self._modules_dict.values():
            module.apply(_init_weights)

    def apply_freeze(self) -> None:
        c = self.config

        if c.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if c.freeze_inter and self.inter is not None:
            for p in self.inter.parameters():
                p.requires_grad = False
        if c.freeze_inter_AB and self.inter_AB is not None:
            for p in self.inter_AB.parameters():
                p.requires_grad = False
        if c.freeze_inter_B and self.inter_B is not None:
            for p in self.inter_B.parameters():
                p.requires_grad = False
        if c.freeze_decoder_mask:
            for decoder in self._get_decoder_modules():
                for name, module in decoder.named_modules():
                    if 'upscalem' in name or 'out_convm' in name:
                        for p in module.parameters():
                            p.requires_grad = False

    def _get_decoder_modules(self) -> list:
        decoders = []
        if self.decoder_src is not None:
            decoders.append(self.decoder_src)
        if self.decoder_dst is not None:
            decoders.append(self.decoder_dst)
        if self.decoder is not None:
            decoders.append(self.decoder)
        return decoders

    def build_optimizers(self) -> None:
        c = self.config
        all_modules = []
        if not getattr(c, 'freeze_encoder', False):
            all_modules.append(self.encoder)
        if self.inter is not None and not getattr(c, 'freeze_inter', False):
            all_modules.append(self.inter)
        if self.inter_AB is not None and not getattr(c, 'freeze_inter_AB', False):
            all_modules.append(self.inter_AB)
        if self.inter_B is not None and not getattr(c, 'freeze_inter_B', False):
            all_modules.append(self.inter_B)
        if self.decoder is not None:
            all_modules.append(self.decoder)
        if self.decoder_src is not None:
            all_modules.append(self.decoder_src)
        if self.decoder_dst is not None:
            all_modules.append(self.decoder_dst)
        params = [p for m in all_modules for p in m.parameters() if p.requires_grad]

        lr = c.lr
        if c.optimizer == 'adabelief' and c.lr_dropout != 'n':
            _logger.warning("AdaBelief+LRD同时启用: DFL建议AB下无需LRD，但允许继续")
        self._lr_dropout_rate = 0.3 if (c.lr_dropout in ('y', 'cpu') and not c.pretrain) else 1.0

        OptimClass = torch.optim.Adam
        optim_kw = dict(lr=lr, betas=(0.9, 0.999), eps=1e-8)
        if c.optimizer == 'adamw':
            OptimClass = torch.optim.AdamW
            optim_kw['weight_decay'] = 0.0
        elif c.optimizer == 'adabelief':
            try:
                from adabelief_pytorch import AdaBelief as _AdaBelief
                OptimClass = lambda params, **kw: _AdaBelief(params, print_change_log=False, **kw)
            except ImportError:
                from faceswap.shared.adabelief import AdaBelief as _AdaBelief
                OptimClass = _AdaBelief
                _logger.info("Using built-in AdaBelief (adabelief-pytorch not installed)")
            optim_kw = dict(lr=lr, betas=(0.9, 0.999), eps=1e-8)
        elif c.optimizer == 'lion':
            try:
                from lion_pytorch import Lion
                OptimClass = Lion
                optim_kw = dict(lr=lr, betas=(0.9, 0.99))
            except ImportError:
                _logger.warning("Lion not installed, falling back to Adam")

        self.src_dst_opt = OptimClass(params, **optim_kw)
        self.register_optimizer('src_dst_opt', self.src_dst_opt)

        self._lr_masks = None

        if self.code_discriminator is not None:
            d_optim_kw = {k: v for k, v in optim_kw.items() if k != 'weight_decay'}
            self.D_code_opt = OptimClass(self.code_discriminator.parameters(), **d_optim_kw)
            self.register_optimizer('D_code_opt', self.D_code_opt)
        else:
            self.D_code_opt = None

        if self.D_src is not None:
            d_optim_kw = {k: v for k, v in optim_kw.items() if k != 'weight_decay'}
            self.D_src_opt = OptimClass(self.D_src.parameters(), **d_optim_kw)
            self.register_optimizer('D_src_opt', self.D_src_opt)
        else:
            self.D_src_opt = None

    def on_pretrain_override(self) -> None:
        c = self.config
        c.gan_power = 0.0
        c.random_warp = False
        c.random_hsv_power = 0.0
        c.vgg_perceptual_power = 0.0
        c.uniform_yaw = True
        c.lr_dropout = 'n'
        _logger.info("Pretrain mode: disabled gan/warp/hsv/vgg, enabled uniform_yaw")

    def forward(self, warped_src: torch.Tensor, warped_dst: torch.Tensor) -> dict:
        if self.config.archi_type == 'df':
            return self._forward_df(warped_src, warped_dst)
        return self._forward_liae(warped_src, warped_dst)

    def _forward_df(self, warped_src, warped_dst):
        src_code = self.inter(self.encoder(warped_src))
        dst_code = self.inter(self.encoder(warped_dst))
        pred_src_src, pred_src_srcm = self.decoder_src(src_code)
        pred_dst_dst, pred_dst_dstm = self.decoder_dst(dst_code)
        pred_src_dst, pred_src_dstm = self.decoder_src(dst_code)
        return {
            'src_code': src_code, 'dst_code': dst_code,
            'pred_src_src': pred_src_src, 'pred_src_srcm': pred_src_srcm,
            'pred_dst_dst': pred_dst_dst, 'pred_dst_dstm': pred_dst_dstm,
            'pred_src_dst': pred_src_dst, 'pred_src_dstm': pred_src_dstm,
        }

    def _forward_liae(self, warped_src, warped_dst):
        src_code = self.encoder(warped_src)
        src_inter_ab = self.inter_AB(src_code)
        src_code_cat = torch.cat([src_inter_ab, src_inter_ab], dim=1)
        dst_code = self.encoder(warped_dst)
        dst_inter_b = self.inter_B(dst_code)
        dst_inter_ab = self.inter_AB(dst_code)
        dst_code_cat = torch.cat([dst_inter_b, dst_inter_ab], dim=1)
        src_dst_code_cat = torch.cat([dst_inter_ab, dst_inter_ab], dim=1)
        pred_src_src, pred_src_srcm = self.decoder(src_code_cat)
        pred_dst_dst, pred_dst_dstm = self.decoder(dst_code_cat)
        pred_src_dst, pred_src_dstm = self.decoder(src_dst_code_cat)
        return {
            'src_code': src_code_cat, 'dst_code': dst_code_cat,
            'pred_src_src': pred_src_src, 'pred_src_srcm': pred_src_srcm,
            'pred_dst_dst': pred_dst_dst, 'pred_dst_dstm': pred_dst_dstm,
            'pred_src_dst': pred_src_dst, 'pred_src_dstm': pred_src_dstm,
        }

    def compute_loss(self, batch_src: dict, batch_dst: dict, fw: dict) -> dict:
        c = self.config
        target_srcm = batch_src['target_mask']
        target_dstm = batch_dst['target_mask']

        src_loss_vec, dst_loss_vec, extra_style_loss, extra_masked_gan_loss = self._compute_losses(
            batch_src['target_image'], batch_dst['target_image'],
            target_srcm, target_dstm,
            batch_src['target_em_mask'], batch_dst['target_em_mask'], fw)

        G_loss = src_loss_vec.mean() + dst_loss_vec.mean() + extra_style_loss + extra_masked_gan_loss

        D_code_loss = None
        if c.true_face_power != 0.0 and not c.pretrain and c.archi_type == 'df' and self.code_discriminator is not None:
            src_code_d = self.code_discriminator(fw['src_code'])
            dst_code_d = self.code_discriminator(fw['dst_code'])
            G_loss = G_loss + c.true_face_power * F.binary_cross_entropy_with_logits(
                src_code_d, torch.ones_like(src_code_d))
            D_code_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(dst_code_d, torch.ones_like(dst_code_d)) +
                F.binary_cross_entropy_with_logits(src_code_d.detach(), torch.zeros_like(src_code_d)))

        D_gan_loss = None
        effective_gan_power = c.gan_power
        if c.gan_power != 0.0 and self.D_src is not None:
            from faceswap.core.saehd_utils import compute_effective_gan_power
            iter_count = self._aux_state.get('iter_count', 0)
            effective_gan_power = compute_effective_gan_power(
                iter_count, getattr(c, 'target_iter', 0), c.gan_power,
                getattr(c, 'ramp_start_ratio', 0.2), getattr(c, 'pretrain', False))
            self._aux_state['progressive_gan_state'] = {
                'effective_gan_power': effective_gan_power,
                'ramp_progress': (effective_gan_power / c.gan_power) if c.gan_power > 0 else 0.0}
            if effective_gan_power > 0.0 and self.D_src is not None:
                pred_src_src = fw['pred_src_src']
                target_src = batch_src['target_image']
                target_src_masked_opt = target_src * target_srcm if c.masked_training else target_src
                pred_src_src_masked_opt = pred_src_src * target_srcm if c.masked_training else pred_src_src
                pred_d1, pred_d2 = self.D_src(pred_src_src_masked_opt)
                tgt_d1, tgt_d2 = self.D_src(target_src_masked_opt)
                D_gan_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(tgt_d1, torch.ones_like(tgt_d1)) +
                    F.binary_cross_entropy_with_logits(pred_d1.detach(), torch.zeros_like(pred_d1))
                ) + 0.5 * (
                    F.binary_cross_entropy_with_logits(tgt_d2, torch.ones_like(tgt_d2)) +
                    F.binary_cross_entropy_with_logits(pred_d2.detach(), torch.zeros_like(pred_d2))
                )
                G_loss = G_loss + effective_gan_power * (
                    F.binary_cross_entropy_with_logits(pred_d1, torch.ones_like(pred_d1)) +
                    F.binary_cross_entropy_with_logits(pred_d2, torch.ones_like(pred_d2)))

        return {
            'G_loss': G_loss,
            'D_code_loss': D_code_loss,
            'D_gan_loss': D_gan_loss,
            'src_loss': float(src_loss_vec.mean().detach()),
            'dst_loss': float(dst_loss_vec.mean().detach()),
        }

    def _compute_losses(self, target_src, target_dst, target_srcm, target_dstm,
                        target_srcm_em, target_dstm_em, fw):
        c = self.config
        resolution = c.resolution

        pred_src_src = fw['pred_src_src']
        pred_src_srcm = fw['pred_src_srcm']
        pred_dst_dst = fw['pred_dst_dst']
        pred_dst_dstm = fw['pred_dst_dstm']
        pred_src_dst = fw['pred_src_dst']
        pred_src_dstm = fw['pred_src_dstm']

        k_blur = max(1, resolution // 32)
        if c.adaptive_mask_dilation and c.masked_training:
            from faceswap.core.saehd_utils import adaptive_dilate_mask
            target_srcm_blur = adaptive_dilate_mask(target_srcm, sigma=c.mask_dilation_sigma, radius=c.mask_dilation_radius)
            target_dstm_blur = adaptive_dilate_mask(target_dstm, sigma=c.mask_dilation_sigma, radius=c.mask_dilation_radius)
        else:
            target_srcm_blur = gaussian_blur(target_srcm, k_blur)
            target_srcm_blur = torch.clamp(target_srcm_blur, 0.0, 0.5) * 2.0
            target_dstm_blur = gaussian_blur(target_dstm, k_blur)
            target_dstm_blur = torch.clamp(target_dstm_blur, 0.0, 0.5) * 2.0

        style_mask_blur = target_srcm_blur.detach()
        style_mask_anti_blur = 1.0 - style_mask_blur
        target_srcm_anti_blur = 1.0 - target_srcm_blur

        target_src_masked_opt = target_src * target_srcm_blur if c.masked_training else target_src
        target_dst_masked_opt = target_dst * target_dstm_blur if c.masked_training else target_dst
        pred_src_src_masked_opt = pred_src_src * target_srcm_blur if c.masked_training else pred_src_src
        pred_dst_dst_masked_opt = pred_dst_dst * target_dstm_blur if c.masked_training else pred_dst_dst

        fs1 = max(1, int(resolution / 11.6))
        fs2 = max(1, int(resolution / 23.2))

        use_ms = getattr(c, 'use_ms_ssim', False)
        try:
            if use_ms:
                from faceswap.models.saehd.losses import ms_ssim as _ms_ssim
                if resolution < 256:
                    src_loss = _ms_ssim(target_src_masked_opt, pred_src_src_masked_opt, fs1) * 10
                    dst_loss = _ms_ssim(target_dst_masked_opt, pred_dst_dst_masked_opt, fs1) * 10
                else:
                    src_loss = (_ms_ssim(target_src_masked_opt, pred_src_src_masked_opt, fs1) * 5 +
                                _ms_ssim(target_src_masked_opt, pred_src_src_masked_opt, fs2) * 5)
                    dst_loss = (_ms_ssim(target_dst_masked_opt, pred_dst_dst_masked_opt, fs1) * 5 +
                                _ms_ssim(target_dst_masked_opt, pred_dst_dst_masked_opt, fs2) * 5)
            else:
                raise RuntimeError("fallback")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if use_ms and not isinstance(e, RuntimeError) or (isinstance(e, RuntimeError) and str(e) != "fallback"):
                _logger.warning("MS-SSIM显存不足，已降级为单尺度DSSIM")
            if resolution < 256:
                src_loss = dssim(target_src_masked_opt, pred_src_src_masked_opt, fs1).mean(dim=[1, 2, 3]) * 10
                dst_loss = dssim(target_dst_masked_opt, pred_dst_dst_masked_opt, fs1).mean(dim=[1, 2, 3]) * 10
            else:
                src_loss = (dssim(target_src_masked_opt, pred_src_src_masked_opt, fs1).mean(dim=[1, 2, 3]) * 5 +
                            dssim(target_src_masked_opt, pred_src_src_masked_opt, fs2).mean(dim=[1, 2, 3]) * 5)
                dst_loss = (dssim(target_dst_masked_opt, pred_dst_dst_masked_opt, fs1).mean(dim=[1, 2, 3]) * 5 +
                            dssim(target_dst_masked_opt, pred_dst_dst_masked_opt, fs2).mean(dim=[1, 2, 3]) * 5)

        src_loss = src_loss + ((target_src_masked_opt - pred_src_src_masked_opt) ** 2).mean(dim=[1, 2, 3]) * 10
        dst_loss = dst_loss + ((target_dst_masked_opt - pred_dst_dst_masked_opt) ** 2).mean(dim=[1, 2, 3]) * 10

        if c.eyes_mouth_prio:
            src_loss = src_loss + (target_src * target_srcm_em - pred_src_src * target_srcm_em).abs().mean(dim=[1, 2, 3]) * 300
            dst_loss = dst_loss + (target_dst * target_dstm_em - pred_dst_dst * target_dstm_em).abs().mean(dim=[1, 2, 3]) * 300

        mask_loss_weight = 10.0 * (resolution / 256.0) ** 0.5
        src_loss = src_loss + ((target_srcm - pred_src_srcm) ** 2).mean(dim=[1, 2, 3]) * mask_loss_weight
        dst_loss = dst_loss + ((target_dstm - pred_dst_dstm) ** 2).mean(dim=[1, 2, 3]) * mask_loss_weight


        extra_style_loss = torch.tensor(0.0, device=self.device)

        if c.vgg_perceptual_power > 0.0 and self.vgg_extractor is not None:
            vgg_w = c.vgg_perceptual_power / 50.0
            with torch.no_grad():
                t_src_vgg = self.vgg_extractor(target_src)
                t_dst_vgg = self.vgg_extractor(target_dst)
            p_src_vgg = self.vgg_extractor(pred_src_src)
            p_dst_vgg = self.vgg_extractor(pred_dst_dst)
            src_loss = src_loss + vgg_w * sum(F.l1_loss(pf, tf) for pf, tf in zip(p_src_vgg, t_src_vgg))
            dst_loss = dst_loss + vgg_w * sum(F.l1_loss(pf, tf) for pf, tf in zip(p_dst_vgg, t_dst_vgg))

        extra_masked_gan_loss = torch.tensor(0.0, device=self.device)
        effective_gp = self._aux_state.get('progressive_gan_state', {}).get('effective_gan_power', c.gan_power)
        if c.masked_training and effective_gp != 0.0:
            target_src_anti_masked = target_src * target_srcm_anti_blur
            pred_src_src_anti_masked = pred_src_src * target_srcm_anti_blur
            extra_masked_gan_loss = extra_masked_gan_loss + 0.000001 * total_variation_mse(pred_src_src)
            extra_masked_gan_loss = extra_masked_gan_loss + 0.02 * ((pred_src_src_anti_masked - target_src_anti_masked) ** 2).mean()

        return src_loss, dst_loss, extra_style_loss, extra_masked_gan_loss

    def get_preview_section_names(self) -> list[str]:
        return ["原图预览", "遮罩下", "原始输入", "合并预览"]

    @torch.no_grad()
    def generate_preview_data(self, src_dataset, dst_dataset,
                              src_indices, dst_indices):
        import cv2
        import numpy as np
        c = self.config
        res = c.resolution
        sections = {name: [] for name in self.get_preview_section_names()}

        for si, di in zip(src_indices, dst_indices):
            s_img = cv2.imread(str(src_dataset.image_paths[si]))
            d_img = cv2.imread(str(dst_dataset.image_paths[di]))
            if s_img is None or d_img is None:
                continue

            has_weight_nan = False
            for m in self._modules_dict.values():
                for p in m.parameters():
                    if p.isnan().any() or p.isinf().any():
                        has_weight_nan = True
                        break
                if has_weight_nan:
                    break
            if has_weight_nan:
                _plog = get_logger("saehd_preview")
                _plog.warning("Preview skipped: weight NaN/Inf detected")
                break

            s_img = cv2.resize(s_img, (res, res))
            d_img = cv2.resize(d_img, (res, res))
            from faceswap.shared.image_utils import bgr_to_rgb, rgb_to_bgr
            s_img_rgb = bgr_to_rgb(s_img)
            d_img_rgb = bgr_to_rgb(d_img)
            s_t = torch.from_numpy(s_img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            d_t = torch.from_numpy(d_img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

            fw = self.forward(s_t, d_t)

            def _to_img(t):
                arr = t[0].cpu().float().numpy().transpose(1, 2, 0)
                n_nan = np.isnan(arr).sum()
                n_inf = np.isinf(arr).sum()
                if n_nan > 0 or n_inf > 0:
                    _plog = get_logger("saehd_preview")
                    _plog.warning(f"Preview output has {n_nan} NaN, {n_inf} Inf pixels")
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

            S = rgb_to_bgr(_to_img(s_t))
            D = rgb_to_bgr(_to_img(d_t))
            SS = rgb_to_bgr(_to_img(fw['pred_src_src']))
            DD = rgb_to_bgr(_to_img(fw['pred_dst_dst']))
            SD = rgb_to_bgr(_to_img(fw['pred_src_dst']))
            from faceswap.shared.logger import get_logger
            _plog = get_logger("saehd_preview")
            _plog.info(f"S[m{S.mean():.0f},{S.max()}] SS[m{SS.mean():.0f},{SS.max()}] D[m{D.mean():.0f},{D.max()}] DD[m{DD.mean():.0f},{DD.max()}] SD[m{SD.mean():.0f},{SD.max()}]")
            _plog.info(f"raw_pred[{fw['pred_src_src'].min():.4f},{fw['pred_src_src'].max():.4f}] mean={fw['pred_src_src'].mean():.4f}")
            SSM = np.nan_to_num(fw['pred_src_srcm'][0, 0].cpu().float().numpy(), nan=0.0)
            DDM = np.nan_to_num(fw['pred_dst_dstm'][0, 0].cpu().float().numpy(), nan=0.0)
            SDM = np.nan_to_num(fw['pred_src_dstm'][0, 0].cpu().float().numpy(), nan=0.0)
            tgt_srcm = src_dataset.get_preview_mask(si, (res, res))
            tgt_dstm = dst_dataset.get_preview_mask(di, (res, res))
            tgt_srcm = np.nan_to_num(tgt_srcm, nan=0.0)
            tgt_dstm = np.nan_to_num(tgt_dstm, nan=0.0)

            WS = src_dataset[si]['warped_image'].permute(1, 2, 0).cpu().float().numpy()
            WD = dst_dataset[di]['warped_image'].permute(1, 2, 0).cpu().float().numpy()
            WS_u8 = rgb_to_bgr(np.clip(WS * 255.0, 0, 255).astype(np.uint8))
            WD_u8 = rgb_to_bgr(np.clip(WD * 255.0, 0, 255).astype(np.uint8))

            DDM3 = np.repeat(DDM[..., None], 3, axis=-1)
            if c.face_type != 'head':
                dst_merge_mask = np.repeat(tgt_dstm[..., None], 3, axis=-1) * DDM3
            else:
                dst_merge_mask = np.repeat(tgt_dstm[..., None], 3, axis=-1)
            try:
                from faceswap.core.color_transfer import reinhard_color_transfer
                sd_rct = reinhard_color_transfer(SD, D,
                                                 target_mask=dst_merge_mask,
                                                 source_mask=dst_merge_mask)
            except Exception:
                sd_rct = SD

            from faceswap.core.preview_renderer import PreviewData, render_all_sections
            pd = PreviewData(
                S=S, D=D, SS=SS, DD=DD, SD=SD,
                tgt_srcm=tgt_srcm, tgt_dstm=tgt_dstm,
                SSM=SSM, DDM=DDM, SDM=SDM,
                WS=WS_u8, WD=WD_u8,
                face_type=c.face_type,
                sd_rct=sd_rct,
            )
            rendered = render_all_sections(pd)
            for name in sections:
                sections[name].append(rendered[name])

        return sections

    @torch.no_grad()
    def merge(self, warped_dst):
        import numpy as np
        c = self.config
        x = torch.from_numpy(warped_dst).float().to(self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if c.archi_type == 'df':
            dst_code = self.inter(self.encoder(x))
            pred_src_dst, pred_src_dstm = self.decoder_src(dst_code)
            _, pred_dst_dstm = self.decoder_dst(dst_code)
        else:
            dst_code = self.encoder(x)
            dst_inter_b = self.inter_B(dst_code)
            dst_inter_ab = self.inter_AB(dst_code)
            dst_code_cat = torch.cat([dst_inter_b, dst_inter_ab], dim=1)
            src_dst_code_cat = torch.cat([dst_inter_ab, dst_inter_ab], dim=1)
            pred_src_dst, pred_src_dstm = self.decoder(src_dst_code_cat)
            _, pred_dst_dstm = self.decoder(dst_code_cat)
        return (
            pred_src_dst.detach().cpu().numpy(),
            pred_dst_dstm.detach().cpu().numpy(),
            pred_src_dstm.detach().cpu().numpy(),
        )

    def get_module_info(self) -> list[tuple[str, int, str]]:
        info = []
        for name, module in self._modules_dict.items():
            n_params = sum(p.numel() for p in module.parameters())
            frozen = not any(p.requires_grad for p in module.parameters())
            status = 'frozen' if frozen else 'active'
            info.append((name, n_params, status))
        return info
