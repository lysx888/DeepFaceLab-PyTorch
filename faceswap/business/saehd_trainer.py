import json
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Callable

from faceswap.business.base_trainer import BaseTrainer
from faceswap.business.saehd_dataset import SAEHDDataset
from faceswap.models.saehd.saehd_model import SAEHDModel
from faceswap.shared.logger import get_logger

_logger = get_logger("saehd_trainer")


class TrainingConfig:
    _VALID_FACE_TYPES = ('h', 'mf', 'f', 'wf', 'head')
    _VALID_ARCHI_TYPES = ('df', 'liae')
    _VALID_ARCHI_OPTS = ('', 'ud', 'udt', 'd', 'dt')

    def __init__(self, **kwargs):
        self.face_type = kwargs.get('face_type', 'wf')
        if self.face_type not in self._VALID_FACE_TYPES:
            raise ValueError(f"face_type='{self.face_type}' 无效，可选: {self._VALID_FACE_TYPES}")
        self.resolution = kwargs.get('resolution', 0)
        if self.resolution <= 0:
            ft_res_map = {'h': 64, 'mf': 128, 'f': 128, 'wf': 256, 'head': 384}
            self.resolution = ft_res_map.get(self.face_type, 128)
        self.archi = kwargs.get('archi', 'liae-ud')
        archi_type = self.archi.split('-')[0]
        archi_opts = self.archi.split('-')[1] if '-' in self.archi else ''
        if archi_type not in self._VALID_ARCHI_TYPES:
            raise ValueError(f"archi类型'{archi_type}'无效，可选: {self._VALID_ARCHI_TYPES}")
        if archi_opts not in self._VALID_ARCHI_OPTS:
            raise ValueError(f"archi选项'{archi_opts}'无效，可选: {self._VALID_ARCHI_OPTS}")
        self.ae_dims = max(16, min(kwargs.get('ae_dims', 256), 1024))
        self.e_dims = max(8, min(kwargs.get('e_dims', 64), 256))
        self.d_dims = max(8, min(kwargs.get('d_dims', 64), 256))
        self.d_mask_dims = max(8, min(kwargs.get('d_mask_dims', 22), 128))
        self.batch_size = max(1, kwargs.get('batch_size', 8))
        self.masked_training = kwargs.get('masked_training', True)
        self.eyes_mouth_prio = kwargs.get('eyes_mouth_prio', False)
        self.uniform_yaw = kwargs.get('uniform_yaw', False)
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.lr = max(1e-7, min(kwargs.get('lr', 5e-5), 1e-2))
        self.lr_dropout = kwargs.get('lr_dropout', 'n')
        self.lr_cos = max(0, kwargs.get('lr_cos', 0))
        self.random_warp = kwargs.get('random_warp', True)
        self.random_src_flip = kwargs.get('random_src_flip', True)
        self.random_dst_flip = kwargs.get('random_dst_flip', True)
        self.random_hsv_power = max(0.0, min(kwargs.get('random_hsv_power', 0.0), 1.0))
        self.gan_power = max(0.0, kwargs.get('gan_power', 0.0))
        self.gan_patch_size = max(8, min(kwargs.get('gan_patch_size', self.resolution // 8), self.resolution))
        self.gan_dims = max(4, min(kwargs.get('gan_dims', 16), 64))
        self.true_face_power = max(0.0, kwargs.get('true_face_power', 0.0))
        self.vgg_perceptual_power = max(0.0, kwargs.get('vgg_perceptual_power', 0.0))
        self.ct_mode = kwargs.get('ct_mode', 'none')
        self.clipgrad = kwargs.get('clipgrad', False)
        self.pretrain = kwargs.get('pretrain', False)
        self.amp_mode = kwargs.get('amp_mode', 'bf16')
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)
        self.target_iter = max(0, kwargs.get('target_iter', 0))
        self.backup_interval = max(0, kwargs.get('backup_interval', 0))
        self.freeze_encoder = kwargs.get('freeze_encoder', False)
        self.freeze_inter = kwargs.get('freeze_inter', False)
        self.freeze_inter_AB = kwargs.get('freeze_inter_AB', False)
        self.freeze_inter_B = kwargs.get('freeze_inter_B', False)
        self.freeze_decoder_mask = kwargs.get('freeze_decoder_mask', False)
        self.enable_torch_compile = kwargs.get('enable_torch_compile', True)
        self.use_ms_ssim = kwargs.get('use_ms_ssim', True)
        self.adaptive_mask_dilation = kwargs.get('adaptive_mask_dilation', True)
        self.mask_dilation_sigma = max(0.5, kwargs.get('mask_dilation_sigma', 2.0))
        self.mask_dilation_radius = max(1, kwargs.get('mask_dilation_radius', 3))
        self.ramp_start_ratio = max(0.0, min(kwargs.get('ramp_start_ratio', 0.2), 0.8))
        self.smart_stop_enabled = kwargs.get('smart_stop_enabled', True)
        self.smart_stop_window = max(100, kwargs.get('smart_stop_window', 500))
        self.smart_stop_threshold = max(0.001, kwargs.get('smart_stop_threshold', 0.1))

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**d)

    @property
    def archi_type(self) -> str:
        return self.archi.split('-')[0]

    @property
    def archi_opts(self) -> str:
        parts = self.archi.split('-')
        return parts[1] if len(parts) > 1 else ''


class SAEHDTrainer(BaseTrainer):
    def __init__(self, config: TrainingConfig, model_dir: Path,
                 src_aligned_dir: Path, dst_aligned_dir: Path,
                 device: Optional[torch.device] = None,
                 progress_callback: Optional[Callable] = None,
                 preview_callback: Optional[Callable] = None):
        if config.pretrain:
            config.gan_power = 0.0
            config.random_warp = False
            config.random_hsv_power = 0.0
            config.vgg_perceptual_power = 0.0
            config.uniform_yaw = True
            config.lr_dropout = 'n'
            config.ramp_start_ratio = 0.0
            pretrain_dir = model_dir.parent / 'pretrain_faces'
            if pretrain_dir.exists():
                src_aligned_dir = pretrain_dir
                dst_aligned_dir = pretrain_dir
                _logger.info(f"Pretrain mode: using pretrain data from {pretrain_dir}")
            else:
                _logger.warning(f"Pretrain mode: {pretrain_dir} not found, using src/dst data")
            _logger.info("Pretrain mode: disabled gan/warp/hsv/vgg, enabled uniform_yaw")

        if device is None:
            from faceswap.shared.config import auto_select_device
            device = auto_select_device()

        _t_model = time.time()
        model = SAEHDModel(config, model_dir, device)
        _logger.info(f"[DIAG] SAEHDModel.__init__: {time.time()-_t_model:.2f}s")
        super().__init__(model, src_aligned_dir, dst_aligned_dir,
                         device, progress_callback, preview_callback)

        if config.enable_torch_compile:
            if getattr(config, 'gradient_checkpointing', False):
                _logger.warning("torch.compile与gradient_checkpointing不兼容，已禁用gradient_checkpointing")
                config.gradient_checkpointing = False
            _logger.info("torch.compile预热中，首次编译需要1-3分钟...")
            _t_compile = time.time()
            from faceswap.core.saehd_utils import apply_torch_compile
            apply_torch_compile(self.model, enable=True)
            _logger.info(f"[DIAG] apply_torch_compile: {time.time()-_t_compile:.2f}s")

    def create_datasets(self):
        c = self.config
        src_dataset = SAEHDDataset(
            self.src_aligned_dir, resolution=c.resolution, face_type=c.face_type,
            random_warp=c.random_warp, random_flip=c.random_src_flip,
            random_hsv_power=c.random_hsv_power, ct_mode=c.ct_mode,
            uniform_yaw=c.uniform_yaw,
            is_src=True)
        dst_dataset = SAEHDDataset(
            self.dst_aligned_dir, resolution=c.resolution, face_type=c.face_type,
            random_warp=c.random_warp, random_flip=c.random_dst_flip,
            random_hsv_power=0.0, ct_mode='none',
            uniform_yaw=c.uniform_yaw,
            is_src=False)
        return src_dataset, dst_dataset

    def preprocess_batch(self, batch_src: dict, batch_dst: dict) -> tuple[dict, dict]:
        batch_src = {k: v.to(self.device) for k, v in batch_src.items()}
        batch_dst = {k: v.to(self.device) for k, v in batch_dst.items()}
        return batch_src, batch_dst

    def postprocess_step(self, losses: dict) -> tuple[float, float]:
        return losses['src_loss'], losses['dst_loss']

    def save_model(self) -> None:
        self.model.save(self._iter_count)

    @torch.no_grad()
    def AE_merge(self, warped_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.model.merge(warped_dst)

    def predictor_func(self, face: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.config
        res = c.resolution
        if face.ndim == 3:
            face_in = face[None, ...]
        else:
            face_in = face
        face_resized = np.stack([cv2.resize(f, (res, res)) for f in face_in])
        face_nchw = face_resized.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        bgr, mask_dst, mask_src = self.AE_merge(face_nchw)
        bgr = bgr[0].transpose(1, 2, 0).astype(np.float32)
        mask_src = mask_src[0, 0].astype(np.float32)
        mask_dst = mask_dst[0, 0].astype(np.float32)
        return bgr, mask_src, mask_dst

    @torch.no_grad()
    def export_dfm(self, output_path: str, precision: str = 'fp32') -> None:
        import torch.nn as nn
        c = self.config
        m = self.model

        class _DFMWrapper(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.encoder = parent.encoder
                self.is_df = c.archi_type == 'df'
                if self.is_df:
                    self.inter = parent.inter
                    self.decoder_src = parent.decoder_src
                    self.decoder_dst = parent.decoder_dst
                else:
                    self.inter_AB = parent.inter_AB
                    self.inter_B = parent.inter_B
                    self.decoder = parent.decoder

            def forward(self, in_face):
                x = in_face.permute(0, 3, 1, 2).contiguous()
                if self.is_df:
                    code = self.inter(self.encoder(x))
                    out_celeb_face, out_celeb_face_mask = self.decoder_src(code)
                    _, out_face_mask = self.decoder_dst(code)
                else:
                    code = self.encoder(x)
                    inter_b = self.inter_B(code)
                    inter_ab = self.inter_AB(code)
                    code_dst = torch.cat([inter_b, inter_ab], dim=1)
                    code_src_dst = torch.cat([inter_ab, inter_ab], dim=1)
                    out_celeb_face, out_celeb_face_mask = self.decoder(code_src_dst)
                    _, out_face_mask = self.decoder(code_dst)
                out_face_mask = out_face_mask.permute(0, 2, 3, 1).contiguous()
                out_celeb_face = out_celeb_face.permute(0, 2, 3, 1).contiguous()
                out_celeb_face_mask = out_celeb_face_mask.permute(0, 2, 3, 1).contiguous()
                return out_face_mask, out_celeb_face, out_celeb_face_mask

        wrapper = _DFMWrapper(m)
        wrapper.eval()
        if precision == 'fp16':
            wrapper = wrapper.to('cpu', dtype=torch.float16)
            dummy = torch.zeros(1, c.resolution, c.resolution, 3, dtype=torch.float16)
        else:
            wrapper = wrapper.to('cpu', dtype=torch.float32)
            dummy = torch.zeros(1, c.resolution, c.resolution, 3, dtype=torch.float32)
        with torch.no_grad():
            _ = wrapper(dummy)

        export_kwargs = dict(
            input_names=['in_face:0'],
            output_names=['out_face_mask:0', 'out_celeb_face:0', 'out_celeb_face_mask:0'],
            dynamic_axes={
                'in_face:0': {0: 'batch'},
                'out_face_mask:0': {0: 'batch'},
                'out_celeb_face:0': {0: 'batch'},
                'out_celeb_face_mask:0': {0: 'batch'},
            },
            opset_version=12,
        )
        try:
            torch.onnx.export(wrapper, dummy, output_path, dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(wrapper, dummy, output_path, **export_kwargs)
        _logger.info(f"Exported .dfm to {output_path}")

    def get_module_info(self) -> list[tuple[str, int, str]]:
        return self.model.get_module_info()
