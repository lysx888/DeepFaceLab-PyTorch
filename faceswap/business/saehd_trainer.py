import torch
from pathlib import Path
from typing import Optional, Callable

from faceswap.business.base_trainer import BaseTrainer
from faceswap.business.saehd_dataset import SAEHDDataset
from faceswap.models.saehd.saehd_model import SAEHDModel
from faceswap.shared.logger import get_logger
from faceswap.shared.hardware_utils import get_available_ram_gb, image_cache_size as _image_cache_size

_logger = get_logger("saehd_trainer")


class TrainingConfig:
    _VALID_FACE_TYPES = ('wf', 'head')
    _VALID_ARCHI = ('df', 'liae')
    _VALID_LR_DROPOUT = ('n', 'y', 'cpu', 'ca')

    def __init__(self, **kwargs):
        self.face_type = kwargs.get('face_type', 'wf')
        if self.face_type not in self._VALID_FACE_TYPES:
            raise ValueError(f"face_type='{self.face_type}' 无效，可选: {self._VALID_FACE_TYPES}")

        self.resolution = kwargs.get('resolution', 0)
        if self.resolution <= 0:
            ft_res_map = {'wf': 256, 'head': 384}
            self.resolution = ft_res_map.get(self.face_type, 256)

        self.archi = kwargs.get('archi', 'df')
        archi_base = self.archi.split('-')[0]
        if archi_base not in self._VALID_ARCHI:
            raise ValueError(f"archi='{self.archi}' 无效，可选: {self._VALID_ARCHI}")

        self.ae_dims = max(16, min(kwargs.get('ae_dims', 256), 1024))
        self.e_dims = max(8, min(kwargs.get('e_dims', 64), 256))
        self.d_dims = max(8, min(kwargs.get('d_dims', 64), 256))
        self.d_mask_dims = max(8, min(kwargs.get('d_mask_dims', 22), 128))

        self.masked_training = kwargs.get('masked_training', True)
        self.eyes_mouth_prio = kwargs.get('eyes_mouth_prio', False)
        self.uniform_yaw = kwargs.get('uniform_yaw', False)
        self.blur_out_mask = kwargs.get('blur_out_mask', False)
        self.multiscale_loss_power = max(0.0, min(kwargs.get('multiscale_loss_power', 0.0), 10.0))
        self.visibility_loss_power = max(0.0, min(kwargs.get('visibility_loss_power', 0.0), 10.0))
        self.adabelief = kwargs.get('adabelief', True)

        self.lr_dropout = kwargs.get('lr_dropout', 'n')
        if self.lr_dropout not in self._VALID_LR_DROPOUT:
            raise ValueError(f"lr_dropout='{self.lr_dropout}' 无效，可选: {self._VALID_LR_DROPOUT}")

        self.random_warp = kwargs.get('random_warp', True)
        self.random_src_flip = kwargs.get('random_src_flip', True)
        self.random_dst_flip = kwargs.get('random_dst_flip', True)
        self.random_hsv_power = max(0.0, min(kwargs.get('random_hsv_power', 0.0), 1.0))

        self.true_face_power = max(0.0, min(kwargs.get('true_face_power', 0.0), 10.0))
        self.face_style_power = max(0.0, min(kwargs.get('face_style_power', 0.0), 100.0))
        self.bg_style_power = max(0.0, min(kwargs.get('bg_style_power', 0.0), 100.0))

        self.gan_power = max(0.0, kwargs.get('gan_power', 0.0))
        # DFL 对齐：GAN patch size 范围 3-640（DFL: np.clip(input_int(..., add_info="3-640"), 3, 640)），
        # 仅用于 find_archi 匹配最接近的感受野，不做额外下限/分辨率上限修正。
        self.gan_patch_size = max(3, min(kwargs.get('gan_patch_size', 16), 640))
        self.gan_dims = max(4, min(kwargs.get('gan_dims', 16), 64))

        self.ct_mode = kwargs.get('ct_mode', 'none')
        self.clipgrad = kwargs.get('clipgrad', False)
        self.pretrain = kwargs.get('pretrain', False)

        self.batch_size = max(1, kwargs.get('batch_size', 8))
        self.lr = max(1e-7, min(kwargs.get('lr', 5e-5), 1e-2))
        self.amp_mode = kwargs.get('amp_mode', 'fp16')

        if self.amp_mode == 'bf16' and not self.clipgrad:
            _logger.warning(
                "bf16模式下GradScaler被禁用，clipgrad是唯一的梯度保护。"
                "已自动开启clipgrad=True，防止大梯度破坏参数和污染优化器状态。")
            self.clipgrad = True
        self.target_iter = max(0, kwargs.get('target_iter', 0))
        self.backup_interval = max(0, kwargs.get('backup_interval', 0))
        self.enable_torch_compile = kwargs.get('enable_torch_compile', False)
        self.auto_batch = kwargs.get('auto_batch', False)
        self.gradient_accumulation_steps = max(1, kwargs.get('gradient_accumulation_steps', 1))

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**d)


class SAEHDTrainer(BaseTrainer):
    def __init__(self, config: TrainingConfig, model_dir: Path,
                 src_aligned_dir: Path, dst_aligned_dir: Path,
                 device: Optional[torch.device] = None,
                 progress_callback: Optional[Callable] = None,
                 preview_callback: Optional[Callable] = None):
        if device is None:
            from faceswap.shared.config import auto_select_device
            device = auto_select_device()

        if config.pretrain:
            from faceswap.setting import SAEHD_PRETRAIN_DATA_DIR
            use_pretrain_dir = False
            # 预训练数据源固定: workspace/pretrain_faces/faceset.pak (名称锁定)
            pak_path = SAEHD_PRETRAIN_DATA_DIR / "faceset.pak"
            if pak_path.is_file():
                # 先解压到临时目录, 成功后再替换旧数据, 避免失败时污染目录
                try:
                    import tempfile, shutil
                    from faceswap.business.face_tool import FaceTool
                    _logger.info(f"解压预训练数据集 {pak_path.name} -> {SAEHD_PRETRAIN_DATA_DIR}")
                    SAEHD_PRETRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
                    for f in SAEHD_PRETRAIN_DATA_DIR.iterdir():
                        if f.is_dir() and f.name.startswith('.unpack_'):
                            shutil.rmtree(f, ignore_errors=True)
                    tmp_dir = Path(tempfile.mkdtemp(dir=str(SAEHD_PRETRAIN_DATA_DIR), prefix=".unpack_"))
                    try:
                        FaceTool().unpack(pak_path, tmp_dir)
                        # 解压成功: 清理上次残留, 移入新数据
                        for f in SAEHD_PRETRAIN_DATA_DIR.iterdir():
                            if f == tmp_dir:
                                continue
                            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.json'):
                                f.unlink()
                        for f in tmp_dir.iterdir():
                            shutil.move(str(f), str(SAEHD_PRETRAIN_DATA_DIR / f.name))
                    finally:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    use_pretrain_dir = True
                except Exception as e:
                    _logger.warning(f"解压预训练数据集失败: {e}, 使用 src/dst 兜底")
            elif SAEHD_PRETRAIN_DATA_DIR.exists() and any(SAEHD_PRETRAIN_DATA_DIR.iterdir()):
                use_pretrain_dir = True
            if use_pretrain_dir:
                src_aligned_dir = SAEHD_PRETRAIN_DATA_DIR
            dst_aligned_dir = src_aligned_dir

        model = SAEHDModel(config, model_dir, device)
        super().__init__(model, src_aligned_dir, dst_aligned_dir,
                         device, progress_callback, preview_callback)

        if config.enable_torch_compile:
            _logger.info("torch.compile预热中，首次编译需要1-3分钟...")
            from faceswap.core.saehd_utils import apply_torch_compile
            apply_torch_compile(self.model, enable=True)

    def create_datasets(self):
        c = self.config
        if c.pretrain:
            uniform_yaw = True
            src_flip = True
            dst_flip = True
        else:
            uniform_yaw = c.uniform_yaw
            src_flip = c.random_src_flip
            dst_flip = c.random_dst_flip

        from faceswap.shared.config import get_num_workers
        n_workers = max(1, get_num_workers(self.device))
        _cache_n = _image_cache_size(get_available_ram_gb(), workers=n_workers)

        src_dataset = SAEHDDataset(
            self.src_aligned_dir, resolution=c.resolution, face_type=c.face_type,
            random_warp=c.random_warp, random_flip=src_flip,
            random_hsv_power=c.random_hsv_power, ct_mode=c.ct_mode,
            uniform_yaw=uniform_yaw,
            is_src=True,
            need_em_mask=c.eyes_mouth_prio,
            need_vis_mask=c.visibility_loss_power > 0,
            image_cache_size=_cache_n)
        dst_dataset = SAEHDDataset(
            self.dst_aligned_dir, resolution=c.resolution, face_type=c.face_type,
            random_warp=c.random_warp, random_flip=dst_flip,
            random_hsv_power=0.0, ct_mode='none',
            uniform_yaw=uniform_yaw,
            is_src=False,
            need_em_mask=c.eyes_mouth_prio,
            need_vis_mask=c.visibility_loss_power > 0,
            image_cache_size=_cache_n)
        return src_dataset, dst_dataset

    def preprocess_batch(self, batch_src: dict, batch_dst: dict) -> tuple[dict, dict]:
        batch_src = {k: v.to(self.device) for k, v in batch_src.items()}
        batch_dst = {k: v.to(self.device) for k, v in batch_dst.items()}
        return batch_src, batch_dst
