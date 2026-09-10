"""通用模型训练器，供 AMP/Quick96 共用。"""
import torch
from pathlib import Path
from typing import Optional, Callable

from faceswap.business.base_trainer import BaseTrainer
from faceswap.business.saehd_dataset import SAEHDDataset
from faceswap.shared.logger import get_logger

_logger = get_logger("generic_trainer")


class GenericTrainer(BaseTrainer):
    """通用训练器，通过 model_class 和 config 创建模型。"""

    def __init__(self, model_class, config, model_dir: Path,
                 src_aligned_dir: Path, dst_aligned_dir: Path,
                 device: Optional[torch.device] = None,
                 progress_callback: Optional[Callable] = None,
                 preview_callback: Optional[Callable] = None):
        if device is None:
            from faceswap.shared.config import auto_select_device
            device = auto_select_device()

        model = model_class(config, model_dir, device)
        super().__init__(model, src_aligned_dir, dst_aligned_dir,
                         device, progress_callback, preview_callback)

    def create_datasets(self):
        c = self.config
        uniform_yaw = getattr(c, 'uniform_yaw', False)
        random_warp = getattr(c, 'random_warp', True)
        src_flip = getattr(c, 'random_src_flip', True)
        dst_flip = getattr(c, 'random_dst_flip', True)
        ct_mode = getattr(c, 'ct_mode', 'none')
        random_hsv_power = getattr(c, 'random_hsv_power', 0.0)
        eyes_mouth_prio = getattr(c, 'eyes_mouth_prio', False)
        vis_power = getattr(c, 'visibility_loss_power', 0.0)

        src_dataset = SAEHDDataset(
            self.src_aligned_dir, resolution=c.resolution, face_type=c.face_type,
            random_warp=random_warp, random_flip=src_flip,
            random_hsv_power=random_hsv_power, ct_mode=ct_mode,
            uniform_yaw=uniform_yaw, is_src=True,
            need_em_mask=eyes_mouth_prio, need_vis_mask=vis_power > 0)
        dst_dataset = SAEHDDataset(
            self.dst_aligned_dir, resolution=c.resolution, face_type=c.face_type,
            random_warp=random_warp, random_flip=dst_flip,
            random_hsv_power=0.0, ct_mode='none',
            uniform_yaw=uniform_yaw, is_src=False,
            need_em_mask=eyes_mouth_prio, need_vis_mask=vis_power > 0)
        return src_dataset, dst_dataset

    def preprocess_batch(self, batch_src: dict, batch_dst: dict) -> tuple[dict, dict]:
        batch_src = {k: v.to(self.device) for k, v in batch_src.items()}
        batch_dst = {k: v.to(self.device) for k, v in batch_dst.items()}
        return batch_src, batch_dst
