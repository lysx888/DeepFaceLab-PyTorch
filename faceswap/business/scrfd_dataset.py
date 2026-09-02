import json
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("scrfd_dataset")

_INPUT_SIZE = 640
_NUM_KPS = 5
_KPS_FLIP_ORDER = [1, 0, 2, 4, 3]
_CROP_CHOICES = [0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
_CROP_CHOICES_FINETUNE = [0.8, 1.0, 1.2, 1.4]
_IMAGE_CACHE_MAX = 256


def _random_square_crop(img, bboxes, keypointss, crop_choices=_CROP_CHOICES):
    h, w = img.shape[:2]
    max_scale = max(crop_choices)

    scale = float(np.random.choice(crop_choices))
    for _ in range(250):
        short_side = min(w, h)
        cw = int(scale * short_side)
        ch = cw

        if w == cw:
            left = 0
        elif w > cw:
            left = random.randint(0, w - cw)
        else:
            left = random.randint(w - cw, 0)
        if h == ch:
            top = 0
        elif h > ch:
            top = random.randint(0, h - ch)
        else:
            top = random.randint(h - ch, 0)

        patch = np.array((left, top, left + cw, top + ch), dtype=np.int32)

        centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
        mask = (
            (centers[:, 0] > patch[0])
            & (centers[:, 1] > patch[1])
            & (centers[:, 0] < patch[2])
            & (centers[:, 1] < patch[3])
        )
        if not mask.any():
            continue

        rimg = np.ones((ch, cw, 3), dtype=img.dtype) * 128
        patch_from = patch.copy()
        patch_from[0] = max(0, patch_from[0])
        patch_from[1] = max(0, patch_from[1])
        patch_from[2] = min(img.shape[1], patch_from[2])
        patch_from[3] = min(img.shape[0], patch_from[3])
        patch_to = patch.copy()
        patch_to[0] = max(0, -patch_to[0])
        patch_to[1] = max(0, -patch_to[1])
        patch_to[2] = patch_to[0] + (patch_from[2] - patch_from[0])
        patch_to[3] = patch_to[1] + (patch_from[3] - patch_from[1])
        rimg[patch_to[1]:patch_to[3], patch_to[0]:patch_to[2], :] = (
            img[patch_from[1]:patch_from[3], patch_from[0]:patch_from[2], :]
        )

        new_bboxes = bboxes[mask].copy()
        new_keypointss = keypointss[mask].copy()

        new_bboxes[:, 0] -= patch[0]
        new_bboxes[:, 1] -= patch[1]
        new_bboxes[:, 2] -= patch[0]
        new_bboxes[:, 3] -= patch[1]

        new_keypointss[:, :, 0] -= patch[0]
        new_keypointss[:, :, 1] -= patch[1]

        return rimg, new_bboxes, new_keypointss

    return img, bboxes, keypointss


def _photo_metric_distortion(img, brightness_delta=32, contrast_range=(0.5, 1.5),
                              saturation_range=(0.5, 1.5), hue_delta=18):
    img = img.astype(np.float32)

    if random.randint(0, 1):
        delta = random.uniform(-brightness_delta, brightness_delta)
        img += delta

    mode = random.randint(0, 1)
    if mode == 1:
        if random.randint(0, 1):
            alpha = random.uniform(*contrast_range)
            img *= alpha

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if random.randint(0, 1):
        img[..., 1] *= random.uniform(*saturation_range)

    if random.randint(0, 1):
        img[..., 0] += random.uniform(-hue_delta, hue_delta)
        img[..., 0][img[..., 0] > 360] -= 360
        img[..., 0][img[..., 0] < 0] += 360

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    if mode == 0:
        if random.randint(0, 1):
            alpha = random.uniform(*contrast_range)
            img *= alpha

    if random.randint(0, 1):
        img = img[..., np.random.permutation(3)]

    return img


def _random_flip(img, bboxes, keypointss, flip_ratio=0.5):
    if random.random() >= flip_ratio:
        return img, bboxes, keypointss, False

    h, w = img.shape[:2]
    img = img[:, ::-1, :].copy()

    flipped_bboxes = bboxes.copy()
    flipped_bboxes[:, 0] = w - bboxes[:, 2]
    flipped_bboxes[:, 2] = w - bboxes[:, 0]

    flipped_kps = keypointss.copy()
    for idx, a in enumerate(_KPS_FLIP_ORDER):
        flipped_kps[:, idx, :] = keypointss[:, a, :]
    flipped_kps[:, :, 0] = w - flipped_kps[:, :, 0]

    return img, flipped_bboxes, flipped_kps, True


class SCRFDDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        augment: bool = True,
        input_size: int = _INPUT_SIZE,
        is_finetune: bool = False,
        deform_aug: bool = False,
    ):
        self._data_dir = Path(data_dir)
        self._augment = augment
        self._input_size = input_size
        self._crop_choices = _CROP_CHOICES_FINETUNE if is_finetune else _CROP_CHOICES
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._samples: list[tuple[Path, np.ndarray, np.ndarray]] = []

        self._deform_aug = deform_aug
        self._scan()

        if len(self._samples) == 0:
            raise ValueError(
                f"No annotated faces with bbox and kps_5 found in {self._data_dir}. "
                "Please annotate faces with the manual annotator first."
            )
        _logger.info(f"SCRFDDataset: {len(self._samples)} samples from {self._data_dir}")

    def _scan(self) -> None:
        if not self._data_dir.exists():
            return
        for img_path in FileManager.find_images(self._data_dir):
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                continue
            try:
                with open(str(json_path), "r", encoding="utf-8") as f:
                    ann = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            bbox = ann.get("bbox")
            kps_5 = ann.get("kps_5")
            if bbox is None or kps_5 is None:
                continue
            bbox = np.asarray(bbox, dtype=np.float32)
            if bbox.shape != (4,):
                continue
            kps = np.asarray(kps_5, dtype=np.float32)
            if kps.shape != (_NUM_KPS, 2):
                continue
            self._samples.append((img_path, bbox, kps))

    def _read_image(self, img_path: Path) -> np.ndarray | None:
        key = str(img_path)
        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            return self._image_cache[key]
        img = cv2.imread(str(img_path))
        if img is not None:
            if len(self._image_cache) >= _IMAGE_CACHE_MAX:
                self._image_cache.popitem(last=False)
            self._image_cache[key] = img
        return img

    def __len__(self) -> int:
        return len(self._samples)

    def _apply_deform(self, img, bboxes, keypointss):
        H, W = img.shape[:2]
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            fw = max(x2 - x1, 1.0)
            fh = max(y2 - y1, 1.0)
            if random.random() < 0.25:
                cx = random.uniform(x1, x2)
                cy = random.uniform(y1 + fh * 0.3, y2)
                rx = random.uniform(fw * 0.15, fw * 0.4)
                ry = rx * random.uniform(0.7, 1.2)
                angle = random.uniform(0, 180)
                angle_rad = angle * np.pi / 180.0
                box_half_w = np.sqrt(rx**2 * np.cos(angle_rad)**2 + ry**2 * np.sin(angle_rad)**2)
                box_half_h = np.sqrt(rx**2 * np.sin(angle_rad)**2 + ry**2 * np.cos(angle_rad)**2)
                bx1 = max(0, int(cx - box_half_w))
                by1 = max(0, int(cy - box_half_h))
                bx2 = min(W, int(cx + box_half_w) + 1)
                by2 = min(H, int(cy + box_half_h) + 1)
                bw = bx2 - bx1
                bh = by2 - by1
                if bw > 0 and bh > 0:
                    mask = np.zeros((bh, bw), dtype=np.uint8)
                    cv2.ellipse(mask, (int(cx - bx1), int(cy - by1)), (int(rx), int(ry)),
                                angle, 0, 360, 255, -1)
                    color = np.random.randint(70, 200, 3).tolist()
                    noise = np.random.normal(0, 18, (bh, bw, 3))
                    occ = np.clip(color + noise, 0, 255).astype(np.uint8)
                    img = img.copy()
                    sub = img[by1:by2, bx1:bx2]
                    sub[mask > 0] = occ[mask > 0]
                    img[by1:by2, bx1:bx2] = sub
                    for j in range(_NUM_KPS):
                        kx, ky = int(keypointss[i, j, 0]), int(keypointss[i, j, 1])
                        if bx1 <= kx < bx2 and by1 <= ky < by2 and mask[ky - by1, kx - bx1] > 0:
                            keypointss[i, j, 2] = 0.0
            if random.random() < 0.25:
                mouth_l = keypointss[i, 3, :2]
                mouth_r = keypointss[i, 4, :2]
                x_mid = (mouth_l[0] + mouth_r[0]) / 2
                y_mid = (mouth_l[1] + mouth_r[1]) / 2
                open_amt = random.uniform(3.0, 12.0) * (fh / 192.0)
                R = max(16.0, fh * 0.3)
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                d = np.sqrt((xx - x_mid) ** 2 + (yy - y_mid) ** 2)
                w = np.clip(1.0 - d / R, 0, 1) ** 1.5
                dy_field = np.sign(yy - y_mid) * open_amt * w
                dy_field = cv2.GaussianBlur(dy_field, (0, 0), 2.0).astype(np.float32)
                map_y = (yy - dy_field).astype(np.float32)
                img = cv2.remap(img, xx, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
                for j in range(_NUM_KPS):
                    kx = int(np.clip(keypointss[i, j, 0], 0, W - 1))
                    ky = int(np.clip(keypointss[i, j, 1], 0, H - 1))
                    keypointss[i, j, 1] += dy_field[ky, kx]
        return img, bboxes, keypointss

    def __getitem__(self, index: int) -> dict:
        img_path, bbox, kps = self._samples[index]
        img = self._read_image(img_path)
        if img is None:
            img = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            bbox = np.array([0, 0, self._input_size, self._input_size], dtype=np.float32)
            kps = np.zeros((_NUM_KPS, 2), dtype=np.float32)

        bboxes = bbox.reshape(1, 4).copy()
        keypointss = np.zeros((1, _NUM_KPS, 3), dtype=np.float32)
        keypointss[:, :, :2] = kps.reshape(1, _NUM_KPS, 2)
        keypointss[:, :, 2] = 1.0

        if self._augment:
            img, bboxes, keypointss = _random_square_crop(img, bboxes, keypointss, crop_choices=self._crop_choices)

        crop_h, crop_w = img.shape[:2]
        scale = self._input_size / max(crop_h, crop_w)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        img = cv2.resize(img, (new_w, new_h))
        padded = np.zeros((self._input_size, self._input_size, 3), dtype=img.dtype)
        padded[:new_h, :new_w] = img
        img = padded
        bboxes[:, 0] *= scale
        bboxes[:, 1] *= scale
        bboxes[:, 2] *= scale
        bboxes[:, 3] *= scale
        keypointss[:, :, 0] *= scale
        keypointss[:, :, 1] *= scale

        if self._augment:
            img, bboxes, keypointss, _ = _random_flip(img, bboxes, keypointss)
            img = _photo_metric_distortion(img)

        if self._deform_aug and self._augment:
            img, bboxes, keypointss = self._apply_deform(img, bboxes, keypointss)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        gt_bboxes = torch.from_numpy(bboxes).float()
        gt_labels = torch.zeros(len(bboxes), dtype=torch.long)
        gt_keypointss = torch.from_numpy(keypointss).float()

        return {
            'image': img,
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'gt_keypointss': gt_keypointss,
            'image_path': str(img_path),
        }
