import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    _HAS_AUG = True
except ImportError:
    _HAS_AUG = False

from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("if_landmark_dataset")

_INPUT_SIZE = 192
_NUM_LANDMARKS = 106
_HALF_SIZE = _INPUT_SIZE // 2
_IMAGE_CACHE_MAX = 256

_FLIP_PAIRS = [
    (101, 43), (105, 48), (104, 49), (103, 51), (102, 50), (97, 46), (98, 47), (99, 45), (100, 44),
    (93, 35), (96, 41), (94, 40), (95, 42), (89, 39), (90, 37), (87, 33), (91, 36),
    (88, 34), (92, 38),
    (75, 81), (76, 82), (77, 83), (78, 84), (79, 85),
    (65, 54), (66, 57), (62, 60), (70, 69),
    (52, 55), (64, 56), (63, 53), (71, 59), (67, 58), (68, 61),
    (1, 24), (9, 23), (10, 22), (11, 21), (12, 20), (13, 19), (14, 18), (15, 32),
    (16, 31), (2, 30), (3, 29), (4, 28), (5, 27), (6, 26), (7, 25), (8, 17),
]

FLIP_MAP_106 = list(range(_NUM_LANDMARKS))
for _a, _b in _FLIP_PAIRS:
    FLIP_MAP_106[_a] = _b
    FLIP_MAP_106[_b] = _a
FLIP_MAP_106 = np.array(FLIP_MAP_106, dtype=np.int64)


def _get_similar_transform(
    center: np.ndarray,
    scale: float,
    output_size: int,
) -> np.ndarray:
    return np.array([
        [scale, 0.0, output_size / 2.0 - center[0] * scale],
        [0.0, scale, output_size / 2.0 - center[1] * scale],
    ], dtype=np.float32)


def _trans_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = pts.astype(np.float32)
    new_pts = np.zeros_like(pts)
    new_pts[:, 0] = M[0, 0] * pts[:, 0] + M[0, 1] * pts[:, 1] + M[0, 2]
    new_pts[:, 1] = M[1, 0] * pts[:, 0] + M[1, 1] * pts[:, 1] + M[1, 2]
    return new_pts


def _build_augment(augment: bool) -> "A.ReplayCompose":
    transform_list = []
    if augment and _HAS_AUG:
        transform_list += [
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
            A.ToGray(p=0.1),
            A.GaussianBlur(blur_limit=(1, 5), p=0.1),
            A.MotionBlur(blur_limit=(3, 7), p=0.1),
            A.ImageCompression(quality_range=(50, 90), p=0.05),
            A.Affine(
                translate_percent=0.05, scale=(0.9, 1.1), rotate=20,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
        ]
    return A.ReplayCompose(
        transform_list,
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    )


class IFLandmarkDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        augment: bool = True,
        input_size: int = _INPUT_SIZE,
    ):
        self._data_dir = Path(data_dir)
        self._augment = augment and _HAS_AUG
        self._input_size = input_size
        self._aug = _build_augment(self._augment)
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        self._samples: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
        self._scan()

        if len(self._samples) == 0:
            raise ValueError(
                f"No annotated faces with 106 landmarks found in {self._data_dir}. "
                "Please annotate faces with the manual annotator first."
            )
        _logger.info(f"IFLandmarkDataset: {len(self._samples)} samples from {self._data_dir}")

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

            landmarks = ann.get("landmarks_106")
            bbox = ann.get("bbox")
            if landmarks is None or bbox is None:
                continue
            lm = np.asarray(landmarks, dtype=np.float32)
            if lm.shape != (_NUM_LANDMARKS, 2):
                continue
            bbox = np.asarray(bbox, dtype=np.float32)
            if bbox.shape != (4,):
                continue
            vis = ann.get("landmarks_106_visibility")
            if vis is not None and len(vis) == _NUM_LANDMARKS:
                visibility = np.asarray(vis, dtype=bool)
            else:
                visibility = np.ones(_NUM_LANDMARKS, dtype=bool)
            self._samples.append((img_path, lm, bbox, visibility))

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

    def __getitem__(self, index: int) -> dict:
        img_path, landmarks, bbox, ann_visible = self._samples[index]
        img = self._read_image(img_path)
        if img is None:
            img = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
            landmarks = np.zeros((_NUM_LANDMARKS, 2), dtype=np.float32)
            ann_visible = np.zeros(_NUM_LANDMARKS, dtype=bool)
        else:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            max_wh = max(w, h)
            if max_wh < 1e-6:
                img = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
                landmarks = np.zeros((_NUM_LANDMARKS, 2), dtype=np.float32)
                ann_visible = np.zeros(_NUM_LANDMARKS, dtype=bool)
            else:
                center = np.array([(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2], dtype=np.float32)
                scale = self._input_size / (max_wh * 1.5)
                M = _get_similar_transform(center, scale, self._input_size)
                img = cv2.warpAffine(img, M, (self._input_size, self._input_size),
                                     flags=cv2.INTER_LINEAR, borderValue=0)
                landmarks = _trans_points(landmarks.copy(), M)

        img_pre_aug = img
        landmarks_pre_aug = landmarks.copy()

        t = self._aug(image=img, keypoints=landmarks.tolist())
        img_aug = t['image']
        landmarks_aug = np.array(t['keypoints'], dtype=np.float32)

        _OOB_EPS = 1.0
        in_bounds = (
            (landmarks_aug[:, 0] >= -_OOB_EPS)
            & (landmarks_aug[:, 0] <= self._input_size + _OOB_EPS)
            & (landmarks_aug[:, 1] >= -_OOB_EPS)
            & (landmarks_aug[:, 1] <= self._input_size + _OOB_EPS)
        )

        visible = ann_visible & in_bounds

        flipped = False
        if self._augment:
            for trans in t["replay"]["transforms"]:
                if trans["__class_fullname__"].endswith('HorizontalFlip'):
                    if trans.get("applied"):
                        flipped = True

        if visible.sum() < _NUM_LANDMARKS * 0.5:
            img = img_pre_aug
            landmarks = landmarks_pre_aug
            visible = ann_visible.copy()
            flipped = False
        else:
            img = img_aug
            landmarks = landmarks_aug
            if flipped:
                landmarks = landmarks[FLIP_MAP_106, :]
                visible = visible[FLIP_MAP_106]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        landmarks /= _HALF_SIZE
        landmarks -= 1.0
        label = landmarks.flatten()
        label = torch.from_numpy(label).float()
        visible_t = torch.from_numpy(visible.astype(np.float32))

        return {'image': img, 'label': label, 'visible': visible_t, 'image_path': str(img_path)}
