import json
import orjson
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


def _build_augment(augment: bool, is_finetune: bool = False) -> "A.ReplayCompose":
    transform_list = []
    if augment and _HAS_AUG:
        transform_list += [
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
            A.ToGray(p=0.1),
            A.GaussianBlur(blur_limit=(1, 5), p=0.1),
            A.MotionBlur(blur_limit=(3, 7), p=0.1),
            A.ImageCompression(quality_range=(50, 90), p=0.05),
        ]
        if is_finetune:
            transform_list += [
                A.Affine(
                    translate_percent=0.02, scale=(0.95, 1.05), rotate=0,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.3,
                ),
                A.HorizontalFlip(p=0.5),
            ]
        else:
            transform_list += [
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
        is_finetune: bool = False,
        deform_aug: bool = False,
        full_regression: bool = True,
        cache_size: int = _IMAGE_CACHE_MAX,
    ):
        self._data_dir = Path(data_dir)
        self._augment = augment and _HAS_AUG
        self._input_size = input_size
        self._aug = _build_augment(self._augment, is_finetune=is_finetune)
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._full_regression = full_regression
        self._dataset_id = 0

        self._deform_aug = None
        if deform_aug:
            from faceswap.models.if_landmark.deform_aug import DeformAug
            self._deform_aug = DeformAug(size=input_size)

        self._samples: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
        self._scan()

        if len(self._samples) == 0:
            raise ValueError(
                f"No annotated faces with 106 landmarks found in {self._data_dir}. "
                "Please annotate faces with the manual annotator first."
            )

    def _scan(self) -> None:
        if not self._data_dir.exists():
            return
        from faceswap.core.landmarks106 import wflw_98_to_if_106
        for json_path in sorted(self._data_dir.glob("*.json")):
            try:
                with open(str(json_path), "rb") as f:
                    ann = orjson.loads(f.read())
            except (ValueError, OSError):
                continue

            bbox = ann.get("bbox")
            if bbox is None:
                continue
            bbox = np.asarray(bbox, dtype=np.float32)
            if bbox.shape != (4,):
                continue

            landmarks = ann.get("landmarks_106")
            if landmarks is not None:
                lm = np.asarray(landmarks, dtype=np.float32)
                if lm.shape != (_NUM_LANDMARKS, 2):
                    continue
                vis = ann.get("landmarks_106_visibility")
                if vis is not None and len(vis) == _NUM_LANDMARKS:
                    visibility = np.asarray(vis, dtype=bool)
                else:
                    visibility = np.ones(_NUM_LANDMARKS, dtype=bool)
            else:
                landmarks_98 = ann.get("landmarks_98")
                if landmarks_98 is None:
                    continue
                lm98 = np.asarray(landmarks_98, dtype=np.float32)
                if lm98.shape != (98, 2):
                    continue
                lm, vis_mask = wflw_98_to_if_106(lm98)
                visibility = vis_mask.astype(bool)

            stem = json_path.stem
            base = stem.rsplit("_", 1)[0]
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = json_path.parent / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                for ext in [".jpg", ".jpeg", ".png"]:
                    candidate = json_path.parent / (base + ext)
                    if candidate.exists():
                        img_path = candidate
                        break
            if img_path is None:
                continue

            self._samples.append((img_path, lm, bbox, visibility))

    @classmethod
    def merge(cls, datasets: list["IFLandmarkDataset"]) -> "IFLandmarkDataset":
        merged = cls.__new__(cls)
        merged._data_dir = datasets[0]._data_dir if datasets else Path(".")
        merged._augment = any(d._augment for d in datasets)
        merged._input_size = datasets[0]._input_size if datasets else _INPUT_SIZE
        merged._aug = datasets[0]._aug if datasets else _build_augment(False)
        merged._image_cache = OrderedDict()
        merged._cache_size = max(d._cache_size for d in datasets) if datasets else _IMAGE_CACHE_MAX
        merged._full_regression = datasets[0]._full_regression if datasets else True
        merged._deform_aug = datasets[0]._deform_aug if datasets else None
        merged._samples = []
        merged._sample_dataset_ids = []
        for ds_id, ds in enumerate(datasets):
            ds._dataset_id = ds_id
            merged._samples.extend(ds._samples)
            merged._sample_dataset_ids.extend([ds_id] * len(ds._samples))
        merged._num_datasets = len(datasets)
        merged._dataset_id = 0
        return merged

    @property
    def sample_dataset_ids(self) -> list[int]:
        return getattr(self, '_sample_dataset_ids', [0] * len(self._samples))

    @property
    def num_datasets(self) -> int:
        return getattr(self, '_num_datasets', 1)

    def _read_image(self, img_path: Path) -> np.ndarray | None:
        key = str(img_path)
        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            return self._image_cache[key]
        img = cv2.imread(str(img_path))
        if img is not None:
            if len(self._image_cache) >= self._cache_size:
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

        if self._deform_aug is not None:
            img, landmarks, vis_out = self._deform_aug(img, landmarks, visible)
            if vis_out is not None:
                visible = vis_out

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        landmarks /= _HALF_SIZE
        landmarks -= 1.0
        label = landmarks.flatten()
        label = torch.from_numpy(label).float()
        visible_t = torch.from_numpy(visible.astype(np.float32))
        ds_id = self._sample_dataset_ids[index] if hasattr(self, '_sample_dataset_ids') else 0

        return {'image': img, 'label': label, 'visible': visible_t, 'image_path': str(img_path), 'dataset_id': ds_id}
