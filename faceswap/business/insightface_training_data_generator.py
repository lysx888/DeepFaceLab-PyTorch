import hashlib
import orjson
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from faceswap.core.metadata_manager import FaceMetadata
from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("insightface_training_data_generator")

_KPS5_NAMES = ["right_eye", "left_eye", "nose", "right_mouth", "left_mouth"]


def _compute_file_md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(str(file_path), 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


class InsightFaceTrainingDataGenerator:
    def __init__(self, workspace_dir: Path):
        self._workspace_dir = Path(workspace_dir)
        self._train_dir = self._workspace_dir / "insightface_train"
        self._to_annotate_dir = self._train_dir / "to_annotate"
        self._manual_annotated_dir = self._train_dir / "manual_annotated"

    @property
    def train_dir(self) -> Path:
        return self._train_dir

    @property
    def to_annotate_dir(self) -> Path:
        return self._to_annotate_dir

    @property
    def manual_annotated_dir(self) -> Path:
        return self._manual_annotated_dir

    def find_manual_annotation(self, source_image_path: Path) -> list[Path]:
        if not self._manual_annotated_dir.exists():
            return []
        name = _compute_file_md5(source_image_path)
        results = sorted(self._manual_annotated_dir.glob(f"{name}_*.json"))
        return results

    def add_to_annotate(self, source_image_path: Path) -> Path:
        self._to_annotate_dir.mkdir(parents=True, exist_ok=True)
        dest = self._to_annotate_dir / source_image_path.name
        if dest.exists():
            return dest
        img = cv2.imread(str(source_image_path))
        if img is not None:
            from faceswap.shared.file_manager import imwrite_auto
            imwrite_auto(dest, img)
        return dest

    def get_to_annotate_images(self) -> list[Path]:
        if not self._to_annotate_dir.exists():
            return []
        images = sorted(
            [f for f in self._to_annotate_dir.iterdir()
             if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
            key=lambda p: p.name
        )
        return images

    def generate_from_annotation(
        self,
        metadata: FaceMetadata,
        source_image_path: Optional[Path] = None,
        source_stem: Optional[str] = None,
        face_idx: int = 0,
    ) -> Path:
        self._manual_annotated_dir.mkdir(parents=True, exist_ok=True)

        if source_image_path is not None and source_image_path.exists():
            name = _compute_file_md5(source_image_path)
            full_name = f"{name}_{face_idx}"
            dest_img = self._manual_annotated_dir / f"{name}.jpg"
            if not dest_img.exists():
                shutil.copy2(str(source_image_path), str(dest_img))
        elif source_stem:
            full_name = f"{source_stem}_{face_idx}"
        else:
            full_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        annotation = {
            "source_filename": metadata.source_filename,
            "source_image_path": str(source_image_path) if source_image_path else None,
            "bbox": metadata.source_rect,
            "kps_5": metadata.source_kps_5.astype(np.float32).tolist() if metadata.source_kps_5 is not None else None,
            "kps_5_visibility": metadata.kps_5_visibility,
            "kps_5_names": _KPS5_NAMES,
            "landmarks_106": metadata.source_landmarks_106.astype(np.int64).tolist() if metadata.source_landmarks_106 is not None else None,
            "landmarks_106_visibility": metadata.landmarks_106_visibility,
            "annotation_time": datetime.now().isoformat(),
        }

        annotation_path = self._manual_annotated_dir / f"{full_name}.json"
        json_str = orjson.dumps(annotation)
        FileManager.atomic_write(annotation_path, json_str)

        return annotation_path
