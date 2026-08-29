import hashlib
import json
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

    def find_manual_annotation(self, source_image_path: Path) -> Optional[Path]:
        if not self._manual_annotated_dir.exists():
            return None
        name = _compute_file_md5(source_image_path)
        json_path = self._manual_annotated_dir / f"{name}.json"
        return json_path if json_path.exists() else None

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
    ) -> Path:
        self._manual_annotated_dir.mkdir(parents=True, exist_ok=True)

        if source_image_path is not None and source_image_path.exists():
            name = _compute_file_md5(source_image_path)
            dest_img = self._manual_annotated_dir / f"{name}.jpg"
            shutil.copy2(str(source_image_path), str(dest_img))
        elif source_stem:
            name = source_stem
        else:
            name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        annotation = {
            "source_filename": metadata.source_filename,
            "source_image_path": str(source_image_path) if source_image_path else None,
            "bbox": metadata.source_rect,
            "source_face_rect": metadata.source_face_rect,
            "kps_5": metadata.source_kps_5.astype(np.float32).tolist() if metadata.source_kps_5 is not None else None,
            "kps_5_visibility": metadata.kps_5_visibility,
            "kps_5_names": _KPS5_NAMES,
            "landmarks_106": metadata.source_landmarks_106.astype(np.int64).tolist() if metadata.source_landmarks_106 is not None else None,
            "landmarks_106_visibility": metadata.landmarks_106_visibility,
            "face_type": int(metadata.face_type),
            "output_size": metadata.output_size,
            "annotation_time": datetime.now().isoformat(),
        }

        annotation_path = self._manual_annotated_dir / f"{name}.json"
        json_str = json.dumps(annotation, ensure_ascii=False, indent=2)
        FileManager.atomic_write(annotation_path, json_str)

        _logger.info(f"Training data generated: {annotation_path}")
        return annotation_path
