import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from DeepFaceLab.shared.file_manager import FileManager, imwrite_auto
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("synthetics_data_preparer")


@dataclass
class SyntheticsDataStats:
    total_images: int = 0
    valid_images: int = 0
    missing_images: int = 0
    skipped_incomplete: int = 0
    skipped_align_fail: int = 0


class SyntheticsDataPreparer:
    def __init__(self, workspace_dir: Path):
        self._workspace_dir = Path(workspace_dir)
        self._train_dir = self._workspace_dir / "insightface_train"
        self._synthetics_dir = self._train_dir / "synthetics"
        self._manual_annotated_dir = self._train_dir / "manual_annotated"

    @property
    def synthetics_data_dir(self) -> Path:
        return self._synthetics_dir

    def prepare_from_manual_annotated(self, input_size: int = 256) -> SyntheticsDataStats:
        if not self._manual_annotated_dir.exists():
            raise FileNotFoundError(f"manual_annotated目录不存在: {self._manual_annotated_dir}")

        annotations = self._load_manual_annotated()
        if not annotations:
            raise FileNotFoundError(f"manual_annotated目录为空: {self._manual_annotated_dir}")

        FileManager.ensure_dir(self._synthetics_dir)

        from DeepFaceLab.core.insightface_adapter import InsightFaceAdapter
        from DeepFaceLab.setting import FaceType

        adapter = InsightFaceAdapter()
        stats = SyntheticsDataStats(total_images=len(annotations))

        X = []
        Y = []

        for i, ann in enumerate(annotations):
            lm_106 = ann.get("landmarks_106")
            kps_5 = ann.get("kps_5")

            if lm_106 is None or len(lm_106) < 106:
                stats.skipped_incomplete += 1
                _logger.warning(f"Annotation {i}: incomplete landmarks, skipping")
                continue

            src_img_path = ann.get("source_image_path")
            if src_img_path and Path(src_img_path).exists():
                src_path = Path(src_img_path)
            else:
                src_filename = ann.get("source_filename", "")
                src_path = self._workspace_dir / "data_src" / src_filename
                if not src_path.exists():
                    for ext in [".png", ".jpg", ".jpeg"]:
                        alt = src_path.with_suffix(ext)
                        if alt.exists():
                            src_path = alt
                            break

            if not src_path.exists():
                stats.missing_images += 1
                _logger.warning(f"Source image not found: {src_path}")
                continue

            source_img = cv2.imread(str(src_path))
            if source_img is None:
                stats.missing_images += 1
                continue

            lm_np = np.array(lm_106, dtype=np.int64)
            if lm_np.shape != (106, 2):
                stats.skipped_incomplete += 1
                continue

            kps_np = np.array(kps_5, dtype=np.float32) if kps_5 else None

            try:
                aligned = adapter.align_face(source_img, lm_np, FaceType.WHOLE_FACE, input_size, kps_5=kps_np)
            except Exception as e:
                stats.skipped_align_fail += 1
                _logger.warning(f"Alignment failed for {i}: {e}")
                continue

            seq = f"{i:06d}"
            img_name = seq + ".jpg"
            img_path = self._synthetics_dir / img_name
            imwrite_auto(img_path, aligned.image, jpg_quality=100)

            aligned_lm = cv2.transform(
                lm_np.reshape(1, -1, 2).astype(np.float32),
                aligned.transform_matrix,
            ).reshape(-1, 2)

            normalized = aligned_lm.copy()
            normalized[:, 0] = normalized[:, 0] / (input_size / 2.0) - 1.0
            normalized[:, 1] = normalized[:, 1] / (input_size / 2.0) - 1.0

            X.append(img_name)
            Y.append(normalized.astype(np.float32))
            stats.valid_images += 1

        if X:
            annot_path = self._synthetics_dir / "annot.pkl"
            with open(str(annot_path), "wb") as f:
                pickle.dump((X, Y), f)

        _logger.info(f"Synthetics data prepared: valid={stats.valid_images}, skipped={stats.skipped_incomplete + stats.skipped_align_fail}")
        return stats

    def validate_data(self) -> SyntheticsDataStats:
        stats = SyntheticsDataStats()
        annot_path = self._synthetics_dir / "annot.pkl"
        if not annot_path.exists():
            return stats

        with open(str(annot_path), "rb") as f:
            X, Y = pickle.load(f)

        stats.total_images = len(X)
        for img_name in X:
            if (self._synthetics_dir / img_name).exists():
                stats.valid_images += 1
            else:
                stats.missing_images += 1

        _logger.info(f"Synthetics validation: total={stats.total_images}, valid={stats.valid_images}, missing={stats.missing_images}")
        return stats

    def _load_manual_annotated(self) -> list[dict]:
        annotations = []
        if not self._manual_annotated_dir.exists():
            return annotations
        for json_path in sorted(self._manual_annotated_dir.glob("*.json")):
            try:
                with open(str(json_path), "r", encoding="utf-8") as f:
                    ann = json.load(f)
                annotations.append(ann)
            except Exception as e:
                _logger.warning(f"Failed to load annotation {json_path}: {e}")
        return annotations
