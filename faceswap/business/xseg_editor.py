from __future__ import annotations
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QLabel

from faceswap.core.metadata_manager import MetadataManager, FaceMetadata
from faceswap.shared.file_manager import FileManager
from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.shared.logger import get_logger

_logger = get_logger("xseg_editor")


class XSegEditor:
    def __init__(self) -> None:
        self._aligned_dir: Optional[Path] = None
        self._current_index: int = 0
        self._images: list[Path] = []
        self._polys: dict[str, list[dict]] = {}

    def open(self, aligned_dir: Path) -> None:
        self._aligned_dir = Path(aligned_dir)
        self._images = FileManager.find_images(self._aligned_dir)
        if not self._images:
            _logger.warning(f"No images found in {aligned_dir}")
            return

        for img_path in self._images:
            meta = MetadataManager.load(img_path)
            if meta is not None and meta.seg_ie_polys is not None:
                self._polys[img_path.name] = [
                    {"type": poly.get("type", 1),
                     "pts": [[int(p[0]), int(p[1])] for p in poly.get("pts", [])]}
                    for poly in meta.seg_ie_polys
                ]

        try:
            self._launch_gui()
        except ImportError:
            _logger.error("PyQt6 not available. Cannot open XSeg editor.")
            _logger.info("Install PyQt6: pip install PyQt6")

    def _launch_gui(self) -> None:
        from PyQt6.QtWidgets import (
            QDialog, QWidget, QHBoxLayout,
            QLabel, QPushButton, QVBoxLayout, QListWidget,
        )
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QImage, QPixmap

        dlg = QDialog()
        dlg.setWindowTitle("XSeg Editor")
        dlg.setMinimumSize(800, 600)

        layout = QHBoxLayout(dlg)

        list_widget = QListWidget()
        for img_path in self._images:
            list_widget.addItem(img_path.name)
        layout.addWidget(list_widget)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(image_label)

        btn_layout = QVBoxLayout()
        save_btn = QPushButton("保存")
        close_btn = QPushButton("关闭")
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        def on_item_changed(current, previous):
            if current is None:
                return
            idx = list_widget.row(current)
            if 0 <= idx < len(self._images):
                self._show_image(self._images[idx], image_label)

        list_widget.currentItemChanged.connect(on_item_changed)

        def on_save():
            self._save_all_polys()

        save_btn.clicked.connect(on_save)
        close_btn.clicked.connect(dlg.accept)

        if self._images:
            list_widget.setCurrentRow(0)

        dlg.exec()

    def _show_image(self, img_path: Path, label: QLabel) -> None:
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtCore import Qt

        img = cv2.imread(str(img_path))
        if img is None:
            return

        polys = self._polys.get(img_path.name, [])
        for poly in polys:
            pts_data = poly.get("pts", []) if isinstance(poly, dict) else poly
            if len(pts_data) < 3:
                continue
            pts = np.array(pts_data, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        img_rgb = bgr_to_rgb(img)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _save_all_polys(self) -> None:
        if self._aligned_dir is None:
            return
        for img_path in self._images:
            meta = MetadataManager.load(img_path)
            if meta is None:
                continue
            polys = self._polys.get(img_path.name)
            if polys is not None:
                meta.seg_ie_polys = [
                    {"type": poly.get("type", 1),
                     "pts": [[float(p[0]), float(p[1])] for p in poly.get("pts", [])]}
                    for poly in polys
                ]
            MetadataManager.save(img_path, meta)
        _logger.info("XSeg annotations saved.")

    def fetch_annotated(self, aligned_dir: Path, output_dir: Optional[Path] = None) -> int:
        aligned_dir = Path(aligned_dir)
        out_dir = Path(output_dir) if output_dir else aligned_dir.parent / (aligned_dir.name + "_xseg")

        if out_dir.exists():
            import shutil
            for f in out_dir.iterdir():
                if f.is_file():
                    f.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in FileManager.find_images(aligned_dir):
            meta = MetadataManager.load(img_path)
            if meta is None or meta.seg_ie_polys is None:
                continue
            import shutil
            shutil.copy2(str(img_path), str(out_dir / img_path.name))
            json_src = MetadataManager._sidecar_path(img_path)
            if json_src.exists():
                shutil.copy2(str(json_src), str(out_dir / json_src.name))
            count += 1

        _logger.info(f"Fetched {count} annotated faces to {out_dir}")
        return count

    def remove_annotations(self, aligned_dir: Path) -> int:
        MetadataManager.remove_field(aligned_dir, "seg_ie_polys")
        _logger.info(f"Removed XSeg annotations from {aligned_dir}")
        return 0
