import orjson
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QPointF, QRectF, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QIntValidator
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QWidget, QSplitter, QListWidget,
    QMessageBox, QGroupBox, QListWidgetItem, QScrollArea,
    QSizePolicy, QStackedWidget,
)

from faceswap.setting import FaceType, DATA_SRC_DIR, DATA_DST_DIR, DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, INSIGHTFACE_TRAIN_DIR, WORKSPACE_DIR, PRETRAIN_DATA_DIR
from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger
from faceswap.core.debug_image_generator import save_debug_image
from faceswap.gui_app.gui_utils import save_face_annotation
from faceswap.business.insightface_training_data_generator import InsightFaceTrainingDataGenerator

from faceswap.gui_app.annotator_common import (
    ImageCanvas, _CompactThumbDelegate, _ThumbLoader, _toggle_canvas_visibility,
    _KPS5_NAMES, _KPS5_BUTTON_ORDER, _KPS5_COLORS,
    _LANDMARK_GROUPS_106, _ALL_106_INDICES, _IDX_TO_COLOR_106, _IDX_TO_GROUP_106,
    _106_LINE_CONNECTIONS, _DEFAULT_GROUP_SHAPES,
    _generate_default_group_points, _KPS5_DEFAULT_POS,
    _canvas_kps5_to_insightface,
    _resize_pad,
)

_logger = get_logger("manual_annotator")

class AnnotationStep(IntEnum):
    STEP_0_IDLE = 0
    STEP_2_KPS5 = 2
    STEP_3_LANDMARKS_106 = 3
    STEP_COMPLETE = 4



class _DataSourceDialog(QDialog):
    frame_selected = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择数据路径")
        self.setMinimumSize(900, 600)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.Window)
        self._image_list: list[Path] = []
        self._thumb_loader: Optional[_ThumbLoader] = None
        self._current_source: str = ""
        self._current_dir: Optional[Path] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(4)

        src_btn = QPushButton("源 (SRC)")
        src_btn.setStyleSheet("QPushButton { background-color: #2D5B8E; color: white; padding: 4px 10px; }"
                              "QPushButton:hover { background-color: #3D7BAE; }")
        src_btn.clicked.connect(lambda: self._load_source("src", DATA_SRC_DIR))
        btn_bar.addWidget(src_btn)

        dst_btn = QPushButton("目标 (DST)")
        dst_btn.setStyleSheet("QPushButton { background-color: #2D6A8E; color: white; padding: 4px 10px; }"
                              "QPushButton:hover { background-color: #3D8ABE; }")
        dst_btn.clicked.connect(lambda: self._load_source("dst", DATA_DST_DIR))
        btn_bar.addWidget(dst_btn)

        to_annotate_btn = QPushButton("标注训练")
        to_annotate_btn.setStyleSheet("QPushButton { background-color: #0078D4; color: white; padding: 4px 10px; }"
                                      "QPushButton:hover { background-color: #1098E4; }")
        to_annotate_btn.clicked.connect(lambda: self._load_source("to_annotate",
            InsightFaceTrainingDataGenerator(WORKSPACE_DIR).to_annotate_dir))
        btn_bar.addWidget(to_annotate_btn)

        if PRETRAIN_DATA_DIR.exists():
            for sub in sorted(PRETRAIN_DATA_DIR.iterdir()):
                if sub.is_dir():
                    name = sub.name
                    btn = QPushButton(name)
                    btn.setStyleSheet("QPushButton { background-color: #5B2D8E; color: white; padding: 4px 10px; }"
                                      "QPushButton:hover { background-color: #7B4DAE; }")
                    btn.clicked.connect(lambda _, n=name, d=sub: self._load_source("pretrain:" + n, d))
                    btn_bar.addWidget(btn)

        btn_bar.addStretch()
        layout.addLayout(btn_bar)

        self._thumb_list = QListWidget()
        self._thumb_list.setIconSize(QSize(120, 80))
        self._thumb_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._thumb_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._thumb_list.setSpacing(4)
        self._thumb_list.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self._thumb_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._thumb_list.setItemDelegate(_CompactThumbDelegate(self._thumb_list))
        thumb_font = self._thumb_list.font()
        thumb_font.setPixelSize(10)
        self._thumb_list.setFont(thumb_font)
        self._thumb_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._thumb_list)

        self._hint_label = QLabel("请点击上方按钮选择数据源")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setStyleSheet("color: #888888; font-size: 16px; padding: 40px;")
        layout.addWidget(self._hint_label)

    def _load_source(self, data_source: str, source_dir: Path):
        self._current_source = data_source
        self._current_dir = source_dir
        if self._thumb_loader is not None:
            self._thumb_loader.stop()
            self._thumb_loader.wait(5000)
            self._thumb_loader = None

        self._thumb_list.clear()
        if not source_dir.exists():
            self._hint_label.setText(f"目录不存在: {source_dir}")
            self._hint_label.show()
            return

        images = sorted(FileManager.find_images(source_dir), key=lambda p: p.name)
        if not images:
            self._hint_label.setText(f"目录为空: {source_dir}")
            self._hint_label.show()
            return

        self._hint_label.hide()
        self._image_list = images
        for img_path in images:
            item = QListWidgetItem(img_path.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(img_path))
            item.setSizeHint(QSize(126, 96))
            self._thumb_list.addItem(item)

        self._thumb_loader = _ThumbLoader(images, preload_count=0)
        self._thumb_loader.thumb_loaded.connect(self._on_thumb_loaded)
        self._thumb_loader.start()

    def _on_thumb_loaded(self, idx: int, qimg: QImage, stem: str, path_str: str):
        icon = QIcon(QPixmap.fromImage(qimg))
        if idx < self._thumb_list.count():
            item = self._thumb_list.item(idx)
            item.setIcon(icon)
            item.setSizeHint(QSize(126, 96))

    def _on_item_clicked(self, item):
        path_str = item.data(Qt.ItemDataRole.UserRole)
        self.frame_selected.emit(path_str, self._current_source, str(self._current_dir))
        self.accept()

    def done(self, result):
        if self._thumb_loader is not None:
            self._thumb_loader.stop()
            self._thumb_loader.wait(5000)
        super().done(result)


class ManualAnnotatorDialog(QDialog):
    def __init__(self, parent=None, is_src: bool = True, img_path: Optional[Path] = None):
        super().__init__(parent)
        self.setWindowTitle("手动人脸标注")
        self.setMinimumSize(1300, 800)
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.Window)
        self.showMaximized()
        self._is_src = is_src
        self._current_img_path = img_path
        self._current_img = None
        self._face_type = FaceType.WHOLE_FACE
        self._output_size = 512
        self._kps5_saved = False
        self._annotation_step = AnnotationStep.STEP_0_IDLE
        self._to_annotate_images = []
        self._current_to_annotate_idx = -1
        self._current_face_idx = 0
        self._data_source: str = "src" if is_src else "dst"
        self._pretrain_dir: Optional[Path] = None
        self._existing_annotation_paths: list[Path] = []
        self._build_ui()
        if img_path is not None:
            self._init_navigation_list(img_path)
            self._load_single_image(img_path)
        else:
            self._to_annotate_images = InsightFaceTrainingDataGenerator(WORKSPACE_DIR).get_to_annotate_images()
            if self._to_annotate_images:
                self._load_to_annotate_image(0)
            else:
                self._canvas.set_empty_hint('请先把图片复制到to_annotate目录再打开此页面或点击左上角的\"选择数据路径\"')
                self._status.setText('请先把图片复制到to_annotate目录再打开此页面或点击左上角的\"选择数据路径\"')

    def _build_ui(self):
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()

        src_btn = QPushButton("选择数据路径")
        src_btn.setStyleSheet("QPushButton { background-color: #2D5B8E; color: white; font-weight: bold; padding: 4px 12px; }"
                              "QPushButton:hover { background-color: #3D7BAE; }")
        src_btn.clicked.connect(self._open_data_source)
        toolbar.addWidget(src_btn)

        self._img_label = QLabel("未加载")
        self._img_label.setMinimumWidth(150)
        self._img_label.setStyleSheet("font-weight: bold; padding: 2px 8px;")
        toolbar.addWidget(self._img_label)

        prev_btn = QPushButton("◀ 上一帧")
        prev_btn.clicked.connect(self._prev_image)
        toolbar.addWidget(prev_btn)
        next_btn = QPushButton("下一帧 ▶")
        next_btn.clicked.connect(self._next_image)
        toolbar.addWidget(next_btn)

        toolbar.addSpacing(10)

        ft_label = QLabel("人脸类型:")
        toolbar.addWidget(ft_label)
        self._ft_combo = QComboBox()
        self._ft_combo.addItems(["whole_face", "head"])
        self._ft_combo.setFixedWidth(120)
        self._ft_combo.currentTextChanged.connect(self._on_face_type_changed)
        toolbar.addWidget(self._ft_combo)

        sz_label = QLabel("尺寸:")
        toolbar.addWidget(sz_label)
        self._size_combo = QComboBox()
        self._size_combo.setEditable(True)
        self._size_combo.addItems(["512", "768"])
        self._size_combo.setCurrentText("512")
        self._size_combo.setFixedWidth(80)
        self._size_combo.setValidator(QIntValidator(64, 4096))
        toolbar.addWidget(self._size_combo)

        toolbar.addSpacing(10)

        auto_btn = QPushButton("自动标注")
        auto_btn.setStyleSheet("QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 5px 14px; }")
        auto_btn.clicked.connect(self._auto_annotate)
        toolbar.addWidget(auto_btn)

        apply_next_btn = QPushButton("应用到下一帧")
        apply_next_btn.setStyleSheet("QPushButton { background-color: #2D6A8E; color: white; font-weight: bold; padding: 5px 14px; }")
        apply_next_btn.clicked.connect(self._apply_to_next)
        toolbar.addWidget(apply_next_btn)

        toolbar.addStretch()

        layout.addLayout(toolbar)

        self._canvas = ImageCanvas(show_point_numbers=False, right_click_pan=True)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        splitter.addWidget(self._canvas)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)
        right_layout.setSpacing(2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)

        help_label = QLabel(
            "操作说明:\n"
            "1、左键拖动调整点位置\n"
            "2、按V切换当前组可见/不可见\n"
        )
        help_label.setStyleSheet("font-size: 11px; color: #888888; padding: 4px;")
        scroll_layout.addWidget(help_label)

        self._point_num_toggle_btn = QPushButton("显示/隐藏 序号")
        self._point_num_toggle_btn.setCheckable(True)
        self._point_num_toggle_btn.setChecked(False)
        self._point_num_toggle_btn.setStyleSheet("QPushButton { color: #CCCCCC; padding: 4px 8px; }")
        self._point_num_toggle_btn.clicked.connect(self._on_point_num_toggle)
        scroll_layout.addWidget(self._point_num_toggle_btn)

        self._vis_toggle_btn = QPushButton("标记不可见 (V)")
        self._vis_toggle_btn.setStyleSheet("QPushButton { color: #FF6600; padding: 4px 8px; }")
        self._vis_toggle_btn.clicked.connect(self._toggle_visibility)
        scroll_layout.addWidget(self._vis_toggle_btn)

        kps5_grp = QGroupBox("5点检测器及检测框")
        kps5_grp.setStyleSheet("QGroupBox { padding-top: 14px; padding-right: 4px; padding-bottom: 2px; padding-left: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 2px; }")
        kps5_layout = QVBoxLayout(kps5_grp)
        self._gen_bbox_btn = QPushButton("生成检测框")
        self._gen_bbox_btn.setStyleSheet("QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 4px 8px; }")
        self._gen_bbox_btn.clicked.connect(self._generate_bbox)
        kps5_layout.addWidget(self._gen_bbox_btn)
        kps5_btn_style = "QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 3px 6px; border-radius: 3px; }"
        for row_indices in _KPS5_BUTTON_ORDER:
            kps5_row = QHBoxLayout()
            for kps_idx in row_indices:
                btn = QPushButton(_KPS5_NAMES[kps_idx])
                btn.setStyleSheet(kps5_btn_style)
                btn.clicked.connect(lambda checked, ki=kps_idx: self._on_kps5_point_btn(ki))
                kps5_row.addWidget(btn)
            kps5_layout.addLayout(kps5_row)
        self._kps5_toggle_btn = QPushButton("显示/隐藏 5点和检测框")
        self._kps5_toggle_btn.setCheckable(True)
        self._kps5_toggle_btn.setChecked(True)
        self._kps5_toggle_btn.setStyleSheet("QPushButton { color: #00FFFF; padding: 4px 8px; }")
        self._kps5_toggle_btn.clicked.connect(self._on_kps5_toggle)
        kps5_layout.addWidget(self._kps5_toggle_btn)
        self._kps5_info_label = QLabel("点击按钮在检测框内添加点\n拖动调整位置")
        self._kps5_info_label.setStyleSheet("font-size: 10px; color: #888888;")
        kps5_layout.addWidget(self._kps5_info_label)
        scroll_layout.addWidget(kps5_grp)

        align106_grp = QGroupBox("106点对齐器")
        align106_grp.setStyleSheet("QGroupBox { padding-top: 14px; padding-right: 4px; padding-bottom: 2px; padding-left: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 2px; }")
        align106_layout = QVBoxLayout(align106_grp)
        self._lm106_toggle_btn = QPushButton("显示/隐藏 106点")
        self._lm106_toggle_btn.setCheckable(True)
        self._lm106_toggle_btn.setChecked(True)
        self._lm106_toggle_btn.setStyleSheet("QPushButton { color: #CCCCCC; padding: 4px 8px; }")
        self._lm106_toggle_btn.clicked.connect(self._on_lm106_toggle)
        align106_layout.addWidget(self._lm106_toggle_btn)
        btn_row_106 = None
        for gidx, (gname, gidxs, gcolor) in enumerate(_LANDMARK_GROUPS_106):
            btn = QPushButton(gname)
            btn.setStyleSheet(f"QPushButton {{ color: {gcolor.name()}; padding: 3px 4px; }}")
            btn.clicked.connect(lambda checked, gn=gname: self._on_106_group_btn(gn))
            if gidx % 2 == 0:
                btn_row_106 = QHBoxLayout()
            btn_row_106.addWidget(btn)
            if gidx % 2 == 1 or gidx == len(_LANDMARK_GROUPS_106) - 1:
                align106_layout.addLayout(btn_row_106)

        scroll_layout.addWidget(align106_grp)

        action_grp = QGroupBox("操作")
        action_grp.setStyleSheet("QGroupBox { padding-top: 14px; padding-right: 4px; padding-bottom: 2px; padding-left: 4px; margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 2px; }")
        action_layout = QVBoxLayout(action_grp)
        btn_row = QHBoxLayout()
        save106_btn = QPushButton("保存")
        save106_btn.setStyleSheet("QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 3px 8px; }")
        save106_btn.clicked.connect(self._save_106)
        btn_row.addWidget(save106_btn)
        delete_btn = QPushButton("删除")
        delete_btn.setStyleSheet("QPushButton { background-color: #C42B1C; color: white; font-weight: bold; padding: 3px 8px; }")
        delete_btn.clicked.connect(self._delete_current_group)
        btn_row.addWidget(delete_btn)
        action_layout.addLayout(btn_row)
        scroll_layout.addWidget(action_grp)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([800, 200])

        layout.addWidget(splitter, 1)

        self._face_sel_bar = QWidget()
        self._face_sel_layout = QHBoxLayout(self._face_sel_bar)
        self._face_sel_layout.setContentsMargins(4, 2, 4, 2)
        self._face_sel_layout.setSpacing(4)
        self._face_sel_bar.setVisible(False)
        layout.addWidget(self._face_sel_bar)
        self._detected_faces: list = []
        self._face_btns: list[QPushButton] = []

        self._status = QLabel("请加载图片开始标注")
        self._status.setStyleSheet("padding: 4px; color: #CCCCCC;")
        layout.addWidget(self._status)

        self._canvas.landmark_changed.connect(self._update_progress)

    def _open_data_source(self):
        dlg = _DataSourceDialog(self)
        dlg.frame_selected.connect(self._on_frame_selected)
        dlg.exec()

    def _on_frame_selected(self, img_path: str, data_source: str, source_dir: str):
        p = Path(img_path)
        self._data_source = data_source
        if data_source.startswith("pretrain:"):
            self._pretrain_dir = Path(source_dir)
            self._is_src = True
        elif data_source == "src":
            self._is_src = True
        elif data_source == "dst":
            self._is_src = False
        self._init_navigation_list(p)
        self._load_single_image(p)

    def _on_face_type_changed(self, text):
        size_map = {"whole_face": "512", "head": "768"}
        if text in size_map:
            self._size_combo.setCurrentText(size_map[text])

    def _on_kps5_toggle(self, checked):
        self._canvas.set_visible_group(None)
        self._canvas.set_show_kps5(checked)
        self._canvas.set_show_bbox(checked)

    def _on_lm106_toggle(self, checked):
        self._canvas.set_visible_group(None)
        self._canvas.set_show_106(checked)

    def _on_point_num_toggle(self, checked):
        self._canvas._show_point_numbers = checked
        self._canvas.update()

    def _toggle_visibility(self):
        msg = _toggle_canvas_visibility(self._canvas)
        if msg:
            self._status.setText(msg)
        else:
            QMessageBox.information(self, "提示", "请先双击锁定一个五官部位，或点击选中一个点")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_V:
            self._toggle_visibility()
            return
        super().keyPressEvent(event)

    def _get_current_step(self) -> AnnotationStep:
        if self._current_img is None:
            return AnnotationStep.STEP_0_IDLE
        kps5 = self._canvas.get_kps_5()
        if not self._kps5_saved or len(kps5) < 5 or self._canvas.get_bbox_rect() is None:
            return AnnotationStep.STEP_2_KPS5
        lm106 = self._canvas.get_landmarks_106()
        if len(lm106) < 106:
            return AnnotationStep.STEP_3_LANDMARKS_106
        return AnnotationStep.STEP_COMPLETE

    def _check_step_prerequisite(self, required_step: AnnotationStep) -> bool:
        current = self._get_current_step()
        if current >= required_step:
            return True
        if required_step == AnnotationStep.STEP_2_KPS5:
            QMessageBox.warning(self, "提示", "请先添加5点检测器和检测框")
        elif required_step == AnnotationStep.STEP_3_LANDMARKS_106:
            if len(self._canvas.get_kps_5()) < 5:
                QMessageBox.warning(self, "提示", "请先添加5点检测器")
            elif self._canvas.get_bbox_rect() is None:
                QMessageBox.warning(self, "提示", "请先点击\"生成检测框\"")
            else:
                QMessageBox.warning(self, "提示", "请先添加5点检测器和检测框")
        return False

    def _on_kps5_point_btn(self, kps_idx: int):
        if not self._check_step_prerequisite(AnnotationStep.STEP_2_KPS5):
            return
        kps5 = self._canvas.get_kps_5()
        if kps_idx in kps5:
            QMessageBox.warning(self, "提示", f"{_KPS5_NAMES[kps_idx]}已添加，请拖动调整位置")
            return
        bbox = self._canvas.get_bbox_rect()
        if kps_idx in _KPS5_DEFAULT_POS and bbox is not None:
            fx, fy = _KPS5_DEFAULT_POS[kps_idx]
            pos = QPointF(bbox.x() + fx * bbox.width(), bbox.y() + fy * bbox.height())
        else:
            h, w = self._current_img.shape[:2]
            pos = QPointF(w / 2, h / 2)
        kps5[kps_idx] = pos
        self._canvas.set_kps_5(kps5)
        self._canvas.set_show_kps5(self._kps5_toggle_btn.isChecked())
        if len(kps5) == 5:
            self._kps5_saved = True
        self._status.setText(f"已添加 {_KPS5_NAMES[kps_idx]}，请拖动调整位置")

    def _load_single_image(self, img_path: Path):
        self._current_img_path = img_path
        self._current_img = cv2.imread(str(img_path))
        if self._current_img is None:
            self._status.setText(f"无法读取: {img_path.name}")
            return
        self._canvas.set_empty_hint(None)
        self._canvas.set_image(self._current_img)
        self._canvas.clear_landmarks()
        self._canvas.set_bbox_rect(None)
        self._canvas.set_visible_group(None)
        self._kps5_saved = False
        self._annotation_step = AnnotationStep.STEP_2_KPS5
        self._detected_faces = []
        self._existing_annotation_paths = []
        self._face_btns = []
        self._current_face_idx = 0
        while self._face_sel_layout.count():
            item = self._face_sel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._face_sel_bar.setVisible(False)
        self._img_label.setText(img_path.name)
        self._status.setText(f"已加载: {img_path.name} ({self._current_img.shape[1]}x{self._current_img.shape[0]})")
        self._try_load_existing_annotation(img_path)

    def _load_annotation_to_canvas(self, ann: dict):
        bbox = ann.get("bbox")
        kps_5 = ann.get("kps_5")
        landmarks_106 = ann.get("landmarks_106")

        if bbox is not None and len(bbox) == 4:
            self._canvas.set_bbox_rect(QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

        if kps_5 is not None and len(kps_5) == 5:
            canvas_kps = {
                0: QPointF(kps_5[1][0], kps_5[1][1]),
                1: QPointF(kps_5[0][0], kps_5[0][1]),
                2: QPointF(kps_5[2][0], kps_5[2][1]),
                3: QPointF(kps_5[4][0], kps_5[4][1]),
                4: QPointF(kps_5[3][0], kps_5[3][1]),
            }
            self._canvas.set_kps_5(canvas_kps)

        if landmarks_106 is not None and len(landmarks_106) == 106:
            canvas_lm = {i: QPointF(pt[0], pt[1]) for i, pt in enumerate(landmarks_106)}
            self._canvas.set_landmarks_106(canvas_lm)
            lm106_vis = ann.get("landmarks_106_visibility")
            if lm106_vis is not None and len(lm106_vis) == 106:
                self._canvas.set_lm106_visibility({i: bool(lm106_vis[i]) for i in range(106)})

        self._kps5_saved = True
        self._annotation_step = AnnotationStep.STEP_COMPLETE

    def _try_load_existing_annotation(self, img_path: Path):
        json_paths: list[Path] = []
        if self._data_source.startswith("pretrain:") and self._pretrain_dir is not None:
            jp = self._pretrain_dir / (img_path.stem + ".json")
            if jp.exists():
                json_paths = [jp]
        else:
            generator = InsightFaceTrainingDataGenerator(WORKSPACE_DIR)
            json_paths = generator.find_manual_annotation(img_path)

        if not json_paths:
            return

        self._existing_annotation_paths = json_paths

        if len(json_paths) > 1:
            self._show_existing_face_buttons(json_paths)

        self._load_existing_annotation_by_idx(0)
        self._status.setText(f"已加载: {img_path.name} (已恢复标注)")

    def _show_existing_face_buttons(self, json_paths: list[Path]):
        while self._face_sel_layout.count():
            item = self._face_sel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._face_btns = []
        label = QLabel(f"检测到 {len(json_paths)} 个人脸标注，请选择:")
        label.setStyleSheet("color: #CCCCCC; padding: 2px 6px;")
        self._face_sel_layout.addWidget(label)
        for i, jp in enumerate(json_paths):
            btn = QPushButton(f"人脸{i+1}")
            btn.clicked.connect(lambda _, idx=i: self._on_existing_face_selected(idx))
            self._face_sel_layout.addWidget(btn)
            self._face_btns.append(btn)
        self._face_sel_layout.addStretch()
        self._face_sel_bar.setVisible(True)
        self._update_face_btn_highlight(0)

    def _on_existing_face_selected(self, idx):
        if idx < 0 or idx >= len(self._existing_annotation_paths):
            return
        self._current_face_idx = idx
        self._load_existing_annotation_by_idx(idx)
        self._update_face_btn_highlight(idx)
        self._status.setText(f"已切换到人脸{idx+1}")

    def _load_existing_annotation_by_idx(self, idx: int):
        if idx < 0 or idx >= len(self._existing_annotation_paths):
            return
        try:
            with open(str(self._existing_annotation_paths[idx]), "rb") as f:
                ann = orjson.loads(f.read())
        except (ValueError, OSError):
            return
        self._load_annotation_to_canvas(ann)

    def _init_navigation_list(self, img_path: Path):
        img_path = Path(img_path)
        aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
        debug_dir = aligned_dir.parent / (aligned_dir.name + "_debug")
        to_annotate_dir = InsightFaceTrainingDataGenerator(WORKSPACE_DIR).to_annotate_dir
        try:
            img_parent = img_path.parent.resolve()
        except Exception:
            img_parent = img_path.parent
        if self._data_source.startswith("pretrain:") and self._pretrain_dir is not None:
            self._to_annotate_images = sorted(FileManager.find_images(self._pretrain_dir), key=lambda p: p.name)
        elif debug_dir.exists() and str(img_parent).startswith(str(debug_dir.resolve())):
            self._to_annotate_images = sorted(FileManager.find_images(debug_dir), key=lambda p: p.name)
        elif to_annotate_dir.exists() and str(img_parent).startswith(str(to_annotate_dir.resolve())):
            self._to_annotate_images = InsightFaceTrainingDataGenerator(WORKSPACE_DIR).get_to_annotate_images()
        else:
            frames_dir = DATA_SRC_DIR if self._is_src else DATA_DST_DIR
            self._to_annotate_images = sorted(FileManager.find_images(frames_dir), key=lambda p: p.name)
        for i, p in enumerate(self._to_annotate_images):
            if str(p) == str(img_path):
                self._current_to_annotate_idx = i
                break

    def _load_to_annotate_image(self, idx: int):
        if idx < 0 or idx >= len(self._to_annotate_images):
            return
        self._current_to_annotate_idx = idx
        img_path = self._to_annotate_images[idx]
        self._load_single_image(img_path)
        total = len(self._to_annotate_images)
        self._img_label.setText(f"[{idx + 1}/{total}] {img_path.name}")
        self._status.setText(
            f"[{idx + 1}/{total}] {img_path.name} "
            f"({self._current_img.shape[1]}x{self._current_img.shape[0]})"
        )

    def _prev_image(self):
        if self._current_to_annotate_idx > 0:
            self._load_to_annotate_image(self._current_to_annotate_idx - 1)

    def _next_image(self):
        if self._current_to_annotate_idx < len(self._to_annotate_images) - 1:
            self._load_to_annotate_image(self._current_to_annotate_idx + 1)

    def _generate_bbox(self):
        if self._current_img is None:
            QMessageBox.warning(self, "提示", "请先加载图片")
            return
        if not self._check_step_prerequisite(AnnotationStep.STEP_2_KPS5):
            return
        h, w = self._current_img.shape[:2]
        self._canvas.set_bbox_rect(QRectF(w * 0.25, h * 0.15, w * 0.5, h * 0.7))
        self._status.setText("检测框已生成，请拖动调整位置")

    def _on_106_group_btn(self, group_name: str):
        if not self._check_step_prerequisite(AnnotationStep.STEP_3_LANDMARKS_106):
            return

        missing = []
        kps5 = self._canvas.get_kps_5()
        if len(kps5) < 5:
            missing.append("5点检测器")
        if self._canvas.get_bbox_rect() is None:
            missing.append("检测框")
        if missing:
            QMessageBox.warning(self, "提示", f"缺少: {', '.join(missing)}\n请先添加5点检测器和检测框后再添加106点。")
            return

        h, w = self._current_img.shape[:2]
        bbox = self._canvas.get_bbox_rect()
        if bbox is not None:
            bw, bh = bbox.width(), bbox.height()
            m = max(bw, bh) * 0.3
            anchor_rect = QRectF(bbox.x() - m, bbox.y() - m, bw + 2 * m, bh + 2 * m)
        else:
            m = 0.1
            anchor_rect = QRectF(w * m, h * m, w * (1 - 2 * m), h * (1 - 2 * m))
        default_pts = _generate_default_group_points(group_name, anchor_rect)

        lm106 = self._canvas.get_landmarks_106()
        for gname, gidxs, _ in _LANDMARK_GROUPS_106:
            if gname == group_name:
                for idx in gidxs:
                    if idx not in lm106:
                        if idx in default_pts:
                            lm106[idx] = default_pts[idx]
                        else:
                            center = QPointF(anchor_rect.center().x(), anchor_rect.center().y())
                            lm106[idx] = center
                break

        self._canvas.set_landmarks_106(lm106)
        self._canvas.set_show_106(self._lm106_toggle_btn.isChecked())
        self._canvas.set_visible_group(group_name)
        self._canvas.set_show_kps5(self._kps5_toggle_btn.isChecked())

        for gname, gidxs, _ in _LANDMARK_GROUPS_106:
            if gname == group_name:
                for idx in gidxs:
                    if idx in lm106:
                        self._status.setText(f"编辑: {group_name} - 左键拖动调整")
                        return
                if gidxs:
                    self._status.setText(f"查看: {group_name} - 无特征点")
                return

    def _apply_face_to_canvas(self, face):
        if face.landmarks_106 is not None and face.landmarks_106.shape[0] == 106:
            lm = face.landmarks_106
            landmarks_106 = {}
            for i in range(106):
                landmarks_106[i] = QPointF(float(lm[i, 0]), float(lm[i, 1]))
            self._canvas.set_landmarks_106(landmarks_106)
        if face.kps_5 is not None and face.kps_5.shape[0] == 5:
            kps5_dict = {}
            kps5_dict[0] = QPointF(float(face.kps_5[1, 0]), float(face.kps_5[1, 1]))
            kps5_dict[1] = QPointF(float(face.kps_5[0, 0]), float(face.kps_5[0, 1]))
            kps5_dict[2] = QPointF(float(face.kps_5[2, 0]), float(face.kps_5[2, 1]))
            kps5_dict[3] = QPointF(float(face.kps_5[4, 0]), float(face.kps_5[4, 1]))
            kps5_dict[4] = QPointF(float(face.kps_5[3, 0]), float(face.kps_5[3, 1]))
            self._canvas.set_kps_5(kps5_dict)
            self._canvas.set_show_kps5(self._kps5_toggle_btn.isChecked())
        if face.bbox is not None:
            bx, by, bx2, by2 = face.bbox
            self._canvas.set_bbox_rect(QRectF(bx, by, bx2 - bx, by2 - by))

    def _auto_annotate_silent(self):
        if self._current_img is None:
            return
        from faceswap.core.insightface_adapter import InsightFaceAdapter
        adapter = InsightFaceAdapter.get_instance()
        faces = adapter.detect_faces(self._current_img, max_num=1)
        if not faces:
            return
        self._apply_face_to_canvas(faces[0])

    def _auto_annotate(self):
        if self._current_img is None:
            QMessageBox.warning(self, "提示", "请先加载图片")
            return
        try:
            from faceswap.core.insightface_adapter import InsightFaceAdapter
            adapter = InsightFaceAdapter.get_instance()
            faces = adapter.detect_faces(self._current_img, max_num=0)
            if not faces:
                QMessageBox.information(self, "自动标注失败", "未检测到人脸，请手动标注。")
                return
            self._detected_faces = faces
            if len(faces) == 1:
                self._face_sel_bar.setVisible(False)
                self._apply_face_and_finalize(0)
            else:
                self._show_face_selection_bar(faces)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动标注失败: {e}")

    def _show_face_selection_bar(self, faces):
        while self._face_sel_layout.count():
            item = self._face_sel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._face_btns = []
        label = QLabel(f"检测到 {len(faces)} 张人脸，请选择:")
        label.setStyleSheet("color: #CCCCCC; padding: 2px 6px;")
        self._face_sel_layout.addWidget(label)
        for i, face in enumerate(faces):
            bx, by, bx2, by2 = face.bbox
            w, h = int(bx2 - bx), int(by2 - by)
            btn = QPushButton(f"人脸{i+1} ({w}x{h})")
            btn.clicked.connect(lambda _, idx=i: self._on_face_selected(idx))
            self._face_sel_layout.addWidget(btn)
            self._face_btns.append(btn)
            del_btn = QPushButton("×")
            del_btn.setStyleSheet("QPushButton { background-color: #C0392B; color: white; padding: 3px 6px; font-weight: bold; }"
                                  "QPushButton:hover { background-color: #E74C3C; }")
            del_btn.setToolTip(f"跳过人脸{i+1}")
            del_btn.clicked.connect(lambda _, idx=i: self._on_face_delete(idx))
            self._face_sel_layout.addWidget(del_btn)
        self._face_sel_layout.addStretch()
        self._face_sel_bar.setVisible(True)
        self._apply_face_and_finalize(0)
        self._update_face_btn_highlight(0)
        self._status.setText(f"检测到 {len(faces)} 张人脸，已默认选择人脸1，可点击切换")

    def _apply_face_and_finalize(self, idx):
        if idx < 0 or idx >= len(self._detected_faces):
            return
        self._current_face_idx = idx
        self._apply_face_to_canvas(self._detected_faces[idx])
        if not self._validate_auto_annotate_result():
            return
        self._canvas.set_show_106(True)
        self._canvas.set_visible_group(None)
        self._kps5_saved = True
        self._annotation_step = AnnotationStep.STEP_COMPLETE
        if len(self._detected_faces) == 1:
            self._status.setText("自动标注完成，可拖动调整各点")

    def _update_face_btn_highlight(self, active_idx: int):
        for i, btn in enumerate(self._face_btns):
            if i == active_idx:
                btn.setStyleSheet("QPushButton { background-color: #00BC8C; color: white; padding: 3px 10px; font-weight: bold; }"
                                  "QPushButton:hover { background-color: #00D49A; }")
            else:
                btn.setStyleSheet("QPushButton { background-color: #2D5B8E; color: white; padding: 3px 10px; }"
                                  "QPushButton:hover { background-color: #3D7BAE; }")

    def _on_face_selected(self, idx):
        if idx < 0 or idx >= len(self._detected_faces):
            return
        self._current_face_idx = idx
        self._apply_face_to_canvas(self._detected_faces[idx])
        self._canvas.set_show_106(True)
        self._canvas.set_visible_group(None)
        self._kps5_saved = True
        self._annotation_step = AnnotationStep.STEP_COMPLETE
        self._update_face_btn_highlight(idx)
        self._status.setText(f"已切换到人脸{idx+1}，可拖动调整各点")

    def _on_face_delete(self, idx):
        if idx < 0 or idx >= len(self._detected_faces):
            return
        self._detected_faces.pop(idx)
        if not self._detected_faces:
            self._face_sel_bar.setVisible(False)
            self._canvas.set_landmarks_106({})
            self._canvas.set_kps_5({})
            self._canvas.set_bbox_rect(None)
            self._kps5_saved = False
            self._annotation_step = AnnotationStep.STEP_2_KPS5
            self._status.setText("已跳过所有人脸，可重新点击自动标注或手动标注")
            return
        next_idx = min(idx, len(self._detected_faces) - 1)
        self._show_face_selection_bar(self._detected_faces)
        self._on_face_selected(next_idx)

    def _validate_auto_annotate_result(self) -> bool:
        missing = []
        kps5 = self._canvas.get_kps_5()
        if len(kps5) < 5:
            missing.append("5点检测器")
        lm106 = self._canvas.get_landmarks_106()
        if len(lm106) < 106:
            missing.append(f"106点 (当前: {len(lm106)})")
        if missing:
            QMessageBox.warning(self, "自动标注失败",
                                f"自动标注结果缺少: {', '.join(missing)}\n\n"
                                "请采用正常标注流程：\n"
                                "1. 点击\"生成人脸区域框\"\n"
                                "2. 添加5点检测器并保存\n"
                                "3. 添加106点并保存")
            return False
        return True

    def _apply_to_next(self):
        if self._current_to_annotate_idx < 0 or self._current_to_annotate_idx >= len(self._to_annotate_images) - 1:
            QMessageBox.warning(self, "提示", "没有下一帧")
            return
        lm106 = self._canvas.get_landmarks_106()
        if len(lm106) < 5:
            QMessageBox.warning(self, "提示", "请先标注特征点")
            return
        saved_106 = dict(lm106)
        saved_kps5 = dict(self._canvas.get_kps_5())
        saved_bbox = QRectF(self._canvas.get_bbox_rect()) if self._canvas.get_bbox_rect() is not None else None
        saved_kps5_saved = self._kps5_saved
        self._next_image()
        if saved_106:
            self._canvas.set_landmarks_106(saved_106)
        if saved_kps5:
            self._canvas.set_kps_5(saved_kps5)
            self._kps5_saved = saved_kps5_saved
        if saved_bbox:
            self._canvas.set_bbox_rect(saved_bbox)

    def _delete_current_group(self):
        reply = QMessageBox.question(self, "确认删除", "是否要删除？删除的话则106点、5点和检测框将被全部删除。",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._canvas.clear_landmarks()
            self._canvas._kps_5.clear()
            self._canvas.set_show_kps5(False)
            self._canvas.set_bbox_rect(None)
            self._kps5_saved = False
            self._status.setText("106点、5点和检测框已全部删除，请重新添加")

    def _save_106(self):
        lm106 = self._canvas.get_landmarks_106()

        missing = []
        if not self._kps5_saved or len(self._canvas.get_kps_5()) < 5:
            missing.append("5点检测器")
        if len(lm106) < 106:
            missing.append(f"106点不完整 (当前: {len(lm106)}/106)")
        if missing:
            QMessageBox.warning(self, "无法保存", f"缺少: {', '.join(missing)}")
            return

        self._save_full_annotation()

    def _save_pretrain(self):
        lm106 = self._canvas.get_landmarks_106()
        canvas_kps5 = self._canvas.get_kps_5()
        if len(lm106) < 106 or len(canvas_kps5) < 5:
            QMessageBox.warning(self, "提示", "标注不完整")
            return

        lm_106 = np.zeros((106, 2), dtype=np.float32)
        for idx, pt in lm106.items():
            lm_106[idx] = [pt.x(), pt.y()]
        kps_5 = _canvas_kps5_to_insightface(canvas_kps5)

        source_rect = [0, 0, self._current_img.shape[1], self._current_img.shape[0]]
        bbox_rect = self._canvas.get_bbox_rect()
        if bbox_rect is not None:
            source_rect = [bbox_rect.x(), bbox_rect.y(), bbox_rect.x() + bbox_rect.width(), bbox_rect.y() + bbox_rect.height()]

        canvas_vis = self._canvas.get_lm106_visibility()
        lm106_vis = []
        for i in range(106):
            if i in lm106:
                lm106_vis.append(canvas_vis.get(i, True))
            else:
                lm106_vis.append(True)

        annotation = {
            "landmarks_106": lm_106.astype(np.float32).tolist(),
            "bbox": source_rect,
            "kps_5": kps_5.astype(np.float32).tolist(),
            "landmarks_106_visibility": lm106_vis,
        }

        stem = self._current_img_path.stem
        json_path = self._pretrain_dir / f"{stem}.json"
        json_str = orjson.dumps(annotation)
        FileManager.atomic_write(json_path, json_str)

        saved_name = self._current_img_path.name
        self._status.setText(f"已保存: {saved_name}")

    def _save_full_annotation(self):
        lm106 = self._canvas.get_landmarks_106()
        if len(lm106) < 106:
            return

        if self._data_source.startswith("pretrain:") and self._pretrain_dir is not None:
            self._save_pretrain()
            return

        ft_map = {"whole_face": FaceType.WHOLE_FACE, "head": FaceType.HEAD}
        face_type = ft_map.get(self._ft_combo.currentText(), FaceType.WHOLE_FACE)
        output_size = int(self._size_combo.currentText())

        lm_106 = np.zeros((106, 2), dtype=np.float32)
        for idx, pt in lm106.items():
            lm_106[idx] = [pt.x(), pt.y()]

        canvas_kps5 = self._canvas.get_kps_5()
        if canvas_kps5 and len(canvas_kps5) == 5:
            kps_5 = _canvas_kps5_to_insightface(canvas_kps5)
        else:
            QMessageBox.warning(self, "缺少5点标注", "请先标注5个关键点后再保存")
            return

        canvas_vis = self._canvas.get_lm106_visibility()
        lm106_vis = []
        for i in range(106):
            if i in lm106:
                lm106_vis.append(canvas_vis.get(i, True))
            else:
                lm106_vis.append(True)
        kps5_vis = [True] * 5

        source_rect = [0, 0, self._current_img.shape[1], self._current_img.shape[0]]
        bbox_rect = self._canvas.get_bbox_rect()
        if bbox_rect is not None:
            source_rect = [bbox_rect.x(), bbox_rect.y(), bbox_rect.x() + bbox_rect.width(), bbox_rect.y() + bbox_rect.height()]

        aligned_dir = DATA_SRC_ALIGNED_DIR if self._is_src else DATA_DST_ALIGNED_DIR
        to_annotate_dir = InsightFaceTrainingDataGenerator(WORKSPACE_DIR).to_annotate_dir
        skip_aligned = False
        try:
            if to_annotate_dir.exists() and str(Path(self._current_img_path).parent.resolve()).startswith(str(to_annotate_dir.resolve())):
                skip_aligned = True
        except Exception:
            pass
        existing_face_path = None
        debug_stem = self._current_img_path.stem
        if not skip_aligned and aligned_dir.exists():
            from faceswap.core.metadata_manager import MetadataManager
            source_index = MetadataManager.get_source_index(aligned_dir)
            aligned_names = source_index.get(debug_stem, [])
            if self._current_face_idx < len(aligned_names):
                candidate = aligned_dir / aligned_names[self._current_face_idx]
                if candidate.exists():
                    existing_face_path = candidate

        result = save_face_annotation(
            source_img=self._current_img,
            lm_106=lm_106,
            kps_5=kps_5,
            face_type=face_type,
            output_size=output_size,
            is_src=self._is_src,
            source_filename=self._current_img_path.name,
            source_rect=source_rect,
            landmarks_106_visibility=lm106_vis,
            kps_5_visibility=kps5_vis,
            existing_face_path=existing_face_path,
            skip_aligned=skip_aligned,
            face_idx=self._current_face_idx,
        )

        from faceswap.gui_app.gui_utils import _save_insightface_training_data
        _save_insightface_training_data(result.metadata, self._current_img_path, self._current_img_path.name, face_idx=self._current_face_idx)

        self._annotation_step = AnnotationStep.STEP_COMPLETE
        saved_name = result.face_path.name if result.face_path else self._current_img_path.name
        self._status.setText(f"手动标注完毕: {saved_name}")
        QMessageBox.information(self, "保存成功", f"手动标注完毕\n{saved_name}")

    def _update_progress(self):
        pass

    def load_image(self, img_path: Path):
        self._load_single_image(img_path)


