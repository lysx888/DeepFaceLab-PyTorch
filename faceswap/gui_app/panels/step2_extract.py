import json
import shutil
import sys
import threading
import subprocess
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, QLineEdit,
    QGroupBox, QProgressBar, QFileDialog, QScrollArea, QTextEdit,
    QMessageBox, QDialog, QDialogButtonBox, QButtonGroup,
    QSplitter, QSlider, QFrame, QGridLayout, QProgressDialog,
)
from PyQt6.QtGui import QIntValidator

from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.shared.logger import get_logger
from faceswap.gui_app.gui_utils import install_no_wheel
from faceswap.gui_app.panels.base import (
    StepPanel, AutoScaleLabel, _PreviewSignal, _LogSignal, _ProgressSignal,
    _ClosePreviewSignal, _Worker, _StreamBridge, exec_modal,
)

_logger = get_logger("gui")
from faceswap.gui_app.param_defs import ParamGroup, GROUP_COLORS
from faceswap.setting import (
    WORKSPACE_DIR, MODEL_DIR, XSEG_MODEL_DIR,
    AMP_MODEL_DIR, QUICK96_MODEL_DIR,
    DATA_SRC_DIR, DATA_DST_DIR,
    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR,
    DATA_DST_SWAPPED_DIR, DATA_DST_MERGED_DIR, DATA_DST_MERGED_MASK_DIR,
    IF_LANDMARK_MODEL_DIR, INSIGHTFACE_MANUAL_ANNOTATED_DIR,
    SCRFD_MODEL_DIR, PRETRAIN_DATA_DIR,
    FaceType,
)

class _FaceExtractHalf(QWidget):
    _sig_running = pyqtSignal(bool)
    _sig_index_progress = pyqtSignal(int, int)
    _sig_index_done = pyqtSignal()

    def __init__(self, label: str, is_src: bool, get_config=None, progress_callback=None,
                 check_aligned=None, del_face_cb=None, analyze_cb=None, parent=None):
        super().__init__(parent)
        self._is_src = is_src
        self._get_config = get_config
        self._progress_callback = progress_callback
        self._check_aligned = check_aligned
        self._index_dlg = None
        self._sig_running.connect(self._apply_running)
        self._sig_index_progress.connect(self._on_index_progress)
        self._sig_index_done.connect(self._on_index_done)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        grp = QGroupBox(label)
        grp_layout = QVBoxLayout(grp)
        grp_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("开始提取")
        self._run_btn.setStyleSheet(
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        self._preview_btn = QPushButton("预览生成")
        self._preview_btn.setStyleSheet("QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
        self._preview_btn.clicked.connect(self._on_preview)
        btn_row.addWidget(self._run_btn, 3)
        btn_row.addWidget(self._preview_btn, 3)
        self._extra_btns: list[QPushButton] = []
        if del_face_cb is not None:
            del_btn = QPushButton("删除头像")
            del_btn.setStyleSheet("QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
            del_btn.clicked.connect(del_face_cb)
            btn_row.addWidget(del_btn, 2)
            self._extra_btns.append(del_btn)
        if analyze_cb is not None:
            analyze_btn = QPushButton("分析头像")
            analyze_btn.setStyleSheet("QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
            analyze_btn.clicked.connect(analyze_cb)
            btn_row.addWidget(analyze_btn, 2)
            self._extra_btns.append(analyze_btn)
        grp_layout.addLayout(btn_row)

        layout.addWidget(grp)

    def _apply_running(self, running: bool):
        self._run_btn.setText("提取中..." if running else "开始提取")
        win = self.window()
        if hasattr(win, 'set_busy'):
            win.set_busy(running)

    def _on_index_progress(self, current, total):
        if self._index_dlg is None:
            self._index_dlg = QProgressDialog("正在生成映射文件...", None, 0, total, self)
            self._index_dlg.setWindowTitle("生成映射文件")
            self._index_dlg.setMinimumDuration(0)
            self._index_dlg.show()
        self._index_dlg.setValue(current)

    def _on_index_done(self):
        if self._index_dlg is not None:
            self._index_dlg.close()
            self._index_dlg = None

    def _on_preview(self):
        from faceswap.gui_app.debug_preview_dialog import DebugPreviewDialog
        dlg = DebugPreviewDialog(self._is_src, self)
        exec_modal(self, dlg)

    def _on_run(self):
        if self._check_aligned:
            if not self._check_aligned(self._is_src):
                return

        from faceswap.business.face_extractor import FaceExtractor

        config = self._get_config()
        extractor = FaceExtractor()

        self._sig_running.emit(True)

        def _task():
            try:
                def _index_cb(current, total):
                    self._sig_index_progress.emit(current, total)

                if self._is_src:
                    extractor.extract_src_faces(DATA_SRC_DIR, DATA_SRC_ALIGNED_DIR, config,
                                                progress_callback=self._progress_callback,
                                                index_progress_callback=_index_cb)
                else:
                    extractor.extract_dst_faces(DATA_DST_DIR, DATA_DST_ALIGNED_DIR, config,
                                                progress_callback=self._progress_callback,
                                                index_progress_callback=_index_cb)
            except Exception as e:
                _logger.error(str(e))
            finally:
                self._sig_running.emit(False)
                self._sig_index_done.emit()

        threading.Thread(target=_task, daemon=True).start()


class _ProgressArea(QWidget):
    _sig_progress = pyqtSignal(int, int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sig_progress.connect(self._apply_progress)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        self._progress_bar = QProgressBar()
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("font-size: 11px; color: #666666;")
        layout.addWidget(self._progress_label)

    def _apply_progress(self, current: int, total: int, text: str):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_label.setText(text)

    def on_progress(self, current: int, total: int, elapsed: float, remaining: float, speed: str):
        def _fmt_time(seconds: float) -> str:
            m = int(seconds) // 60
            s = int(seconds) % 60
            return f"{m}:{s:02d}"
        elapsed_str = _fmt_time(elapsed)
        eta_str = _fmt_time(remaining) if remaining > 0 else "--:--"
        text = f"{current}/{total}  [{elapsed_str}<{eta_str}, {speed}]"
        self._sig_progress.emit(current, total, text)


_FACE_TYPE_DEFAULT_SIZE = {
    "whole_face": "512",
    "head": "768",
}


class Step2FaceExtract(StepPanel):
    step_title = "2. 人脸提取"
    step_desc = "从提取的帧画面中检测并对齐人脸。"
    show_run_buttons = False
    _sig_sort_index_progress = pyqtSignal(int, int)
    _sig_sort_index_done = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    def _build_params(self):
        self._sort_progress_sig = _ProgressSignal()
        self._sort_progress_sig.done_ready.connect(lambda t, m: (self._set_sort_buttons_enabled(True), QMessageBox.information(self, t, m)))
        self._sort_progress_sig.error_ready.connect(lambda t, m: (self._set_sort_buttons_enabled(True), QMessageBox.warning(self, t, m)))
        self._sort_index_dlg = None
        self._sig_sort_index_progress.connect(self._on_sort_index_progress)
        self._sig_sort_index_done.connect(self._on_sort_index_done)

        from PyQt6.QtWidgets import QSizePolicy
        params_row = QHBoxLayout()
        params_row.setSpacing(12)

        def _add_param(label: str, widget) -> None:
            box = QVBoxLayout()
            box.setSpacing(2)
            lbl = QLabel(label)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            box.addWidget(lbl)
            box.addWidget(widget)
            params_row.addLayout(box, 1)

        self._face_type = QComboBox()
        self._face_type.addItems(["whole_face", "head"])
        self._face_type.setCurrentText("whole_face")
        self._face_type.currentTextChanged.connect(self._on_face_type_changed)
        _add_param("人脸类型", self._face_type)

        self._max_faces = QSpinBox()
        self._max_faces.setRange(0, 100)
        self._max_faces.setValue(0)
        self._max_faces.setToolTip("0=不限")
        _add_param("最大人脸数", self._max_faces)

        self._det_thresh = QDoubleSpinBox()
        self._det_thresh.setRange(0.1, 1.0)
        self._det_thresh.setValue(0.5)
        self._det_thresh.setSingleStep(0.05)
        self._det_thresh.setDecimals(2)
        _add_param("检测阈值", self._det_thresh)

        self._output_size = QComboBox()
        self._output_size.addItems(["512", "768", "384", "256", "640", "1024", "128", "896"])
        self._output_size.setCurrentText("512")
        self._output_size.setEditable(True)
        self._output_size.setValidator(QIntValidator(64, 4096, self))
        _add_param("图像尺寸", self._output_size)

        self._output_format = QComboBox()
        self._output_format.addItems(["jpg", "png"])
        self._output_format.setCurrentText("jpg")
        _add_param("输出格式", self._output_format)

        self._jpg_quality = QSpinBox()
        self._jpg_quality.setRange(1, 100)
        self._jpg_quality.setValue(100)
        _add_param("JPEG质量", self._jpg_quality)

        self._params_area.addLayout(params_row)

        self._debug = self._add_check("输出调试图像到 aligned_debug", True)

        self._progress = _ProgressArea()

        row = QHBoxLayout()
        self._src_half = _FaceExtractHalf(
            "源人脸 (SRC)", is_src=True, get_config=self._get_config,
            progress_callback=self._progress.on_progress,
            check_aligned=self._check_aligned_dir,
            del_face_cb=lambda: self._on_delete_faces(True),
            analyze_cb=lambda: self._on_analyze_faces(True))
        self._dst_half = _FaceExtractHalf(
            "目标人脸 (DST)", is_src=False, get_config=self._get_config,
            progress_callback=self._progress.on_progress,
            check_aligned=self._check_aligned_dir,
            del_face_cb=lambda: self._on_delete_faces(False),
            analyze_cb=lambda: self._on_analyze_faces(False))
        row.addWidget(self._src_half)
        row.addWidget(self._dst_half)
        self._params_area.addLayout(row)

        self._params_area.addWidget(self._progress)

        src_tool_grp = QGroupBox("源目标 (SRC) 工具栏")
        src_tool_lay = QHBoxLayout(src_tool_grp)
        src_tool_lay.setContentsMargins(8, 14, 8, 4)
        src_tool_lay.setSpacing(8)
        dedup_btn = QPushButton("去重过滤")
        dedup_btn.setStyleSheet("QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
        dedup_btn.clicked.connect(self._on_dedup)
        src_tool_lay.addWidget(dedup_btn)
        self._rename_btn = QPushButton("批量重命名")
        self._rename_btn.setMinimumWidth(100)
        self._rename_btn.clicked.connect(self._on_rename)
        src_tool_lay.addWidget(self._rename_btn)
        src_tool_lay.addStretch()
        rename_desc = QLabel("去重: 3DDFA 状态去冗余  |  重命名: 连续编号")
        rename_desc.setStyleSheet("color: #666; font-size: 11px;")
        src_tool_lay.addWidget(rename_desc)
        self._params_area.addWidget(src_tool_grp)

        tool_grp = QGroupBox("标注与训练")
        tool_lay = QHBoxLayout(tool_grp)
        tool_lay.setContentsMargins(8, 14, 8, 4)
        tool_lay.setSpacing(8)
        manual_btn = QPushButton("手动标注")
        manual_btn.setStyleSheet("QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
        manual_btn.clicked.connect(self._on_manual_annotate)
        tool_lay.addWidget(manual_btn)
        xseg_train_btn = QPushButton("IF训练")
        xseg_train_btn.setStyleSheet("QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 5px 14px; border-radius: 3px; }")
        xseg_train_btn.clicked.connect(self._on_goto_if_train)
        tool_lay.addWidget(xseg_train_btn)
        tool_lay.addStretch()
        tool_desc = QLabel("手动标注 XSeg 遮罩 / 跳转到 IF (InsightFace) 训练页")
        tool_desc.setStyleSheet("color: #666; font-size: 11px;")
        tool_lay.addWidget(tool_desc)
        self._params_area.addWidget(tool_grp)

        sort_grp = QGroupBox("排序筛选")
        sort_outer = QVBoxLayout(sort_grp)
        sort_outer.setContentsMargins(8, 14, 8, 4)
        sort_outer.setSpacing(4)

        target_row = QHBoxLayout()
        target_row.setSpacing(8)
        self._sort_src_rb = QRadioButton("源人脸 (SRC)")
        self._sort_src_rb.setChecked(True)
        self._sort_dst_rb = QRadioButton("目标人脸 (DST)")
        sort_run_btn = QPushButton("排序")
        sort_run_btn.setMinimumWidth(60)
        sort_run_btn.setStyleSheet(
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        sort_run_btn.clicked.connect(self._on_sort)
        target_row.addWidget(self._sort_src_rb)
        target_row.addWidget(self._sort_dst_rb)
        target_row.addStretch()
        target_row.addWidget(sort_run_btn)
        sort_outer.addLayout(target_row)

        algo_row = QHBoxLayout()
        algo_row.setSpacing(2)
        self._sort_algo_btns = []
        _ALGO_BTN_STYLE = (
            "QPushButton { background-color: #FFFFFF; border: 1px solid #0078D4; border-radius: 3px; padding: 4px 8px; font-size: 11px; color: #0078D4; }"
            "QPushButton:hover { background-color: #E8F0FE; }"
            "QPushButton:checked { background-color: #0078D4; color: white; border: 1px solid #0078D4; }"
        )
        for algo in ["blur", "hist", "yaw", "pitch", "brightness", "hue", "oneface", "final"]:
            btn = QPushButton(algo)
            btn.setCheckable(True)
            btn.setStyleSheet(_ALGO_BTN_STYLE)
            btn.clicked.connect(lambda checked, a=algo: self._set_sort_algo(a))
            self._sort_algo_btns.append((algo, btn))
            algo_row.addWidget(btn, 1)
        self._sort_algo_btns[0][1].setChecked(True)
        algo_row.addStretch()
        sort_outer.addLayout(algo_row)

        self._sort_desc = QLabel()
        self._sort_desc.setStyleSheet("color: #555; font-size: 13px; padding-left: 4px;")
        self._sort_desc.setWordWrap(True)
        sort_outer.addWidget(self._sort_desc)
        self._on_sort_algo_changed("blur")

        self._sort_progress = QLabel()
        self._sort_progress.setStyleSheet("color: #0078D4; font-size: 12px; padding-left: 4px;")
        sort_outer.addWidget(self._sort_progress)
        self._sort_progress_sig.progress_ready.connect(self._sort_progress.setText)

        self._params_area.addWidget(sort_grp)

    def _on_face_type_changed(self, text: str):
        default_size = _FACE_TYPE_DEFAULT_SIZE.get(text, "512")
        idx = self._output_size.findText(default_size)
        if idx >= 0:
            self._output_size.setCurrentIndex(idx)
        else:
            self._output_size.setCurrentText(default_size)

    def _on_manual_annotate(self):
        from faceswap.gui_app.manual_annotator_dialog import ManualAnnotatorDialog
        dlg = ManualAnnotatorDialog(self)
        exec_modal(self, dlg)

    def _on_goto_if_train(self):
        w = self.window()
        if hasattr(w, '_switch'):
            w._switch(6)

    def _on_dedup(self):
        from faceswap.gui_app.dedup_dialog import SrcDedupDialog
        dlg = SrcDedupDialog(DATA_SRC_ALIGNED_DIR, parent=self)
        exec_modal(self, dlg)

    def _set_sort_algo(self, algo: str):
        for a, btn in self._sort_algo_btns:
            btn.setChecked(a == algo)
        self._on_sort_algo_changed(algo)

    _SORT_DESCRIPTIONS = {
        "blur": "按模糊度排序：将模糊的图像排到后面，方便删除低质量人脸",
        "hist": "按直方图相似度排序：将相似的人脸排在一起，方便批量筛选",
        "yaw": "按偏航角排序：按人脸左右旋转角度排列",
        "pitch": "按俯仰角排序：按人脸上下旋转角度排列",
        "brightness": "按亮度排序：将过暗或过亮的人脸排到后面",
        "hue": "按色调排序：按肤色色调分组排列",
        "oneface": "单人筛选：只保留每张图中最大的人脸，删除多余人脸",
        "final": "最终排序：综合多种因素进行最终排序筛选",
    }

    def _on_sort_algo_changed(self, text: str):
        desc = self._SORT_DESCRIPTIONS.get(text, "")
        self._sort_desc.setText(desc)

    def _on_delete_faces(self, is_src: bool):
        from faceswap.gui_app.face_delete_dialog import FaceDeleteDialog
        target_dir = DATA_SRC_ALIGNED_DIR if is_src else DATA_DST_ALIGNED_DIR
        has_img = next(target_dir.glob("*.jpg"), None) is not None or next(target_dir.glob("*.png"), None) is not None
        if not target_dir.exists() or not has_img:
            QMessageBox.information(self, "提示", f"目录中没有图片:\n{target_dir}")
            return
        dlg = FaceDeleteDialog(target_dir, parent=self)
        exec_modal(self, dlg)

    def _on_analyze_faces(self, is_src: bool):
        from faceswap.gui_app.dedup_dialog import StateAnalysisDialog
        target_dir = DATA_SRC_ALIGNED_DIR if is_src else DATA_DST_ALIGNED_DIR
        if not target_dir.exists():
            QMessageBox.information(self, "提示", f"目录不存在:\n{target_dir}")
            return
        dlg = StateAnalysisDialog(target_dir, yaw_grid=5.0, pitch_grid=5.0, parent=self)
        exec_modal(self, dlg)

    def _on_sort(self):
        from faceswap.business.face_sorter import FaceSorter, SortAlgorithm

        target_dir = DATA_SRC_ALIGNED_DIR if self._sort_src_rb.isChecked() else DATA_DST_ALIGNED_DIR
        algo_name = "blur"
        for a, btn in self._sort_algo_btns:
            if btn.isChecked():
                algo_name = a
                break
        algo = SortAlgorithm(algo_name)
        sorter = FaceSorter()
        sig = self._sort_progress_sig
        self._sort_progress.setText("Sorting: 0%")
        self._set_sort_buttons_enabled(False)

        def _progress(current, total, elapsed, remaining=0, speed=""):
            if total > 0:
                pct = int(current / total * 100)
                m_e = int(elapsed) // 60
                s_e = int(elapsed) % 60
                m_r = int(remaining) // 60 if remaining > 0 else 0
                s_r = int(remaining) % 60 if remaining > 0 else 0
                sig.progress_ready.emit(f"Sorting: {pct}% | {current}/{total} [{m_e}:{s_e:02d}<{m_r}:{s_r:02d}, {speed}]")

        def _task():
            try:
                def _index_cb(current, total):
                    self._sig_sort_index_progress.emit(current, total)

                count = sorter.sort_aligned(target_dir, algo, progress_callback=_progress,
                                           index_progress_callback=_index_cb)
                sig.done_ready.emit("完成", f"排序完成，共处理 {count} 张人脸")
            except Exception as e:
                sig.error_ready.emit("错误", str(e))
            finally:
                sig.progress_ready.emit("")
                self._sig_sort_index_done.emit()

        threading.Thread(target=_task, daemon=True).start()

    def _set_sort_buttons_enabled(self, enabled: bool):
        # 全局禁用统一管理：运行期间禁所有按钮，结束/失败时全部还原
        win = self.window()
        if hasattr(win, 'set_busy'):
            win.set_busy(not enabled)

    def _on_sort_index_progress(self, current, total):
        if self._sort_index_dlg is None:
            self._sort_index_dlg = QProgressDialog("正在生成映射文件...", None, 0, total, self)
            self._sort_index_dlg.setWindowTitle("生成映射文件")
            self._sort_index_dlg.setMinimumDuration(0)
            self._sort_index_dlg.show()
        self._sort_index_dlg.setValue(current)

    def _on_sort_index_done(self):
        if self._sort_index_dlg is not None:
            self._sort_index_dlg.close()
            self._sort_index_dlg = None

    def _on_rename(self):
        from faceswap.shared.file_manager import FileManager
        target_dir = DATA_SRC_ALIGNED_DIR
        if not target_dir.exists():
            QMessageBox.warning(self, "错误", f"目录不存在: {target_dir}")
            return

        all_paths = FileManager.find_images(target_dir)
        if not all_paths:
            QMessageBox.information(self, "提示", "目录中没有图像文件")
            return

        all_paths.sort(key=lambda p: p.name)
        count = len(all_paths)
        sig = self._sort_progress_sig
        self._set_sort_buttons_enabled(False)

        def _task():
            import os
            try:
                for stale in target_dir.glob("__tmp_*"):
                    stale.unlink()
                for i, img_path in enumerate(all_paths):
                    ext = img_path.suffix
                    os.rename(str(img_path), str(target_dir / f"__tmp_{i:05d}{ext}"))
                    meta_path = img_path.with_suffix(".json")
                    if meta_path.exists():
                        os.rename(str(meta_path), str(target_dir / f"__tmp_{i:05d}.json"))
                for i in range(len(all_paths)):
                    ext = all_paths[i].suffix
                    os.rename(str(target_dir / f"__tmp_{i:05d}{ext}"), str(target_dir / f"{i + 1:05d}_0{ext}"))
                    tmp_meta = target_dir / f"__tmp_{i:05d}.json"
                    if tmp_meta.exists():
                        os.rename(str(tmp_meta), str(target_dir / f"{i + 1:05d}_0.json"))
                sig.done_ready.emit("完成", f"已重命名 {count} 个源人脸文件")
            except Exception as e:
                sig.error_ready.emit("重命名错误", str(e))

        threading.Thread(target=_task, daemon=True).start()

    def _check_aligned_dir(self, is_src: bool) -> bool:
        aligned_dir = DATA_SRC_ALIGNED_DIR if is_src else DATA_DST_ALIGNED_DIR
        if aligned_dir.exists():
            from faceswap.shared.file_manager import FileManager
            files = FileManager.find_images(aligned_dir)
            if files:
                tag = "源" if is_src else "目标"
                reply = QMessageBox.warning(
                    self, "目录非空",
                    f"{tag}人脸目录 ({aligned_dir}) 中已有 {len(files)} 个文件。\n"
                    f"继续提取将覆盖已有文件，是否继续？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return False
        return True

    def _get_config(self):
        from faceswap.business.face_extractor import ExtractConfig


        ft_map = {"whole_face": FaceType.WHOLE_FACE, "head": FaceType.HEAD}
        ft = ft_map.get(self._face_type.currentText(), FaceType.WHOLE_FACE)

        config = ExtractConfig(
            face_type=ft,
            max_faces=self._max_faces.value(),
            det_thresh=self._det_thresh.value(),
            output_size=int(self._output_size.currentText()),
            jpg_quality=self._jpg_quality.value(),
            output_format=self._output_format.currentText(),
            debug_output=self._debug.isChecked(),
        )
        return config

