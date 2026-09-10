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
    QSplitter, QSlider, QFrame, QGridLayout,
)

from faceswap.shared.image_utils import bgr_to_rgb
from faceswap.shared.logger import get_logger
from faceswap.gui_app.gui_utils import install_no_wheel
from faceswap.gui_app.panels.base import (
    StepPanel, AutoScaleLabel, _PreviewSignal, _LogSignal, _ProgressSignal,
    _ClosePreviewSignal, _Worker, _StreamBridge,
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

class Step3XSeg(StepPanel):
    step_title = "3. 遮罩 XSeg"
    step_desc = "编辑、训练和应用XSeg人脸遮罩。"
    show_run_buttons = False

    def _build_params(self):
        self._xseg_action_btns: list = []
        grp_edit = QGroupBox("遮罩编辑")
        edit_lay = QVBoxLayout(grp_edit)
        edit_lay.setSpacing(4)

        target_row = QHBoxLayout()
        self._edit_src_btn = QRadioButton("源人脸（data_src）")
        self._edit_src_btn.setChecked(True)
        self._edit_dst_btn = QRadioButton("目标人脸（data_dst）")
        target_row.addWidget(self._edit_src_btn)
        target_row.addWidget(self._edit_dst_btn)
        target_row.addStretch()
        edit_lay.addLayout(target_row)

        edit_btn_row = QHBoxLayout()
        edit_btn = QPushButton("打开遮罩编辑器")
        edit_btn.setStyleSheet(
            "QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 6px 20px; border-radius: 4px; }"
        )
        edit_btn.clicked.connect(self._on_edit)
        fetch_btn = QPushButton("提取已标注")
        fetch_btn.setProperty("outline", True)
        fetch_btn.clicked.connect(self._on_fetch)
        remove_btn = QPushButton("清除标注")
        remove_btn.setProperty("danger", True)
        remove_btn.clicked.connect(self._on_remove_annotations)
        edit_btn_row.addWidget(edit_btn, 1)
        edit_btn_row.addWidget(fetch_btn, 1)
        edit_btn_row.addWidget(remove_btn, 1)
        self._xseg_action_btns += [edit_btn, fetch_btn, remove_btn]
        edit_lay.addLayout(edit_btn_row)

        self._params_area.addWidget(grp_edit)

        grp_train = QGroupBox("遮罩训练")
        train_lay = QVBoxLayout(grp_train)
        train_lay.setSpacing(4)

        train_row = QHBoxLayout()
        lbl_ft = QLabel("脸部类型:")
        self._xseg_face_type = QComboBox()
        self._xseg_face_type.addItems(["wf", "head"])
        self._xseg_face_type.setCurrentText("wf")
        self._xseg_face_type.setFixedWidth(70)
        _tip_ft = "人脸裁切类型：wf=全脸(256×256)，head=含额头头发(384×384)。应与提取时一致"
        lbl_ft.setToolTip(_tip_ft)
        self._xseg_face_type.setToolTip(_tip_ft)
        lbl3 = QLabel("批次:")
        self._xseg_batch = QSpinBox()
        self._xseg_batch.setRange(1, 64)
        self._xseg_batch.setValue(4)
        self._xseg_batch.setFixedWidth(50)
        _tip_bs = "每步训练的样本数。越大训练越稳定但显存占用越高。RTX 4090建议4-8"
        lbl3.setToolTip(_tip_bs)
        self._xseg_batch.setToolTip(_tip_bs)
        lbl4 = QLabel("迭代:")
        self._xseg_iters = QSpinBox()
        self._xseg_iters.setRange(0, 10000000)
        self._xseg_iters.setValue(0)
        self._xseg_iters.setSingleStep(10000)
        self._xseg_iters.setFixedWidth(90)
        _tip_it = "总训练迭代次数。0=无限训练(手动停止,与DFL一致)，遮罩训练通常5-10万次即可收敛"
        lbl4.setToolTip(_tip_it)
        self._xseg_iters.setToolTip(_tip_it)
        lbl5 = QLabel("混合精度:")
        self._xseg_amp = QComboBox()
        self._xseg_amp.addItems(["fp32", "fp16", "bf16"])
        self._xseg_amp.setCurrentText("fp16")
        self._xseg_amp.setFixedWidth(75)
        _tip_amp = "混合精度训练模式:\nfp16: 半精度+GradScaler, 省显存加速(推荐)\nbf16: BF16半精度, 精度不足易导致梯度爆炸, 不推荐\nfp32: 全精度, 最稳定但最慢"
        lbl5.setToolTip(_tip_amp)
        self._xseg_amp.setToolTip(_tip_amp)
        lbl_lr = QLabel("学习率:")
        self._xseg_lr = QDoubleSpinBox()
        self._xseg_lr.setRange(1e-6, 1e-2)
        self._xseg_lr.setValue(1e-4)
        self._xseg_lr.setSingleStep(1e-5)
        self._xseg_lr.setDecimals(6)
        self._xseg_lr.setFixedWidth(90)
        _tip_lr = "优化器学习率。DFL默认1e-4。值太大训练不稳定，太小收敛慢。通常1e-4~5e-5"
        lbl_lr.setToolTip(_tip_lr)
        self._xseg_lr.setToolTip(_tip_lr)
        lbl6 = QLabel("预训练:")
        self._xseg_pretrain = QCheckBox()
        self._xseg_pretrain.setChecked(False)
        _tip_pt = "先启用DSSIM+MSE灰度图自重建预训练(skip=zeros_like)，让模型学会人脸结构后再用BCE微调mask分割。建议首次训练开启"
        lbl6.setToolTip(_tip_pt)
        self._xseg_pretrain.setToolTip(_tip_pt)
        lbl7 = QLabel("lr_dropout:")
        self._xseg_lr_dropout = QDoubleSpinBox()
        self._xseg_lr_dropout.setRange(0.0, 1.0)
        self._xseg_lr_dropout.setValue(0.3)
        self._xseg_lr_dropout.setSingleStep(0.1)
        self._xseg_lr_dropout.setFixedWidth(65)
        _tip_ld = "学习率随机丢弃率。每步以该概率将梯度置零，相当于随机正则化。0=关闭，0.3=DFL默认。训练后期开启可获更锐利结果"
        lbl7.setToolTip(_tip_ld)
        self._xseg_lr_dropout.setToolTip(_tip_ld)
        train_row.addWidget(lbl_ft)
        train_row.addWidget(self._xseg_face_type)
        train_row.addSpacing(10)
        train_row.addWidget(lbl3)
        train_row.addWidget(self._xseg_batch)
        train_row.addSpacing(10)
        train_row.addWidget(lbl4)
        train_row.addWidget(self._xseg_iters)
        train_row.addSpacing(10)
        train_row.addWidget(lbl5)
        train_row.addWidget(self._xseg_amp)
        train_row.addSpacing(10)
        train_row.addWidget(lbl_lr)
        train_row.addWidget(self._xseg_lr)
        train_row.addSpacing(10)
        train_row.addWidget(lbl7)
        train_row.addWidget(self._xseg_lr_dropout)
        train_row.addSpacing(10)
        train_row.addWidget(lbl6)
        train_row.addWidget(self._xseg_pretrain)
        train_row.addStretch()
        train_lay.addLayout(train_row)

        for w in [self._xseg_face_type, self._xseg_batch, self._xseg_iters,
                  self._xseg_amp, self._xseg_lr, self._xseg_lr_dropout]:
            install_no_wheel(w)

        train_btn_row = QHBoxLayout()
        self._train_btn = QPushButton("开始训练")
        self._train_btn.setStyleSheet(
            "QPushButton { background-color: #D45500; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background-color: #E06010; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._train_btn.clicked.connect(self._on_train)
        self._stop_train_btn = QPushButton("停止训练")
        self._stop_train_btn.setProperty("busy_keep", True)
        self._stop_train_btn.setStyleSheet(
            "QPushButton { background-color: #C42B1C; color: white; font-weight: bold; "
            "padding: 6px 16px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background-color: #D43B2C; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }"
        )
        self._stop_train_btn.setEnabled(False)
        self._stop_train_btn.clicked.connect(self._on_stop_train)
        train_btn_row.addStretch()
        train_btn_row.addWidget(self._train_btn)
        train_btn_row.addWidget(self._stop_train_btn)
        train_lay.addLayout(train_btn_row)

        self._train_log = QTextEdit()
        self._train_log.setReadOnly(True)
        self._train_log.setMinimumHeight(100)
        self._train_log.setMaximumHeight(200)
        self._train_log.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; background-color: #1e1e1e; color: #d4d4d4;")
        train_lay.addWidget(self._train_log)

        self._params_area.addWidget(grp_train)

        grp_apply = QGroupBox("遮罩应用")
        apply_lay = QVBoxLayout(grp_apply)
        apply_lay.setSpacing(4)

        apply_target_row = QHBoxLayout()
        self._apply_src_btn = QRadioButton("源人脸（data_src）")
        self._apply_src_btn.setChecked(True)
        self._apply_dst_btn = QRadioButton("目标人脸（data_dst）")
        apply_target_row.addWidget(self._apply_src_btn)
        apply_target_row.addWidget(self._apply_dst_btn)
        apply_target_row.addStretch()
        apply_lay.addLayout(apply_target_row)

        apply_row = QHBoxLayout()
        apply_trained_btn = QPushButton("应用训练遮罩")
        apply_trained_btn.clicked.connect(self._on_apply_trained)
        remove_trained_btn = QPushButton("移除训练遮罩")
        remove_trained_btn.setProperty("danger", True)
        remove_trained_btn.clicked.connect(self._on_remove_trained)
        apply_generic_btn = QPushButton("应用通用遮罩")
        apply_generic_btn.setProperty("outline", True)
        apply_generic_btn.clicked.connect(self._on_apply_generic)
        apply_row.addWidget(apply_trained_btn)
        apply_row.addWidget(remove_trained_btn)
        apply_row.addWidget(apply_generic_btn)
        self._xseg_action_btns += [apply_trained_btn, remove_trained_btn, apply_generic_btn]
        apply_lay.addLayout(apply_row)

        self._apply_status = QLabel("")
        self._apply_status.setStyleSheet("color: #888; font-size: 11px; font-family: Consolas, monospace;")
        apply_lay.addWidget(self._apply_status)

        self._params_area.addWidget(grp_apply)

        self._xseg_trainer = None
        self._preview_win = None
        self._preview_signal = _PreviewSignal()
        self._preview_signal.preview_ready.connect(self._show_preview)
        self._log_signal = _LogSignal()
        self._log_signal.log_ready.connect(self._append_log)
        self._progress_signal = _ProgressSignal()
        self._progress_signal.progress_ready.connect(self._apply_status.setText)
        self._progress_signal.done_ready.connect(lambda t, m: QMessageBox.information(self, t, m))
        self._progress_signal.error_ready.connect(lambda t, m: QMessageBox.critical(self, t, m))
        self._close_preview_signal = _ClosePreviewSignal()
        self._close_preview_signal.close_ready.connect(self._close_preview)
        self._load_xseg_config()

    def _load_xseg_config(self):
        config_path = Path(XSEG_MODEL_DIR) / "XSeg_config.json"
        if not config_path.exists():
            return
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            self._xseg_face_type.setCurrentText(cfg.get("face_type", "wf"))
            self._xseg_batch.setValue(cfg.get("batch_size", 4))
            self._xseg_iters.setValue(cfg.get("target_iter", 0))
            self._xseg_amp.setCurrentText(cfg.get("amp_mode", "fp16"))
            self._xseg_lr.setValue(cfg.get("learning_rate", 1e-4))
            self._xseg_lr_dropout.setValue(cfg.get("lr_dropout", 0.3))
            pretrain = cfg.get("pretrain", None)
            if pretrain is not None:
                self._xseg_pretrain.setChecked(pretrain)
                if pretrain is False:
                    self._xseg_pretrain.setEnabled(False)
                    self._xseg_pretrain.setToolTip("已退出预训练模式，不可回退")
        except Exception:
            pass

    def _close_preview(self):
        if self._preview_win is not None:
            self._preview_win.close()
            self._preview_win = None

    def _set_xseg_controls_enabled(self, enabled: bool):
        self._train_btn.setEnabled(enabled)
        self._stop_train_btn.setEnabled(not enabled)
        for w in [self._xseg_face_type, self._xseg_batch, self._xseg_iters,
                  self._xseg_amp, self._xseg_lr, self._xseg_lr_dropout,
                  self._xseg_pretrain]:
            w.setEnabled(enabled)
        for w in getattr(self, "_xseg_action_btns", []):
            w.setEnabled(enabled)
        if enabled:
            config_path = Path(XSEG_MODEL_DIR) / "XSeg_config.json"
            if config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text(encoding="utf-8"))
                    if cfg.get("pretrain") is False:
                        self._xseg_pretrain.setEnabled(False)
                        self._xseg_pretrain.setToolTip("已退出预训练模式，不可回退")
                except Exception:
                    pass

    def _set_running(self, running: bool):
        self._running = running

    def _get_edit_target_dir(self) -> Path:
        return DATA_SRC_ALIGNED_DIR if self._edit_src_btn.isChecked() else DATA_DST_ALIGNED_DIR

    def _get_apply_target_dir(self) -> Path:
        return DATA_SRC_ALIGNED_DIR if self._apply_src_btn.isChecked() else DATA_DST_ALIGNED_DIR

    def _show_preview(self, preview_bgr: np.ndarray):
        import cv2
        from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout, QSizePolicy
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtCore import Qt

        if self._preview_win is None:
            self._preview_win = QDialog(self)
            self._preview_win.setWindowTitle("XSeg 训练预览 | [space]:切换 [p]:刷新 [s]:保存 [l]:历史范围 [Enter]:保存并停止")
            self._preview_win.setWindowFlags(self._preview_win.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
            lay = QVBoxLayout(self._preview_win)
            lay.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            lay.addWidget(lbl)
            self._preview_win._label = lbl
            self._preview_win._orig_pixmap = None
            self._preview_win.resize(600, 800)
            self._preview_win.keyPressEvent = lambda e: self._on_preview_key(e)
            original_close = self._preview_win.close
            self._preview_win.closeEvent = lambda e: self._on_preview_close(e, original_close)
            original_resize = self._preview_win.resizeEvent
            def _on_resize(e, orig=original_resize):
                orig(e)
                self._rescale_xseg_preview_pixmap()
            self._preview_win.resizeEvent = _on_resize
            self._preview_win.show()

        if not self._preview_win.isVisible():
            self._preview_win.show()

        rgb = bgr_to_rgb(preview_bgr)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self._preview_win._orig_pixmap = pix

        try:
            screen = self._preview_win.screen()
            max_w = int(screen.availableGeometry().width() * 0.85)
            max_h = int(screen.availableGeometry().height() * 0.85)
        except Exception:
            max_w, max_h = 1400, 900
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            win_w = int(w * scale)
            win_h = int(h * scale)
        else:
            win_w = w
            win_h = h
        self._preview_win.resize(win_w, win_h)

        self._rescale_xseg_preview_pixmap()

    def _append_log(self, line: str, overwrite: bool = False):
        te = self._train_log
        if overwrite:
            cursor = te.textCursor()
            cursor.beginEditBlock()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(line)
            cursor.endEditBlock()
            te.setTextCursor(cursor)
        else:
            te.append(line)
        sb = te.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _rescale_xseg_preview_pixmap(self):
        if self._preview_win is None:
            return
        pix = self._preview_win._orig_pixmap
        if pix is None:
            return
        lbl_size = self._preview_win._label.size()
        if lbl_size.width() < 1 or lbl_size.height() < 1:
            return
        from PyQt6.QtCore import Qt
        scaled = pix.scaled(lbl_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._preview_win._label.setPixmap(scaled)

    def _on_preview_key(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            if self._xseg_trainer is not None:
                self._xseg_trainer._preview_page += 1
                self._xseg_trainer.request_preview()
        elif key == Qt.Key.Key_P:
            if self._xseg_trainer is not None:
                self._xseg_trainer.request_preview()
        elif key == Qt.Key.Key_S:
            if self._xseg_trainer is not None:
                self._xseg_trainer.request_save()
                self._log_signal.log_ready.emit("[save] Checkpoint saved.", False)
        elif key == Qt.Key.Key_L:
            if self._xseg_trainer is not None:
                self._xseg_trainer.cycle_loss_range()
                self._xseg_trainer.request_preview()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._stop_and_close_preview()

    def _on_preview_close(self, event, original_close):
        self._stop_and_close_preview()
        event.accept()

    def _stop_and_close_preview(self):
        self._request_training_stop(self._xseg_trainer, "enter", self._log_signal)

    def _on_edit(self):
        from faceswap.gui_app.xseg_editor_dialog import XSegEditorDialog
        d = self._get_edit_target_dir()
        # 遮罩编辑器打开期间禁用主窗口（防重复多开/误操作），
        # 不依赖 exec() 模态（对话框先 showMaximized 后 exec 的模态可能不生效）。
        self._set_busy(True)
        try:
            dlg = XSegEditorDialog(d, self)
            dlg.exec()
        finally:
            self._set_busy(False)

    def _on_fetch(self):
        from faceswap.business.xseg_editor import XSegEditor
        d = self._get_edit_target_dir()
        editor = XSegEditor()
        count = editor.fetch_annotated(d)
        QMessageBox.information(self, "完成", f"已提取 {count} 个已标注人脸")

    def _on_remove_annotations(self):
        d = self._get_edit_target_dir()
        reply = QMessageBox.warning(
            self, "确认", f"确定要清除 {d} 中所有XSeg标注吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        from faceswap.business.xseg_editor import XSegEditor
        editor = XSegEditor()
        sig = self._progress_signal
        sig.progress_ready.emit("Processing...")

        def _on_progress(current, total, elapsed):
            if total > 0:
                eta = elapsed / current * (total - current) if current > 0 else 0
                rate = current / elapsed if elapsed > 0 else 0
                sig.progress_ready.emit(f"Processing: {current}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")

        def _task():
            try:
                count = editor.remove_annotations(d, progress_callback=_on_progress)
                sig.progress_ready.emit(f"Done: {count} faces")
                sig.done_ready.emit("完成", f"已清除 {count} 张人脸的XSeg标注")
            except Exception as e:
                sig.progress_ready.emit(f"Error: {e}")
                sig.error_ready.emit("错误", str(e))

        self._run_in_thread(_task)

    def _on_train(self):
        from faceswap.business.xseg_trainer import XSegTrainer
        self._xseg_trainer = XSegTrainer()
        self._set_xseg_controls_enabled(False)
        self._train_log.clear()

        self._log_throttle_time = 0.0
        self._log_throttle_interval = 0.5
        self._log_need_newline = True
        self._loss_smooth_buffer = deque(maxlen=100)

        def _on_iter(iter_count, loss_val, iter_ms):
            import time as _time
            now = _time.time()
            if now - self._log_throttle_time < self._log_throttle_interval:
                return
            self._log_throttle_time = now
            self._loss_smooth_buffer.append(loss_val)
            smoothed = sum(self._loss_smooth_buffer) / len(self._loss_smooth_buffer)
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            if iter_ms >= 1000:
                time_str = f"{iter_ms/1000:.1f}s"
            else:
                time_str = f"{int(iter_ms)}ms"
            line = f"[{ts}][#{iter_count}][{time_str}][loss {smoothed:.5f}]"

            overwrite = not self._log_need_newline
            self._log_need_newline = False
            self._log_signal.log_ready.emit(line, overwrite)

        def _on_preview(preview_bgr):
            self._preview_signal.preview_ready.emit(preview_bgr)

        def _on_save(iter_count):
            from datetime import datetime
            ts = datetime.now().strftime("%H:%M:%S")
            self._log_signal.log_ready.emit(f"[{ts}][#{iter_count}] saved", False)
            self._log_need_newline = False

        def _task():
            try:
                self._xseg_trainer.train(
                    DATA_SRC_ALIGNED_DIR, DATA_DST_ALIGNED_DIR, XSEG_MODEL_DIR,
                    batch_size=self._xseg_batch.value(),
                    target_iter=self._xseg_iters.value(),
                    face_type=self._xseg_face_type.currentText(),
                    learning_rate=self._xseg_lr.value(),
                    amp_mode=self._xseg_amp.currentText(),
                    pretrain=self._xseg_pretrain.isChecked(),
                    lr_dropout=self._xseg_lr_dropout.value(),
                    pretrain_data_dir=Path(DATA_SRC_ALIGNED_DIR),
                    on_iter=_on_iter,
                    on_preview=_on_preview,
                    on_save=_on_save,
                )
            finally:
                if self._xseg_trainer is not None:
                    self._xseg_trainer.cleanup()
                    self._xseg_trainer = None
                self._set_xseg_controls_enabled(True)
                self._log_signal.log_ready.emit("已停止", True)
                self._close_preview_signal.close_ready.emit()

        self._run_in_thread(_task)

    def _on_stop_train(self):
        self._stop_train_btn.setEnabled(False)
        self._request_training_stop(self._xseg_trainer, "stop", self._log_signal)

    def _on_apply_trained(self):
        from faceswap.business.xseg_trainer import XSegTrainer
        d = self._get_apply_target_dir()
        trainer = XSegTrainer()
        sig = self._progress_signal
        sig.progress_ready.emit("Processing...")

        def _on_progress(current, total, elapsed):
            if total > 0:
                eta = elapsed / current * (total - current) if current > 0 else 0
                rate = current / elapsed if elapsed > 0 else 0
                sig.progress_ready.emit(f"Processing: {current}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")

        def _task():
            try:
                count = trainer.apply_trained_mask(d, XSEG_MODEL_DIR, progress_callback=_on_progress)
                sig.progress_ready.emit(f"Done: {count} faces")
                sig.done_ready.emit("完成", f"已应用训练遮罩到 {count} 张人脸")
            except Exception as e:
                sig.progress_ready.emit(f"Error: {e}")
                sig.error_ready.emit("错误", str(e))

        self._run_in_thread(_task)

    def _on_remove_trained(self):
        d = self._get_apply_target_dir()
        reply = QMessageBox.warning(
            self, "确认", f"确定要移除 {d} 中所有训练遮罩吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        from faceswap.business.xseg_trainer import XSegTrainer
        trainer = XSegTrainer()
        sig = self._progress_signal
        sig.progress_ready.emit("Processing...")

        def _on_progress(current, total, elapsed):
            if total > 0:
                eta = elapsed / current * (total - current) if current > 0 else 0
                rate = current / elapsed if elapsed > 0 else 0
                sig.progress_ready.emit(f"Processing: {current}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")

        def _task():
            try:
                count = trainer.remove_trained_mask(d, progress_callback=_on_progress)
                sig.progress_ready.emit(f"Done: {count} faces")
                sig.done_ready.emit("完成", f"已移除 {count} 张人脸的训练遮罩")
            except Exception as e:
                sig.progress_ready.emit(f"Error: {e}")
                sig.error_ready.emit("错误", str(e))

        self._run_in_thread(_task)

    def _on_apply_generic(self):
        from faceswap.business.xseg_trainer import XSegTrainer

        d = self._get_apply_target_dir()
        dialog = _AutoMaskSelectionDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        method = dialog.get_method()
        if method is None:
            return

        trainer = XSegTrainer()
        sig = self._progress_signal
        sig.progress_ready.emit("Processing...")

        def _on_progress(current, total, elapsed):
            if total > 0:
                eta = elapsed / current * (total - current) if current > 0 else 0
                rate = current / elapsed if elapsed > 0 else 0
                sig.progress_ready.emit(f"Processing: {current}/{total} [{elapsed:.0f}s<{eta:.0f}s, {rate:.1f}it/s]")

        method_names = {"face_parsing": "Face Parsing", "dfl": "DFL遮罩", "sam3": "SAM3"}

        def _task():
            try:
                count = trainer.apply_auto_mask(d, method, progress_callback=_on_progress)
                sig.progress_ready.emit(f"Done: {count} faces")
                sig.done_ready.emit("完成", f"已应用{method_names.get(method, method)}遮罩到 {count} 张人脸")
            except Exception as e:
                sig.progress_ready.emit(f"Error: {e}")
                sig.error_ready.emit("错误", str(e))

        self._run_in_thread(_task)


class _AutoMaskSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择自动遮罩方法")
        self.setFixedWidth(380)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel("选择要应用的自动遮罩方法："))

        self._dfl_btn = QRadioButton("DFL遮罩（推荐！）")
        self._dfl_btn.setChecked(True)
        self._face_parsing_btn = QRadioButton("Face Parsing（速度很慢！）")
        self._sam3_btn = QRadioButton("SAM3自动分割（速度很慢！）")
        layout.addWidget(self._dfl_btn)
        layout.addWidget(self._face_parsing_btn)
        layout.addWidget(self._sam3_btn)

        hint = QLabel("将对目标目录中每张人脸运行所选方法，\n生成多边形遮罩并保存到元数据。")
        hint.setStyleSheet("color: #666666; font-size: 12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_method(self) -> Optional[str]:
        if self._dfl_btn.isChecked():
            return "dfl"
        if self._face_parsing_btn.isChecked():
            return "face_parsing"
        if self._sam3_btn.isChecked():
            return "sam3"
        return None


class _RenameConflictDialog(QDialog):
    def __init__(self, existing_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型名称冲突")
        self.setFixedWidth(420)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel(f"已存在名称为 '{existing_name}' 的模型，请重命名。"))

        rename_row = QHBoxLayout()
        rename_row.addWidget(QLabel("重命名:"))
        self._name_edit = QLineEdit()
        rename_row.addWidget(self._name_edit)
        layout.addLayout(rename_row)

        hint = QLabel("若不重新命名，则点击保存后将使用原有模型继续训练。")
        hint.setStyleSheet("color: #666666; font-size: 12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_name(self) -> str:
        return self._name_edit.text().strip()

