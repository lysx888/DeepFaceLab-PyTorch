import time
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QGroupBox, QProgressBar, QFileDialog, QTextEdit, QMessageBox,
)

from DeepFaceLab.gui_app.gui_log import gui_log, gui_error
from DeepFaceLab.gui_app.panels import StepPanel
from DeepFaceLab.setting import WORKSPACE_DIR, INSIGHTFACE_SCRFD_DIR, INSIGHTFACE_SYNTHETICS_DIR


class _LogBridge(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def emit_log(self, line: str):
        self.log_signal.emit(line)


class _ProgressBridge(QObject):
    progress_signal = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def emit_progress(self, pct: int, text: str):
        self.progress_signal.emit(pct, text)


class StepInsightFaceTrain(StepPanel):
    step_title = "9. insightface训练"
    step_desc = "训练insightface人脸检测器和关键点检测器模型。"
    show_run_buttons = False

    def _build_params(self):
        self._trainer = None
        self._train_start_time: Optional[float] = None
        self._log_bridge = _LogBridge(self)
        self._progress_bridge = _ProgressBridge(self)
        self._log_bridge.log_signal.connect(self._append_log)
        self._progress_bridge.progress_signal.connect(self._apply_progress)

        self._build_data_prep_section()
        self._build_scrfd_section()
        self._build_synthetics_section()
        self._build_progress_section()
        self._build_export_section()
        self._build_log_section()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._timer.setInterval(1000)

    def _build_data_prep_section(self):
        grp = QGroupBox("数据准备")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        r1 = QHBoxLayout()
        lbl1 = QLabel("数据源:")
        lbl1.setFixedWidth(100)
        self._data_source = QComboBox()
        self._data_source.addItems(["manual_annotated", "WIDERFace", "FaceSynthetics"])
        r1.addWidget(lbl1)
        r1.addWidget(self._data_source, 1)
        lay.addLayout(r1)

        r2 = QHBoxLayout()
        lbl2 = QLabel("目标模型:")
        lbl2.setFixedWidth(100)
        self._target_model = QComboBox()
        self._target_model.addItems(["SCRFD", "2d106"])
        r2.addWidget(lbl2)
        r2.addWidget(self._target_model, 1)
        lay.addLayout(r2)

        r3 = QHBoxLayout()
        lbl3 = QLabel("验证集比例:")
        lbl3.setFixedWidth(100)
        self._val_ratio = QDoubleSpinBox()
        self._val_ratio.setRange(0.0, 0.5)
        self._val_ratio.setValue(0.1)
        self._val_ratio.setSingleStep(0.05)
        self._val_ratio.setDecimals(2)
        r3.addWidget(lbl3)
        r3.addWidget(self._val_ratio)
        r3.addStretch()
        lay.addLayout(r3)

        self._prepare_btn = QPushButton("准备数据")
        self._prepare_btn.clicked.connect(self._on_prepare_data)
        lay.addWidget(self._prepare_btn)

        self._params_area.addWidget(grp)

    def _build_scrfd_section(self):
        grp = QGroupBox("SCRFD 训练配置")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        r1 = QHBoxLayout()
        lbl1 = QLabel("配置文件:")
        lbl1.setFixedWidth(100)
        self._scrfd_config = QComboBox()
        self._scrfd_config.addItems(["dfl_scrfd_10g_bnkps"])
        r1.addWidget(lbl1)
        r1.addWidget(self._scrfd_config, 1)
        lay.addLayout(r1)

        r2 = QHBoxLayout()
        lbl2 = QLabel("GPU数量:")
        lbl2.setFixedWidth(100)
        self._scrfd_gpus = QSpinBox()
        self._scrfd_gpus.setRange(1, 8)
        self._scrfd_gpus.setValue(1)
        r2.addWidget(lbl2)
        r2.addWidget(self._scrfd_gpus)
        r2.addStretch()
        lay.addLayout(r2)

        r3 = QHBoxLayout()
        lbl3 = QLabel("断点续训:")
        lbl3.setFixedWidth(100)
        self._scrfd_resume = QLineEdit()
        self._scrfd_resume.setPlaceholderText("checkpoint路径（可选）")
        btn_browse = QPushButton("浏览")
        btn_browse.setFixedWidth(60)
        btn_browse.setProperty("outline", True)
        btn_browse.clicked.connect(lambda: self._browse_file(self._scrfd_resume, "选择checkpoint"))
        r3.addWidget(lbl3)
        r3.addWidget(self._scrfd_resume, 1)
        r3.addWidget(btn_browse)
        lay.addLayout(r3)

        btn_row = QHBoxLayout()
        self._scrfd_train_btn = QPushButton("开始训练")
        self._scrfd_train_btn.setStyleSheet(
            "QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 6px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #B34700; }"
            "QPushButton:disabled { background-color: #D6D6D6; color: #616161; }"
        )
        self._scrfd_train_btn.clicked.connect(self._on_train_scrfd)
        self._scrfd_stop_btn = QPushButton("停止训练")
        self._scrfd_stop_btn.setProperty("danger", True)
        self._scrfd_stop_btn.setFixedWidth(100)
        self._scrfd_stop_btn.setEnabled(False)
        self._scrfd_stop_btn.clicked.connect(self._on_stop_training)
        btn_row.addWidget(self._scrfd_train_btn)
        btn_row.addWidget(self._scrfd_stop_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._params_area.addWidget(grp)

    def _build_synthetics_section(self):
        grp = QGroupBox("2d106 训练配置")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        r1 = QHBoxLayout()
        lbl1 = QLabel("Backbone:")
        lbl1.setFixedWidth(100)
        self._syn_backbone = QComboBox()
        self._syn_backbone.addItems(["resnet50d", "resnet100d", "resnet34d", "resnet18d"])
        r1.addWidget(lbl1)
        r1.addWidget(self._syn_backbone, 1)
        lay.addLayout(r1)

        r2 = QHBoxLayout()
        lbl2 = QLabel("Batch Size:")
        lbl2.setFixedWidth(100)
        self._syn_batch = QSpinBox()
        self._syn_batch.setRange(1, 256)
        self._syn_batch.setValue(64)
        r2.addWidget(lbl2)
        r2.addWidget(self._syn_batch)
        r2.addStretch()
        lay.addLayout(r2)

        r3 = QHBoxLayout()
        lbl3 = QLabel("GPU数量:")
        lbl3.setFixedWidth(100)
        self._syn_gpus = QSpinBox()
        self._syn_gpus.setRange(1, 8)
        self._syn_gpus.setValue(1)
        r3.addWidget(lbl3)
        r3.addWidget(self._syn_gpus)
        r3.addStretch()
        lay.addLayout(r3)

        r4 = QHBoxLayout()
        lbl4 = QLabel("最大Epoch:")
        lbl4.setFixedWidth(100)
        self._syn_epochs = QSpinBox()
        self._syn_epochs.setRange(1, 1000)
        self._syn_epochs.setValue(80)
        r4.addWidget(lbl4)
        r4.addWidget(self._syn_epochs)
        r4.addStretch()
        lay.addLayout(r4)

        btn_row = QHBoxLayout()
        self._syn_train_btn = QPushButton("开始训练")
        self._syn_train_btn.setStyleSheet(
            "QPushButton { background-color: #D45500; color: white; font-weight: bold; padding: 6px 20px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #B34700; }"
            "QPushButton:disabled { background-color: #D6D6D6; color: #616161; }"
        )
        self._syn_train_btn.clicked.connect(self._on_train_synthetics)
        self._syn_stop_btn = QPushButton("停止训练")
        self._syn_stop_btn.setProperty("danger", True)
        self._syn_stop_btn.setFixedWidth(100)
        self._syn_stop_btn.setEnabled(False)
        self._syn_stop_btn.clicked.connect(self._on_stop_training)
        btn_row.addWidget(self._syn_train_btn)
        btn_row.addWidget(self._syn_stop_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._params_area.addWidget(grp)

    def _build_progress_section(self):
        grp = QGroupBox("训练进度")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        self._progress_bar = QProgressBar()
        self._progress_bar.setValue(0)
        lay.addWidget(self._progress_bar)

        self._status_label = QLabel("等待开始训练...")
        self._status_label.setStyleSheet("font-size: 12px; color: #616161;")
        lay.addWidget(self._status_label)

        self._params_area.addWidget(grp)

    def _build_export_section(self):
        grp = QGroupBox("ONNX 导出与部署")
        lay = QVBoxLayout(grp)
        lay.setSpacing(4)

        r1 = QHBoxLayout()
        lbl1 = QLabel("模型类型:")
        lbl1.setFixedWidth(100)
        self._export_model_type = QComboBox()
        self._export_model_type.addItems(["SCRFD", "2d106"])
        r1.addWidget(lbl1)
        r1.addWidget(self._export_model_type, 1)
        lay.addLayout(r1)

        r2 = QHBoxLayout()
        lbl2 = QLabel("Checkpoint:")
        lbl2.setFixedWidth(100)
        self._export_ckpt = QLineEdit()
        self._export_ckpt.setPlaceholderText("checkpoint路径")
        btn_browse = QPushButton("浏览")
        btn_browse.setFixedWidth(60)
        btn_browse.setProperty("outline", True)
        btn_browse.clicked.connect(lambda: self._browse_file(self._export_ckpt, "选择checkpoint"))
        r2.addWidget(lbl2)
        r2.addWidget(self._export_ckpt, 1)
        r2.addWidget(btn_browse)
        lay.addLayout(r2)

        r3 = QHBoxLayout()
        lbl3 = QLabel("输入尺寸:")
        lbl3.setFixedWidth(100)
        self._export_input_size = QSpinBox()
        self._export_input_size.setRange(64, 2048)
        self._export_input_size.setValue(640)
        self._export_input_size.setSingleStep(128)
        r3.addWidget(lbl3)
        r3.addWidget(self._export_input_size)
        self._auto_deploy = QCheckBox("自动部署")
        r3.addWidget(self._auto_deploy)
        r3.addStretch()
        lay.addLayout(r3)

        btn_row = QHBoxLayout()
        self._export_btn = QPushButton("导出ONNX")
        self._export_btn.clicked.connect(self._on_export)
        self._deploy_btn = QPushButton("部署到antelopev2")
        self._deploy_btn.clicked.connect(self._on_deploy)
        btn_row.addWidget(self._export_btn)
        btn_row.addWidget(self._deploy_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._params_area.addWidget(grp)

    def _build_log_section(self):
        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMaximumHeight(200)
        self._params_area.addWidget(self._log_output)

    def _browse_file(self, line_edit: QLineEdit, title: str):
        p, _ = QFileDialog.getOpenFileName(self, title, line_edit.text())
        if p:
            line_edit.setText(p)

    def _append_log(self, line: str):
        self._log_output.moveCursor(self._log_output.textCursor().MoveOperation.End)
        self._log_output.insertPlainText(line + "\n")
        self._log_output.ensureCursorVisible()

    def _apply_progress(self, pct: int, text: str):
        self._progress_bar.setValue(pct)
        self._status_label.setText(text)

    def _get_trainer(self):
        from DeepFaceLab.business.insightface_trainer import InsightFaceTrainer
        if self._trainer is None:
            self._trainer = InsightFaceTrainer()
        return self._trainer

    def _set_training_ui(self, running: bool, model_type: str = ""):
        is_scrfd = model_type == "scrfd"
        self._scrfd_train_btn.setEnabled(not running or not is_scrfd)
        self._scrfd_stop_btn.setEnabled(running and is_scrfd)
        self._syn_train_btn.setEnabled(not running or is_scrfd)
        self._syn_stop_btn.setEnabled(running and not is_scrfd)
        self._prepare_btn.setEnabled(not running)

        if running:
            self._train_start_time = time.time()
            self._timer.start()
        else:
            self._timer.stop()
            self._train_start_time = None

    def _update_elapsed(self):
        if self._train_start_time is None:
            return
        trainer = self._get_trainer()
        if not trainer.is_training_running():
            self._set_training_ui(False)
            self._status_label.setText("训练已结束")
            return

        elapsed = time.time() - self._train_start_time
        m = int(elapsed) // 60
        s = int(elapsed) % 60
        elapsed_str = f"{m}:{s:02d}"

        if hasattr(trainer, '_current_model_type') and trainer._current_model_type == "scrfd":
            p = trainer.scrfd_progress
            if p.iter_total > 0:
                pct = int(p.iter_cur / p.iter_total * 100)
            else:
                pct = 0
            text = f"Epoch {p.epoch} | Loss: {p.loss:.4f} | {elapsed_str}"
            self._progress_bar.setValue(pct)
            self._status_label.setText(text)
        elif hasattr(trainer, '_current_model_type') and trainer._current_model_type == "synthetics":
            p = trainer.synthetics_progress
            text = f"Epoch {p.epoch} | Loss: {p.loss:.4f} | Val Loss: {p.val_loss:.4f} | {elapsed_str}"
            self._status_label.setText(text)

    def _on_prepare_data(self):
        source = self._data_source.currentText()
        target = self._target_model.currentText()
        val_ratio = self._val_ratio.value()

        self._prepare_btn.setEnabled(False)

        def _task():
            try:
                if target == "SCRFD":
                    from DeepFaceLab.business.scrfd_data_preparer import SCRFDDataPreparer
                    preparer = SCRFDDataPreparer()
                    if source == "manual_annotated":
                        stats = preparer.prepare_from_manual_annotated(val_ratio=val_ratio)
                    elif source == "WIDERFace":
                        wider_dir = QFileDialog.getExistingDirectory(self, "选择WIDERFace数据集目录")
                        if not wider_dir:
                            return
                        stats = preparer.prepare_from_widerface(Path(wider_dir), val_ratio=val_ratio)
                    else:
                        gui_error("SCRFD不支持FaceSynthetics数据源")
                        return
                    self._log_bridge.emit_log(f"SCRFD数据准备完成: {stats}")
                else:
                    from DeepFaceLab.business.synthetics_data_preparer import SyntheticsDataPreparer
                    preparer = SyntheticsDataPreparer()
                    if source == "manual_annotated":
                        stats = preparer.prepare_from_manual_annotated()
                    elif source == "FaceSynthetics":
                        fs_dir = QFileDialog.getExistingDirectory(self, "选择FaceSynthetics数据集目录")
                        if not fs_dir:
                            return
                        stats = preparer.prepare_from_facesynthetics(Path(fs_dir))
                    else:
                        gui_error("2d106不支持WIDERFace数据源")
                        return
                    self._log_bridge.emit_log(f"2d106数据准备完成: {stats}")
            except Exception as e:
                self._log_bridge.emit_log(f"数据准备失败: {e}")
                gui_error(str(e))
            finally:
                self._prepare_btn.setEnabled(True)

        threading.Thread(target=_task, daemon=True).start()

    def _on_train_scrfd(self):
        from DeepFaceLab.business.insightface_trainer import InsightFaceTrainer, SCRFDTrainConfig

        trainer = self._get_trainer()
        config = SCRFDTrainConfig(
            config_name=self._scrfd_config.currentText(),
            gpus=list(range(self._scrfd_gpus.value())),
            resume_from=self._scrfd_resume.text() or None,
        )

        self._set_training_ui(True, "scrfd")
        self._log_bridge.emit_log("启动SCRFD训练...")

        def _on_progress(progress):
            self._progress_bridge.emit_progress(0, "")

        try:
            trainer.train_scrfd(config, on_progress=_on_progress)
        except Exception as e:
            self._set_training_ui(False)
            self._log_bridge.emit_log(f"SCRFD训练启动失败: {e}")
            gui_error(str(e))

    def _on_train_synthetics(self):
        from DeepFaceLab.business.insightface_trainer import InsightFaceTrainer, SyntheticsTrainConfig

        trainer = self._get_trainer()
        config = SyntheticsTrainConfig(
            backbone=self._syn_backbone.currentText(),
            batch_size=self._syn_batch.value(),
            num_gpus=self._syn_gpus.value(),
            max_epochs=self._syn_epochs.value(),
        )

        self._set_training_ui(True, "synthetics")
        self._log_bridge.emit_log("启动2d106训练...")

        def _on_progress(progress):
            self._progress_bridge.emit_progress(0, "")

        try:
            trainer.train_synthetics(config, on_progress=_on_progress)
        except Exception as e:
            self._set_training_ui(False)
            self._log_bridge.emit_log(f"2d106训练启动失败: {e}")
            gui_error(str(e))

    def _on_stop_training(self):
        trainer = self._get_trainer()
        if trainer.is_training_running():
            exit_code = trainer.stop_training()
            self._log_bridge.emit_log(f"训练已停止 (退出码: {exit_code})")
        self._set_training_ui(False)

    def _on_export(self):
        from DeepFaceLab.business.insightface_exporter import InsightFaceExporter

        model_type = self._export_model_type.currentText()
        ckpt_path = self._export_ckpt.text().strip()
        if not ckpt_path:
            gui_error("请选择checkpoint路径")
            return

        input_size = self._export_input_size.value()
        auto_deploy = self._auto_deploy.isChecked()

        self._export_btn.setEnabled(False)

        def _task():
            try:
                exporter = InsightFaceExporter()
                if model_type == "SCRFD":
                    result = exporter.export_scrfd_onnx(
                        checkpoint_path=Path(ckpt_path),
                        input_size=(input_size, input_size),
                        deploy=auto_deploy,
                    )
                else:
                    result = exporter.export_synthetics_onnx(
                        checkpoint_path=Path(ckpt_path),
                        input_size=input_size,
                        deploy=auto_deploy,
                    )
                msg = (
                    f"ONNX导出完成: {result.onnx_path}\n"
                    f"最大绝对误差: {result.max_abs_error:.6f}\n"
                    f"一致性验证: {'通过' if result.is_consistent else '未通过'}\n"
                    f"已部署: {'是' if result.deployed else '否'}"
                )
                self._log_bridge.emit_log(msg)
            except Exception as e:
                self._log_bridge.emit_log(f"ONNX导出失败: {e}")
                gui_error(str(e))
            finally:
                self._export_btn.setEnabled(True)

        threading.Thread(target=_task, daemon=True).start()

    def _on_deploy(self):
        from DeepFaceLab.business.insightface_exporter import InsightFaceExporter

        model_type = self._export_model_type.currentText()
        ckpt_path = self._export_ckpt.text().strip()

        onnx_path_str, _ = QFileDialog.getOpenFileName(
            self, "选择ONNX模型文件", "", "ONNX Files (*.onnx)"
        )
        if not onnx_path_str:
            return

        model_name = "scrfd_10g_bnkps.onnx" if model_type == "SCRFD" else "2d106det.onnx"

        try:
            exporter = InsightFaceExporter()
            exporter.deploy_to_antelopev2(Path(onnx_path_str), model_name)
            self._log_bridge.emit_log(f"ONNX模型已部署到antelopev2目录")
            QMessageBox.information(
                self, "部署完成",
                "ONNX模型已成功部署到antelopev2目录。\n请重启应用以加载新模型。"
            )
        except Exception as e:
            gui_error(str(e))
