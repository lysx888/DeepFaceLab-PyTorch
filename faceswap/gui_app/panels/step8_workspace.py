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
    SCRFD_MODEL_DIR, PRETRAIN_DATA_DIR, SAEHD_PRETRAIN_DATA_DIR,
)

class _InfoSignal(QObject):
    """跨线程弹框信号桥: worker 线程 emit, GUI 线程弹 QMessageBox。"""
    info = pyqtSignal(str, str)
    warn = pyqtSignal(str, str)


class Step8Workspace(StepPanel):
    step_title = "8. 工具"
    step_desc = "工作区管理与DFL权重转换工具。"
    show_run_buttons = False

    def _build_params(self):
        self._pack_sig = _InfoSignal()
        self._pack_sig.info.connect(self._on_msg_info)
        self._pack_sig.warn.connect(self._on_msg_warn)

        ws_grp = QGroupBox("工作区管理")
        ws_lay = QVBoxLayout(ws_grp)
        ws_lay.setContentsMargins(8, 14, 8, 8)
        ws_lay.setSpacing(6)

        self._clear_btn = QPushButton("清除工作区重建目录")
        self._clear_btn.setStyleSheet(
            "QPushButton { background-color: #D45500; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 3px; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }")
        self._clear_btn.clicked.connect(self._on_clear_workspace)
        ws_lay.addWidget(self._clear_btn)

        ws_desc = QLabel("删除 workspace 下所有数据目录并重建空目录结构。")
        ws_desc.setStyleSheet("color: #666; font-size: 11px;")
        ws_lay.addWidget(ws_desc)
        self._params_area.addWidget(ws_grp)

        dfl_grp = QGroupBox("DFL 权重转换 (SAEHD / AMP / Quick96 / XSeg)")
        dfl_lay = QVBoxLayout(dfl_grp)
        dfl_lay.setContentsMargins(8, 14, 8, 8)
        dfl_lay.setSpacing(6)

        row0 = QHBoxLayout()
        row0.addWidget(QLabel("模型类型:"))
        self._dfl_archi_combo = QComboBox()
        self._dfl_archi_combo.addItems(["自动识别", "SAEHD", "AMP", "Quick96", "XSeg"])
        row0.addWidget(self._dfl_archi_combo, 1)
        dfl_lay.addLayout(row0)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("DFL模型目录:"))
        self._dfl_model_le = QLineEdit()
        self._dfl_model_le.setPlaceholderText("含 *_encoder.npy / *_data.dat")
        dfl_browse = QPushButton("浏览")
        dfl_browse.setFixedWidth(80)
        dfl_browse.clicked.connect(lambda: self._browse_dir(self._dfl_model_le))
        row1.addWidget(self._dfl_model_le, 1)
        row1.addWidget(dfl_browse)
        dfl_lay.addLayout(row1)

        self._convert_btn = QPushButton("转换 DFL 权重")
        self._convert_btn.setStyleSheet(
            "QPushButton { background-color: #5B2D8E; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 3px; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }")
        self._convert_btn.clicked.connect(self._on_convert_dfl)
        dfl_lay.addWidget(self._convert_btn)

        self._dfl_log = QTextEdit()
        self._dfl_log.setReadOnly(True)
        self._dfl_log.setMaximumHeight(220)
        self._dfl_log.setStyleSheet(
            "font-size: 11px; font-family: Consolas, 'Courier New', monospace; "
            "background-color: #1e1e1e; color: #d4d4d4;")
        dfl_lay.addWidget(self._dfl_log)
        self._params_area.addWidget(dfl_grp)

        self._convert_sig = _StreamBridge()
        self._convert_sig.stream_signal.connect(self._on_convert_output)

        fs_grp = QGroupBox("Faceset pak 打包 / 解压")
        fs_lay = QVBoxLayout(fs_grp)
        fs_lay.setContentsMargins(8, 14, 8, 8)
        fs_lay.setSpacing(6)

        fs_desc = QLabel("将对齐人脸目录打包为 faceset.pak（zip 容器，含元数据），或解压 .pak 到指定目录。")
        fs_desc.setStyleSheet("color: #666; font-size: 11px;")
        fs_desc.setWordWrap(True)
        fs_lay.addWidget(fs_desc)

        pack_row0 = QHBoxLayout()
        pack_row0.addWidget(QLabel("源图片目录:"))
        self._fs_src_le = QLineEdit()
        self._fs_src_le.setPlaceholderText("选择包含对齐人脸的目录")
        pack_browse = QPushButton("浏览")
        pack_browse.setFixedWidth(80)
        pack_browse.clicked.connect(lambda: self._browse_dir(self._fs_src_le))
        pack_row0.addWidget(self._fs_src_le, 1)
        pack_row0.addWidget(pack_browse)
        fs_lay.addLayout(pack_row0)

        pack_row1 = QHBoxLayout()
        pack_row1.addWidget(QLabel("输出 pak:"))
        self._fs_pak_le = QLineEdit(str(SAEHD_PRETRAIN_DATA_DIR / "faceset.pak"))
        self._fs_pak_le.setReadOnly(True)
        self._fs_pak_le.setToolTip("pak 名称固定为 faceset.pak, 不可自定义\n切换数据集时直接覆盖同名文件, 训练端自动解压并清理残留")
        pack_row1.addWidget(self._fs_pak_le, 1)
        fs_lay.addLayout(pack_row1)

        self._fs_btn = QPushButton("打包为 pak")
        self._fs_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 3px; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }")
        self._fs_btn.clicked.connect(self._on_pack)
        fs_lay.addWidget(self._fs_btn)

        fs_sep = QFrame()
        fs_sep.setFrameShape(QFrame.Shape.HLine)
        fs_sep.setStyleSheet("color: #CCCCCC;")
        fs_lay.addWidget(fs_sep)

        unpack_row0 = QHBoxLayout()
        unpack_row0.addWidget(QLabel("pak 文件:"))
        self._fs_unpak_le = QLineEdit()
        self._fs_unpak_le.setPlaceholderText("选择 .pak 文件")
        unpack_browse = QPushButton("浏览")
        unpack_browse.setFixedWidth(80)
        unpack_browse.clicked.connect(lambda: self._browse_file(self._fs_unpak_le))
        unpack_row0.addWidget(self._fs_unpak_le, 1)
        unpack_row0.addWidget(unpack_browse)
        fs_lay.addLayout(unpack_row0)

        unpack_row1 = QHBoxLayout()
        unpack_row1.addWidget(QLabel("解压目录:"))
        self._fs_out_le = QLineEdit(str(DATA_SRC_ALIGNED_DIR))
        self._fs_out_le.setPlaceholderText("默认 data_src/aligned")
        out_browse = QPushButton("浏览")
        out_browse.setFixedWidth(80)
        out_browse.clicked.connect(lambda: self._browse_dir(self._fs_out_le))
        unpack_row1.addWidget(self._fs_out_le, 1)
        unpack_row1.addWidget(out_browse)
        fs_lay.addLayout(unpack_row1)

        self._fs_unpack_btn = QPushButton("解压到目录")
        self._fs_unpack_btn.setStyleSheet(
            "QPushButton { background-color: #2E7D32; color: white; "
            "font-weight: bold; padding: 6px 16px; border-radius: 3px; }"
            "QPushButton:disabled { background-color: #A0A0A0; color: #E0E0E0; }")
        self._fs_unpack_btn.clicked.connect(self._on_unpack)
        fs_lay.addWidget(self._fs_unpack_btn)

        self._fs_progress = QProgressBar()
        self._fs_progress.setVisible(False)
        fs_lay.addWidget(self._fs_progress)
        self._params_area.addWidget(fs_grp)

    def _browse_dir(self, le):
        p = QFileDialog.getExistingDirectory(self, "选择目录", le.text())
        if p:
            le.setText(p)

    def _browse_file(self, le):
        p = QFileDialog.getOpenFileName(
            self, "选择文件", le.text(), "pak 文件 (*.pak);;所有文件 (*.*)")[0]
        if p:
            le.setText(p)

    def _on_msg_info(self, title, msg):
        QMessageBox.information(self, title, msg)

    def _on_msg_warn(self, title, msg):
        QMessageBox.warning(self, title, msg)

    def _on_clear_workspace(self):
        reply = QMessageBox.question(
            self, "确认", "确定要清除工作区吗？所有数据将被删除！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._clear_btn.setEnabled(False)
        self._clear_btn.setText("正在清除...")
        self._set_busy(True)

        def _task():
            try:
                from faceswap.business.workspace_manager import WorkspaceManager
                ws = WorkspaceManager()
                ws.clear_workspace()
            except Exception as e:
                _logger.error(f"清除工作区失败: {e}")

        t = threading.Thread(target=_task, daemon=True)
        t.start()
        QTimer.singleShot(100, lambda: self._poll_thread(t, self._clear_done))

    def _clear_done(self):
        self._clear_btn.setEnabled(True)
        self._clear_btn.setText("清除工作区重建目录")
        self._set_busy(False)

    def _poll_thread(self, t, on_done):
        if t.is_alive():
            QTimer.singleShot(100, lambda: self._poll_thread(t, on_done))
        else:
            on_done()

    def _on_convert_dfl(self):
        dfl_model = self._dfl_model_le.text().strip()
        if not dfl_model:
            QMessageBox.warning(self, "提示", "请填写 DFL 模型目录")
            return
        if not Path(dfl_model).is_dir():
            QMessageBox.warning(self, "提示", f"DFL 模型目录不存在: {dfl_model}")
            return

        # 检测所有 DFL 模型前缀
        all_prefixes = []
        for f in Path(dfl_model).iterdir():
            if f.is_file() and f.name.endswith('_encoder.npy'):
                all_prefixes.append(f.name[:-len('_encoder.npy')])
            elif f.is_file() and f.name.startswith('XSeg_') and f.name.endswith('.npy') and '_opt' not in f.name:
                all_prefixes.append('XSeg')

        # 确定架构：仅在显式选择时传 --archi，自动识别时让脚本逐模型自行判断
        selected_archi = self._dfl_archi_combo.currentText()
        _archi_map = {"SAEHD": "saehd", "AMP": "amp", "Quick96": "quick96", "XSeg": "xseg"}
        _archi = _archi_map.get(selected_archi)

        # 输出目录自动路由：始终用 MODEL_DIR 作为根，脚本按架构追加子目录
        output_dir = str(MODEL_DIR)

        # 多模型时不传 --model-name，让脚本批量转换；单模型时传前缀
        # XSeg 特判：选 XSeg 时只转 XSeg 模型，避免 --archi xseg 误伤其他模型
        if _archi == 'xseg' and 'XSeg' in all_prefixes:
            model_prefix = 'XSeg'
        else:
            model_prefix = all_prefixes[0] if len(all_prefixes) == 1 else None

        # 已存在检查
        archi_tag = {
            "saehd": ("df", "liae"), "amp": ("AMP",), "quick96": ("Quick96",),
            "xseg": ("XSeg",),
        }.get(_archi or "", ())
        if archi_tag and Path(output_dir).exists():
            for sub in Path(output_dir).iterdir():
                if sub.is_dir() and any(tag in sub.name for tag in archi_tag) \
                        and any(sub.iterdir()):
                    reply = QMessageBox.question(
                        self, "模型已存在",
                        f"输出目录下已存在 {sub.name}，若点击确定则会清空该模型并重新转换！",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No)
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                    shutil.rmtree(str(sub))
                    break

        self._dfl_log.clear()
        self._convert_btn.setEnabled(False)
        self._convert_btn.setText("转换中...")
        self._set_busy(True)

        script_path = Path(__file__).parent.parent.parent / "scripts" / "dfl_models_to_pytorch.py"
        cmd = [sys.executable, str(script_path),
               "--dfl-model", dfl_model, "--output", output_dir, "--verify"]
        if _archi:
            cmd += ["--archi", _archi]
        if model_prefix:
            cmd += ["--model-name", model_prefix]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if sys.platform == "win32" else 0

        def _run():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    creationflags=creationflags)
                for line in proc.stdout:
                    self._convert_sig.emit_stream(line.rstrip(), False)
                proc.wait()
                if proc.returncode == 0:
                    self._convert_sig.emit_stream("\n[完成] 转换成功", False)
                else:
                    self._convert_sig.emit_stream(f"\n[错误] 进程返回码 {proc.returncode}", False)
            except Exception as e:
                self._convert_sig.emit_stream(f"[错误] {e}", False)
            finally:
                self._convert_sig.emit_stream("__DONE__", False)

        threading.Thread(target=_run, daemon=True).start()

    def _on_convert_output(self, line: str, overwrite: bool):
        if line == "__DONE__":
            self._convert_btn.setEnabled(True)
            self._convert_btn.setText("转换 DFL 权重")
            self._set_busy(False)
            return
        self._dfl_log.append(line)

    def _on_pack(self):
        src = self._fs_src_le.text().strip()
        if not src or not Path(src).is_dir():
            QMessageBox.warning(self, "提示", "请选择有效的源图片目录")
            return
        out_path = SAEHD_PRETRAIN_DATA_DIR / "faceset.pak"
        if out_path.is_dir():
            QMessageBox.warning(self, "提示", f"输出路径是目录:\n{out_path}")
            return
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "提示", f"无法创建输出目录: {e}")
            return

        self._fs_btn.setEnabled(False)
        self._fs_btn.setText("打包中...")
        self._fs_progress.setVisible(True)
        self._fs_progress.setValue(0)
        self._set_busy(True)

        def _task():
            try:
                from faceswap.business.face_tool import FaceTool

                def _cb(cur, total, msg):
                    if total > 0:
                        self._fs_progress.setValue(int(cur * 100 / total))

                FaceTool().pack(Path(src), out_path, progress_callback=_cb)
                self._fs_progress.setValue(100)
                self._pack_sig.info.emit("完成", f"打包完成:\n{out_path}")
            except Exception as e:
                _logger.error(f"打包pak失败: {e}")
                self._pack_sig.warn.emit("错误", f"打包失败: {e}")
            finally:
                self._fs_btn.setEnabled(True)
                self._fs_btn.setText("打包为 pak")
                self._fs_progress.setVisible(False)
                self._set_busy(False)

        threading.Thread(target=_task, daemon=True).start()

    def _on_unpack(self):
        pak = self._fs_unpak_le.text().strip()
        out = self._fs_out_le.text().strip()
        if not pak or not Path(pak).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 .pak 文件")
            return
        if not out:
            QMessageBox.warning(self, "提示", "请填写解压目录")
            return

        self._fs_unpack_btn.setEnabled(False)
        self._fs_unpack_btn.setText("解压中...")
        self._fs_progress.setVisible(True)
        self._fs_progress.setValue(0)
        self._set_busy(True)

        def _task():
            try:
                from faceswap.business.face_tool import FaceTool

                def _cb(cur, total, msg):
                    if total > 0:
                        self._fs_progress.setValue(int(cur * 100 / total))

                count = FaceTool().unpack(Path(pak), Path(out), progress_callback=_cb)
                self._fs_progress.setValue(100)
                self._pack_sig.info.emit("完成", f"已解压 {count} 张图片到:\n{out}")
            except Exception as e:
                _logger.error(f"解压pak失败: {e}")
                self._pack_sig.warn.emit("错误", f"解压失败: {e}")
            finally:
                self._fs_unpack_btn.setEnabled(True)
                self._fs_unpack_btn.setText("解压到目录")
                self._fs_progress.setVisible(False)
                self._set_busy(False)

        threading.Thread(target=_task, daemon=True).start()
