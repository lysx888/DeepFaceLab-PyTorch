"""
dedup_dialog.py - src 去重过滤对话框

用 3DDFA 3DMM 参数对 src aligned 人脸做角度/表情/光照状态去重。
含角度/姿态/表情/光照可视化分析。
"""

import threading
from pathlib import Path
from collections import defaultdict

import numpy as np

from PyQt6.QtCore import pyqtSignal, QObject, Qt
from PyQt6.QtGui import QPixmap, QImage, QAction
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar,
    QTextEdit, QMessageBox, QGroupBox, QGridLayout,
    QScrollArea, QWidget, QTabWidget, QComboBox,
    QMenu, QApplication, QFrame,
)

from faceswap.gui_app.gui_utils import install_no_wheel


class DedupProgressSignal(QObject):
    progress_ready = pyqtSignal(int, int, str)
    done_ready = pyqtSignal(str)
    error_ready = pyqtSignal(str)


class AnalyzeSignal(QObject):
    progress_ready = pyqtSignal(int, int, str)
    done_ready = pyqtSignal(list)
    error_ready = pyqtSignal(str)


def _pca_2d(data: np.ndarray):
    """PCA 降到 2D, 返回 (pc1, pc2, explained_variance_ratio)."""
    mean = data.mean(axis=0)
    centered = data - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ Vt[:2].T
    variances = (S ** 2) / (len(data) - 1)
    total_var = variances.sum()
    ratio = (variances[:2] / total_var * 100) if total_var > 0 else [0, 0]
    return coords[:, 0], coords[:, 1], ratio


class _ThumbCell(QFrame):
    """ThumbnailDialog 中的缩略图单元: 双击查看原图, 右键删除(移动到 aligned_angle)."""
    remove_requested = pyqtSignal(object)

    def __init__(self, img_path: Path, thumb_size: int = 150, parent=None):
        super().__init__(parent)
        self.image_path = Path(img_path)
        self._thumb_size = thumb_size
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame {{ background: #fff; border: 1px solid #ccc; border-radius: 3px; }}"
            f"QFrame:hover {{ border: 1px solid #0078D4; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        self._lbl = QLabel()
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setFixedSize(thumb_size, thumb_size)
        self._lbl.setStyleSheet("background: #f0f0f0;")
        lay.addWidget(self._lbl)

        name = self.image_path.name
        if len(name) > 20:
            name = name[:9] + ".." + name[-9:]
        self._name_lbl = QLabel(name)
        self._name_lbl.setStyleSheet("font-size: 10px; color: #666;")
        self._name_lbl.setWordWrap(True)
        self._name_lbl.setFixedWidth(thumb_size)
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._name_lbl)

        self._load_thumbnail()

    def _load_thumbnail(self):
        pix = QPixmap(str(self.image_path))
        if not pix.isNull():
            scaled = pix.scaled(
                self._thumb_size, self._thumb_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._lbl.setPixmap(scaled)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._show_full_image()
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        act_view = QAction("查看原图", self)
        act_view.triggered.connect(self._show_full_image)
        menu.addAction(act_view)
        menu.addSeparator()
        act_del = QAction("删除 (移动到 aligned_angle)", self)
        act_del.triggered.connect(self._request_remove)
        menu.addAction(act_del)
        menu.exec(event.globalPos())

    def _show_full_image(self):
        import cv2
        img = cv2.imread(str(self.image_path))
        if img is None:
            QMessageBox.warning(self, "错误", f"无法读取图片:\n{self.image_path}")
            return
        h, w = img.shape[:2]
        rgb = img[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"原图 - {self.image_path.name} ({w}x{h})")
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        screen = QApplication.primaryScreen()
        if screen:
            max_w = screen.availableGeometry().width() - 40
            max_h = screen.availableGeometry().height() - 80
            if pixmap.width() > max_w or pixmap.height() > max_h:
                pixmap = pixmap.scaled(
                    max_w, max_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
        lbl.setPixmap(pixmap)
        lay.addWidget(lbl)
        dlg.resize(min(pixmap.width() + 20, 1600), min(pixmap.height() + 20, 1000))
        dlg.exec()

    def _request_remove(self):
        reply = QMessageBox.warning(
            self, "确认删除",
            f"将移动到 aligned_angle 文件夹:\n{self.image_path.name}\n\n"
            f"图片及对应 .json 元数据将被移动 (非真正删除)。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.remove_requested.emit(self)


class ThumbnailDialog(QDialog):
    def __init__(self, title, images, aligned_dir=None, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        self.setWindowTitle(title)
        self.setMinimumWidth(900)
        self.setMinimumHeight(700)
        self.resize(1200, 800)

        self._aligned_dir = Path(aligned_dir) if aligned_dir else None
        self._trash_dir = None
        if self._aligned_dir is not None:
            self._trash_dir = self._aligned_dir.parent / "aligned_angle"

        self._thumb_size = 150
        self._cell_spacing = 6
        self._cells: list[_ThumbCell] = []
        self._deleted_count = 0

        layout = QVBoxLayout(self)

        self._info_label = QLabel(f"共 {len(images)} 张  |  双击查看原图  |  右键删除(移动到 aligned_angle)")
        self._info_label.setStyleSheet("color: #555; font-size: 12px; padding: 4px;")
        layout.addWidget(self._info_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)

        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(self._cell_spacing)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._scroll.setWidget(self._grid_widget)
        layout.addWidget(self._scroll)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        for img_path in images:
            cell = _ThumbCell(img_path, self._thumb_size, parent=self)
            cell.remove_requested.connect(self._on_remove)
            self._cells.append(cell)
        self._reflow()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()

    def _compute_cols(self) -> int:
        avail = self.width() - 30
        cell_w = self._thumb_size + self._cell_spacing + 4
        return max(1, avail // cell_w)

    def _reflow(self):
        cols = self._compute_cols()
        for i, cell in enumerate(self._cells):
            row, col = divmod(i, cols)
            self._grid_layout.addWidget(cell, row, col, Qt.AlignmentFlag.AlignTop)

    def _on_remove(self, cell: _ThumbCell):
        if self._trash_dir is None:
            QMessageBox.warning(self, "无法删除", "未提供 aligned_dir, 无法确定 aligned_angle 目标路径。")
            return
        try:
            self._trash_dir.mkdir(parents=True, exist_ok=True)
            img_path = cell.image_path
            json_path = img_path.with_suffix(".json")
            if img_path.exists():
                target_img = self._trash_dir / img_path.name
                if target_img.exists():
                    target_img.unlink()
                img_path.replace(target_img)
            if json_path.exists():
                target_json = self._trash_dir / json_path.name
                if target_json.exists():
                    target_json.unlink()
                json_path.replace(target_json)
        except OSError as e:
            QMessageBox.warning(self, "删除失败", f"移动文件失败:\n{e}")
            return

        self._cells.remove(cell)
        self._grid_layout.removeWidget(cell)
        cell.deleteLater()
        self._deleted_count += 1
        remaining = len(self._cells)
        self._info_label.setText(
            f"剩余 {remaining} 张  |  已删除 {self._deleted_count} 张  |  "
            f"双击查看原图  |  右键删除(移动到 aligned_angle)"
        )
        self._reflow()


class _GridWidget(QWidget):
    """通用 2D 网格可视化组件."""

    def __init__(self, items, x_label, y_label, x_range, y_range,
                 n_bins_x, n_bins_y, unit="", aligned_dir=None, parent=None):
        """
        items: [(path, x_val, y_val), ...]
        x_range, y_range: (min, max)
        n_bins_x, n_bins_y: 网格数量
        unit: 坐标单位 (如 "°")
        aligned_dir: aligned 目录路径, 传给 ThumbnailDialog 用于计算 aligned_angle
        """
        super().__init__(parent)
        self._unit = unit
        self._x_label = x_label
        self._y_label = y_label
        self._x_range = x_range
        self._y_range = y_range
        self._n_bins_x = n_bins_x
        self._n_bins_y = n_bins_y
        self._aligned_dir = aligned_dir

        total_bins_x = n_bins_x + 2
        total_bins_y = n_bins_y + 2

        grid_data = defaultdict(list)
        for path, x_val, y_val in items:
            if x_val < x_range[0]:
                xb = 0
            elif x_val > x_range[1]:
                xb = n_bins_x + 1
            else:
                xb = int((x_val - x_range[0]) / (x_range[1] - x_range[0]) * n_bins_x)
                xb = min(n_bins_x - 1, max(0, xb)) + 1

            if y_val < y_range[0]:
                yb = 0
            elif y_val > y_range[1]:
                yb = n_bins_y + 1
            else:
                yb = int((y_val - y_range[0]) / (y_range[1] - y_range[0]) * n_bins_y)
                yb = min(n_bins_y - 1, max(0, yb)) + 1

            grid_data[(xb, yb)].append(path)

        self._grid_data = grid_data

        layout = QVBoxLayout(self)

        total = len(items)
        covered = len(grid_data)
        all_counts = [len(grid_data.get((xb, yb), []))
                      for xb in range(total_bins_x)
                      for yb in range(total_bins_y)]
        max_count = max(all_counts) if all_counts else 1

        summary = QLabel(
            f"总图片: {total}  覆盖网格: {covered}/{total_bins_x*total_bins_y}  "
            f"最大密度: {max_count}  "
            f"网格: {total_bins_x}×{total_bins_y} (含溢出列/行)"
        )
        summary.setStyleSheet("color: #555; font-size: 12px; padding: 2px;")
        layout.addWidget(summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        grid_container = QWidget()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setSpacing(1)

        header_style = "color: #666; font-size: 10px;"
        overflow_style = "color: #D45500; font-size: 10px; font-weight: bold;"
        x_step = (x_range[1] - x_range[0]) / n_bins_x
        y_step = (y_range[1] - y_range[0]) / n_bins_y

        label = QLabel(f"<{x_range[0]:.0f}{unit}")
        label.setStyleSheet(overflow_style)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(label, 0, 1)

        for xb in range(n_bins_x + 1):
            label = QLabel(f"{x_range[0]+xb*x_step:.0f}{unit}")
            label.setStyleSheet(header_style)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid_layout.addWidget(label, 0, xb + 2)

        label = QLabel(f">{x_range[1]:.0f}{unit}")
        label.setStyleSheet(overflow_style)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(label, 0, n_bins_x + 3)

        for yb in range(total_bins_y - 1, -1, -1):
            row = total_bins_y - yb

            if yb == 0:
                y_label_text = f"<{y_range[0]:.0f}{unit}"
                yl = QLabel(y_label_text)
                yl.setStyleSheet(overflow_style)
            elif yb == n_bins_y + 1:
                y_label_text = f">{y_range[1]:.0f}{unit}"
                yl = QLabel(y_label_text)
                yl.setStyleSheet(overflow_style)
            else:
                y_label_text = f"{y_range[0]+(yb-1)*y_step:.0f}{unit}"
                yl = QLabel(y_label_text)
                yl.setStyleSheet(header_style)
            yl.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignRight)
            grid_layout.addWidget(yl, row, 0)

            for xb in range(total_bins_x):
                col = xb + 1
                count = len(grid_data.get((xb, yb), []))

                cell = QPushButton()
                cell.setFixedSize(22, 22)

                is_overflow = (xb == 0 or xb == n_bins_x + 1 or
                               yb == 0 or yb == n_bins_y + 1)

                if count == 0:
                    cell.setStyleSheet(
                        "QPushButton { background-color: #F0F0F0; border: 1px solid #E0E0E0; border-radius: 2px; }"
                    )
                    cell.setEnabled(False)
                else:
                    intensity = min(count / max_count, 1.0)
                    r = int(255 * (1 - intensity * 0.8))
                    g = int(255 * (1 - intensity * 0.5))
                    b = 255
                    color = f"rgb({r}, {g}, {b})"
                    border_color = "#D45500" if is_overflow else "#0078D4"
                    cell.setStyleSheet(
                        f"QPushButton {{ background-color: {color}; border: 1px solid {border_color}; border-radius: 2px; }}"
                        f"QPushButton:hover {{ border: 2px solid {border_color}; }}"
                    )

                    if xb == 0:
                        x_text = f"<{x_range[0]:.0f}{unit}"
                    elif xb == n_bins_x + 1:
                        x_text = f">{x_range[1]:.0f}{unit}"
                    else:
                        x_center = x_range[0] + (xb - 1 + 0.5) * x_step
                        x_text = f"≈{x_center:.1f}{unit}"

                    if yb == 0:
                        y_text = f"<{y_range[0]:.0f}{unit}"
                    elif yb == n_bins_y + 1:
                        y_text = f">{y_range[1]:.0f}{unit}"
                    else:
                        y_center = y_range[0] + (yb - 1 + 0.5) * y_step
                        y_text = f"≈{y_center:.1f}{unit}"

                    cell.setToolTip(
                        f"{x_label}{x_text}  {y_label}{y_text}  ({count} 张)"
                    )
                    cell.clicked.connect(
                        lambda checked, xt=x_text, yt=y_text, key=(xb, yb):
                        self._on_cell_click(xt, yt, key)
                    )

                grid_layout.addWidget(cell, row, col)

        xl = QLabel(f"{x_label} →")
        xl.setStyleSheet("color: #0078D4; font-size: 11px; font-weight: bold;")
        grid_layout.addWidget(xl, total_bins_y + 1, 1, 1, 5)

        yl2 = QLabel(f"↑ {y_label}")
        yl2.setStyleSheet("color: #0078D4; font-size: 11px; font-weight: bold;")
        grid_layout.addWidget(yl2, 1, total_bins_x + 2)

        scroll.setWidget(grid_container)
        layout.addWidget(scroll)

    def _on_cell_click(self, x_text, y_text, key):
        paths = self._grid_data.get(key, [])
        if not paths:
            return
        title = (f"{self._x_label}{x_text}  "
                 f"{self._y_label}{y_text}  ({len(paths)} 张)")
        dlg = ThumbnailDialog(title, paths, aligned_dir=self._aligned_dir, parent=self)
        dlg.exec()


class StateAnalysisDialog(QDialog):
    def __init__(self, aligned_dir, yaw_grid=5.0, pitch_grid=5.0, use_cpu=False, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        self._aligned_dir = Path(aligned_dir)
        self._yaw_grid = yaw_grid
        self._pitch_grid = pitch_grid
        self._use_cpu = use_cpu
        self._states = []

        self._sig = AnalyzeSignal()
        self._sig.progress_ready.connect(self._on_progress)
        self._sig.done_ready.connect(self._on_analyze_done)
        self._sig.error_ready.connect(self._on_error)

        self.setWindowTitle("人脸状态分析 (3DDFA)")
        self.setMinimumWidth(1100)
        self.setMinimumHeight(800)
        self.resize(1100, 800)

        layout = QVBoxLayout(self)

        self._info_label = QLabel(f"目录: {self._aligned_dir}")
        self._info_label.setStyleSheet("color: #555; font-size: 12px; padding: 4px;")
        layout.addWidget(self._info_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel()
        self._progress_label.setStyleSheet("color: #0078D4; font-size: 12px;")
        layout.addWidget(self._progress_label)

        self._tabs = QTabWidget()
        placeholder = QLabel("分析中...")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #999; font-size: 16px; padding: 40px;")
        self._tabs.addTab(placeholder, "加载中")
        layout.addWidget(self._tabs)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._on_analyze()

    def _on_progress(self, current, total, msg):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_label.setText(msg)

    def _on_error(self, err_msg):
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        QMessageBox.warning(self, "错误", err_msg)

    def _on_analyze(self):
        from faceswap.shared.file_manager import FileManager
        images = FileManager.find_images(self._aligned_dir)
        if not images:
            QMessageBox.warning(self, "错误", f"目录中没有图片:\n{self._aligned_dir}")
            return

        self._progress_bar.setVisible(True)
        self._progress_label.setText("加载中...")

        def _task():
            try:
                from faceswap.business.face_dedup import analyze_states

                def _progress_cb(current, total, msg):
                    self._sig.progress_ready.emit(current, total, msg)

                results = analyze_states(
                    str(self._aligned_dir),
                    batch_size=32,
                    use_cpu=self._use_cpu,
                    progress_callback=_progress_cb,
                )
                self._sig.done_ready.emit(results)
            except Exception as e:
                import traceback
                self._sig.error_ready.emit(f"{e}\n{traceback.format_exc()}")

        threading.Thread(target=_task, daemon=True).start()

    def _on_analyze_done(self, results):
        self._states = results
        self._progress_bar.setVisible(False)
        self._progress_label.setText(f"分析完成: {len(results)} 张图片")

        self._tabs.clear()

        self._build_angle_tab()
        self._build_pose_tab()
        self._build_expr_tab()
        self._build_lighting_tab()

    def _build_angle_tab(self):
        items = [(s[0], s[1], s[2]) for s in self._states]
        gw = _GridWidget(
            items, "yaw", "pitch",
            (-75, 75), (-75, 75),
            int(150 / self._yaw_grid), int(150 / self._pitch_grid),
            unit="°", aligned_dir=self._aligned_dir,
        )
        self._tabs.addTab(gw, "角度 (yaw×pitch)")

    def _build_pose_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        combo = QComboBox()
        combo.addItem("yaw × roll", ("yaw", -75, 75, "roll", -75, 75))
        combo.addItem("pitch × roll", ("pitch", -75, 75, "roll", -75, 75))
        install_no_wheel(combo)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        def _update_grid():
            data = combo.currentData()
            x_name, x_min, x_max, y_name, y_min, y_max = data
            idx_map = {"yaw": 1, "pitch": 2, "roll": 3}
            xi = idx_map[x_name]
            yi = idx_map[y_name]
            items = [(s[0], s[xi], s[yi]) for s in self._states]
            gw = _GridWidget(
                items, x_name, y_name,
                (x_min, x_max), (y_min, y_max),
                int((x_max - x_min) / self._yaw_grid),
                int((y_max - y_min) / self._yaw_grid),
                unit="°", aligned_dir=self._aligned_dir,
            )
            scroll.setWidget(gw)

        combo.currentIndexChanged.connect(lambda: _update_grid())
        tab_layout.addWidget(combo)
        tab_layout.addWidget(scroll)
        _update_grid()

        self._tabs.addTab(tab, "姿态 (roll)")

    def _build_expr_tab(self):
        exp_data = np.array([s[4] for s in self._states])
        pc1, pc2, ratio = _pca_2d(exp_data)

        padding = 0.05
        x_min, x_max = float(pc1.min()), float(pc1.max())
        y_min, y_max = float(pc2.min()), float(pc2.max())
        x_pad = (x_max - x_min) * padding + 1e-6
        y_pad = (y_max - y_min) * padding + 1e-6
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        items = [(self._states[i][0], float(pc1[i]), float(pc2[i]))
                 for i in range(len(self._states))]

        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        info = QLabel(
            f"PCA 降维: 64维表情 → 2D  "
            f"PC1={ratio[0]:.1f}%  PC2={ratio[1]:.1f}%  "
            f"累计={ratio[0]+ratio[1]:.1f}%"
        )
        info.setStyleSheet("color: #555; font-size: 12px; padding: 4px;")
        tab_layout.addWidget(info)

        gw = _GridWidget(
            items, "PC1", "PC2",
            (x_min, x_max), (y_min, y_max),
            20, 20, aligned_dir=self._aligned_dir,
        )
        tab_layout.addWidget(gw)

        self._tabs.addTab(tab, "表情 (PCA)")

    def _build_lighting_tab(self):
        sh_data = np.array([s[5] for s in self._states])
        pc1, pc2, ratio = _pca_2d(sh_data)

        padding = 0.05
        x_min, x_max = float(pc1.min()), float(pc1.max())
        y_min, y_max = float(pc2.min()), float(pc2.max())
        x_pad = (x_max - x_min) * padding + 1e-6
        y_pad = (y_max - y_min) * padding + 1e-6
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        items = [(self._states[i][0], float(pc1[i]), float(pc2[i]))
                 for i in range(len(self._states))]

        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        info = QLabel(
            f"PCA 降维: 27维SH光照 → 2D  "
            f"PC1={ratio[0]:.1f}%  PC2={ratio[1]:.1f}%  "
            f"累计={ratio[0]+ratio[1]:.1f}%"
        )
        info.setStyleSheet("color: #555; font-size: 12px; padding: 4px;")
        tab_layout.addWidget(info)

        gw = _GridWidget(
            items, "PC1", "PC2",
            (x_min, x_max), (y_min, y_max),
            20, 20, aligned_dir=self._aligned_dir,
        )
        tab_layout.addWidget(gw)

        self._tabs.addTab(tab, "光照 (PCA)")


class SrcDedupDialog(QDialog):
    def __init__(self, aligned_dir, parent=None):
        super().__init__(parent)
        self._aligned_dir = Path(aligned_dir)
        self._sig = DedupProgressSignal()
        self._sig.progress_ready.connect(self._on_progress)
        self._sig.done_ready.connect(self._on_done)
        self._sig.error_ready.connect(self._on_error)

        self.setWindowTitle("src 去重过滤 (3DDFA)")
        self.setMinimumWidth(520)
        self.setMinimumHeight(560)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        info = QLabel(
            f"目录: {self._aligned_dir}\n"
            "用 3DDFA 3DMM 参数 (角度+表情+光照) 去冗余,\n"
            "保留角度/表情/光照全覆盖, 移除同状态冗余帧。\n"
            "阈值基于 SAEHD 插值能力: 角度≤5°/表情 L2<1.0/光照 L2<0.3\n"
            "可保证被删帧通过 SAEHD 训练后近似无损恢复。"
        )
        info.setStyleSheet("color: #555; font-size: 12px; padding: 4px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        param_grp = QGroupBox("参数")
        param_lay = QVBoxLayout(param_grp)
        param_lay.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("每簇保留:"))
        self._keep_per_cluster = QSpinBox()
        self._keep_per_cluster.setRange(1, 10)
        self._keep_per_cluster.setValue(2)
        install_no_wheel(self._keep_per_cluster)
        row1.addWidget(self._keep_per_cluster)
        row1.addSpacing(12)
        row1.addWidget(QLabel("大角度保护:"))
        self._protect_yaw = QDoubleSpinBox()
        self._protect_yaw.setRange(0.0, 90.0)
        self._protect_yaw.setValue(45.0)
        self._protect_yaw.setSuffix("°")
        install_no_wheel(self._protect_yaw)
        row1.addWidget(self._protect_yaw)
        row1.addStretch()
        param_lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("角度网格:"))
        self._yaw_grid = QDoubleSpinBox()
        self._yaw_grid.setRange(1.0, 30.0)
        self._yaw_grid.setValue(5.0)
        self._yaw_grid.setSuffix("°")
        install_no_wheel(self._yaw_grid)
        row2.addWidget(self._yaw_grid)
        row2.addSpacing(12)
        row2.addWidget(QLabel("表情阈值:"))
        self._exp_thresh = QDoubleSpinBox()
        self._exp_thresh.setRange(0.1, 20.0)
        self._exp_thresh.setValue(1.0)
        install_no_wheel(self._exp_thresh)
        row2.addWidget(self._exp_thresh)
        row2.addStretch()
        param_lay.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("光照阈值:"))
        self._sh_thresh = QDoubleSpinBox()
        self._sh_thresh.setRange(0.01, 5.0)
        self._sh_thresh.setValue(0.3)
        install_no_wheel(self._sh_thresh)
        row3.addWidget(self._sh_thresh)
        row3.addSpacing(12)
        row3.addWidget(QLabel("批大小:"))
        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 128)
        self._batch_size.setValue(32)
        install_no_wheel(self._batch_size)
        row3.addWidget(self._batch_size)
        row3.addStretch()
        param_lay.addLayout(row3)

        row4 = QHBoxLayout()
        self._dry_run = QCheckBox("仅分析 (不移动文件)")
        self._dry_run.setChecked(True)
        row4.addWidget(self._dry_run)
        self._use_cpu = QCheckBox("用 CPU")
        row4.addWidget(self._use_cpu)
        row4.addStretch()
        param_lay.addLayout(row4)

        layout.addWidget(param_grp)

        btn_row = QHBoxLayout()
        self._analyze_btn = QPushButton("分析头像")
        self._analyze_btn.setStyleSheet(
            "QPushButton { background-color: #5B2D8E; color: white; font-weight: bold; padding: 6px 20px; border-radius: 3px; }"
        )
        self._analyze_btn.clicked.connect(self._on_analyze)
        btn_row.addWidget(self._analyze_btn)
        self._run_btn = QPushButton("开始去重")
        self._run_btn.setStyleSheet(
            "QPushButton { background-color: #0078D4; color: white; font-weight: bold; padding: 6px 20px; border-radius: 3px; }"
            "QPushButton:disabled { background-color: #999; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        self._close_btn = QPushButton("关闭")
        self._close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._close_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel()
        self._progress_label.setStyleSheet("color: #0078D4; font-size: 12px;")
        layout.addWidget(self._progress_label)

        self._report = QTextEdit()
        self._report.setReadOnly(True)
        self._report.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(self._report)

    def _on_analyze(self):
        if not self._aligned_dir.exists():
            QMessageBox.warning(self, "错误", f"目录不存在:\n{self._aligned_dir}")
            return

        dlg = StateAnalysisDialog(
            self._aligned_dir,
            yaw_grid=self._yaw_grid.value(),
            pitch_grid=self._yaw_grid.value(),
            use_cpu=self._use_cpu.isChecked(),
            parent=self,
        )
        dlg.exec()

    def _on_progress(self, current, total, msg):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_label.setText(msg)

    def _on_done(self, report_text):
        self._report.setPlainText(report_text)
        self._progress_bar.setVisible(False)
        self._progress_label.setText("完成")
        self._run_btn.setEnabled(True)
        self._close_btn.setText("关闭")

    def _on_error(self, err_msg):
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        self._run_btn.setEnabled(True)
        QMessageBox.warning(self, "错误", err_msg)

    def _on_run(self):
        if not self._aligned_dir.exists():
            QMessageBox.warning(self, "错误", f"目录不存在:\n{self._aligned_dir}")
            return

        from faceswap.shared.file_manager import FileManager
        images = FileManager.find_images(self._aligned_dir)
        if not images:
            QMessageBox.warning(self, "错误", f"目录中没有图片:\n{self._aligned_dir}")
            return

        self._run_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_label.setText("加载中...")
        self._report.clear()

        params = dict(
            input_dir=str(self._aligned_dir),
            dry_run=self._dry_run.isChecked(),
            batch_size=self._batch_size.value(),
            keep_per_cluster=self._keep_per_cluster.value(),
            yaw_grid=self._yaw_grid.value(),
            pitch_grid=self._yaw_grid.value(),
            roll_grid=self._yaw_grid.value(),
            exp_thresh=self._exp_thresh.value(),
            sh_thresh=self._sh_thresh.value(),
            protect_yaw=self._protect_yaw.value(),
            use_cpu=self._use_cpu.isChecked(),
        )

        def _task():
            try:
                from faceswap.business.face_dedup import run_dedup

                def _progress_cb(current, total, msg):
                    self._sig.progress_ready.emit(current, total, msg)

                keep, remove, report_text = run_dedup(
                    progress_callback=_progress_cb,
                    **params,
                )
                self._sig.done_ready.emit(report_text)
            except Exception as e:
                import traceback
                self._sig.error_ready.emit(f"{e}\n{traceback.format_exc()}")

        threading.Thread(target=_task, daemon=True).start()
