from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QStatusBar,
    QScrollArea,
)

from faceswap.gui_app.theme import PRIMARY
from faceswap.gui_app.panels import (
    Step1VideoExtract, Step2FaceExtract, Step3XSeg,
    Step4Train, Step5Merge, Step6Output, Step7IFTrain, Step8Workspace,
)

_STEPS = [
    ("1. 视频提取", Step1VideoExtract),
    ("2. 人脸提取", Step2FaceExtract),
    ("3. 遮罩XSeg", Step3XSeg),
    ("4. 训练", Step4Train),
    ("5. 合成融合", Step5Merge),
    ("6. 导出视频", Step6Output),
    ("7. IF训练", Step7IFTrain),
    ("8. 工具", Step8Workspace),
]

_WINDOW_CTRL_SS = (
    "QPushButton { background-color: transparent; border: none; "
    "color: #FFFFFF; font-size: 14px; font-weight: bold; "
    "min-width: 32px; min-height: 32px; }"
    "QPushButton:hover { background-color: rgba(255,255,255,0.12); }"
)
_CLOSE_BTN_SS = (
    "QPushButton { background-color: transparent; border: none; "
    "color: #FFFFFF; font-size: 14px; font-weight: bold; "
    "min-width: 32px; min-height: 32px; }"
    "QPushButton:hover { background-color: #E81123; }"
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepFace")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setMinimumSize(960, 640)
        self.resize(1200, 760)
        self._drag_pos = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QWidget()
        header.setObjectName("topNav")
        header.setFixedHeight(48)
        header.setStyleSheet(
            f"#topNav {{ background-color: {PRIMARY}; }}"
            f"#topNav QPushButton {{ background-color: transparent; color: #FFFFFF; border: none; "
            f"padding: 0 16px; font-size: 13px; font-weight: 500; min-height: 48px; }}"
            f"#topNav QPushButton:hover {{ background-color: rgba(255,255,255,0.12); }}"
            f"#topNav QPushButton:checked {{ background-color: rgba(255,255,255,0.2); "
            f"border-bottom: 3px solid #FFFFFF; }}"
            f"#topNav QLabel {{ color: #FFFFFF; font-size: 15px; font-weight: 700; padding: 0 16px; }}"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 0, 0)
        header_layout.setSpacing(0)

        logo = QLabel("DeepFace")
        logo.setObjectName("topNav")
        header_layout.addWidget(logo)

        self._nav_btns = []
        for i, (title, _) in enumerate(_STEPS):
            btn = QPushButton(title)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self._switch(idx))
            header_layout.addWidget(btn)
            self._nav_btns.append(btn)

        header_layout.addStretch()

        btn_min = QPushButton("\u2014")
        btn_min.setStyleSheet(_WINDOW_CTRL_SS)
        btn_min.setToolTip("最小化")
        btn_min.clicked.connect(self.showMinimized)
        header_layout.addWidget(btn_min)
        self._btn_min = btn_min

        btn_max = QPushButton("\u25A1")
        btn_max.setStyleSheet(_WINDOW_CTRL_SS)
        btn_max.setToolTip("最大化")
        btn_max.clicked.connect(self._toggle_maximize)
        header_layout.addWidget(btn_max)
        self._btn_max = btn_max

        btn_close = QPushButton("\u2715")
        btn_close.setStyleSheet(_CLOSE_BTN_SS)
        btn_close.setToolTip("关闭")
        btn_close.clicked.connect(self.close)
        header_layout.addWidget(btn_close)
        self._btn_close = btn_close

        root.addWidget(header)

        self._stack = QStackedWidget()
        self._panels: list = [None] * len(_STEPS)
        for _name, _panel_cls in _STEPS:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            self._stack.addWidget(scroll)
        root.addWidget(self._stack, 1)

        status = QStatusBar()
        self.setStatusBar(status)
        from faceswap.shared.config import auto_select_device, get_device_name
        from PyQt6.QtWidgets import QToolButton
        dev = auto_select_device()
        dev_name = get_device_name(dev)
        self._device_btn = QToolButton()
        self._device_btn.setText(f"  {dev.type.upper()}: {dev_name}  ")
        self._device_btn.setStyleSheet(
            "QToolButton { color: #888; font-size: 11px; border: none; "
            "padding: 2px 8px; }"
            "QToolButton:hover { color: #333; background-color: rgba(0,0,0,0.06); }"
        )
        self._device_btn.setToolTip("点击切换计算设备: CPU ↔ CUDA（下次训练/合成/分割生效）")
        self._device_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._device_btn.clicked.connect(self._cycle_device)
        status.addPermanentWidget(self._device_btn)
        status.showMessage("就绪")

        self._switch(0)

    def _switch(self, idx: int):
        for i, btn in enumerate(self._nav_btns):
            btn.setChecked(i == idx)
        if self._panels[idx] is None:
            scroll = self._stack.widget(idx)
            panel = _STEPS[idx][1]()
            scroll.setWidget(panel)
            self._panels[idx] = panel
            from faceswap.gui_app.gui_utils import install_no_wheel
            from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox
            for child in panel.findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
                install_no_wheel(child)
        self._stack.setCurrentIndex(idx)
        self.statusBar().showMessage(f"步骤 {idx + 1}: {_STEPS[idx][0]}")

    def _cycle_device(self):
        """右下角设备管理器: 单卡点击循环 CPU<->CUDA, 多卡弹出菜单选择设备。"""
        import torch
        from PyQt6.QtWidgets import QMenu
        from faceswap.shared.config import (
            auto_select_device, get_device_name, set_manual_device, get_manual_device,
        )
        cur = auto_select_device()
        n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if n_cuda <= 1:
            # 单卡/无卡: 点击循环 CPU <-> CUDA:0
            if cur.type == "cuda":
                set_manual_device("cpu")
            elif n_cuda == 1:
                set_manual_device("cuda:0")
            else:
                self.statusBar().showMessage("无可用 CUDA 设备，仅 CPU 可用", 3000)
                return
        else:
            # 多卡: 弹出菜单选择 CPU / CUDA:0..n-1
            menu = QMenu(self)
            cur_key = get_manual_device() or ("cuda:0" if cur.type == "cuda" else "cpu")
            items = [("CPU", "cpu")]
            items += [(f"CUDA: {i}  {torch.cuda.get_device_name(i)}", f"cuda:{i}")
                      for i in range(n_cuda)]
            for label, key in items:
                act = menu.addAction(label)
                act.setCheckable(True)
                act.setChecked(key == cur_key)
                act.triggered.connect(lambda _, k=key: set_manual_device(k))
            menu.exec(self._device_btn.mapToGlobal(self._device_btn.rect().bottomLeft()))

        dev = auto_select_device()
        dev_name = get_device_name(dev)
        self._device_btn.setText(f"  {dev.type.upper()}: {dev_name}  ")
        self.statusBar().showMessage(f"计算设备已切换: {dev.type.upper()} {dev_name}", 3000)

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def set_busy(self, busy: bool):
        """运行时禁用所有可交互控件，放行两类例外：
        1. 窗口控制按钮（最小化/最大化/关闭）
        2. 带 busy_keep 属性的控件（各面板的"停止"按钮）

        任务运行期间除停止外全部防误操作；滚动条/日志/进度不受影响。
        """
        from PyQt6.QtWidgets import (
            QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
            QLineEdit, QCheckBox, QRadioButton, QSlider, QToolButton,
        )
        _interactive = (QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
                        QLineEdit, QCheckBox, QRadioButton, QSlider, QToolButton)
        _skip = {id(self._btn_min), id(self._btn_max), id(self._btn_close)}
        if busy:
            if not hasattr(self, '_busy_count') or self._busy_count == 0:
                self._busy_disabled = []
                for child in self.findChildren(_interactive):
                    if id(child) in _skip or child.property("busy_keep"):
                        continue
                    if child.isEnabled():
                        child.setEnabled(False)
                        self._busy_disabled.append(child)
            self._busy_count = getattr(self, '_busy_count', 0) + 1
        else:
            self._busy_count = max(0, getattr(self, '_busy_count', 1) - 1)
            if self._busy_count == 0:
                for w in getattr(self, '_busy_disabled', []):
                    try:
                        w.setEnabled(True)
                    except RuntimeError:
                        pass
                self._busy_disabled = []

    def closeEvent(self, event):
        for i in range(self._stack.count()):
            scroll = self._stack.widget(i)
            panel = scroll.widget() if hasattr(scroll, 'widget') else None
            if panel is not None and getattr(panel, '_running', False):
                from PyQt6.QtWidgets import QMessageBox
                reply = QMessageBox.warning(
                    self, "确认关闭",
                    "有任务正在运行，关闭窗口可能丢失进度。确定关闭？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                break
        super().closeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.position().y() < 48:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            else:
                self._drag_pos = None

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if self.isMaximized():
                w_max = self.width()
                self.showNormal()
                ratio = event.position().x() / w_max
                new_x = event.globalPosition().toPoint().x() - int(self.width() * ratio)
                new_y = event.globalPosition().toPoint().y() - self._drag_pos.y()
                self.move(new_x, new_y)
                self._drag_pos = event.globalPosition().toPoint() - self.pos()
            else:
                self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        if event.position().y() < 48:
            self._toggle_maximize()
