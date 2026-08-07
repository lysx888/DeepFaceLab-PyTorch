from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QStatusBar,
    QScrollArea,
)

from faceswap.gui_app.theme import PRIMARY
from faceswap.gui_app.panels import (
    Step1VideoExtract, Step2FaceExtract, Step3XSeg,
    Step4Train, Step5Merge, Step6Output, Step7Tools, Step8Workspace,
)

_STEPS = [
    ("1. 视频提取", Step1VideoExtract),
    ("2. 人脸提取", Step2FaceExtract),
    ("3. 遮罩XSeg", Step3XSeg),
    ("4. 训练", Step4Train),
    ("5. 合成融合", Step5Merge),
    ("6. 导出视频", Step6Output),
    ("7. 人脸工具", Step7Tools),
    ("8. 工作区", Step8Workspace),
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

        btn_max = QPushButton("\u25A1")
        btn_max.setStyleSheet(_WINDOW_CTRL_SS)
        btn_max.setToolTip("最大化")
        btn_max.clicked.connect(self._toggle_maximize)
        header_layout.addWidget(btn_max)

        btn_close = QPushButton("\u2715")
        btn_close.setStyleSheet(_CLOSE_BTN_SS)
        btn_close.setToolTip("关闭")
        btn_close.clicked.connect(self.close)
        header_layout.addWidget(btn_close)

        root.addWidget(header)

        self._stack = QStackedWidget()
        for _, panel_cls in _STEPS:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel_cls())
            self._stack.addWidget(scroll)
        root.addWidget(self._stack, 1)

        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("就绪")

        self._switch(0)
        self._install_no_wheel_global()

    def _switch(self, idx: int):
        for i, btn in enumerate(self._nav_btns):
            btn.setChecked(i == idx)
        self._stack.setCurrentIndex(idx)
        self.statusBar().showMessage(f"步骤 {idx + 1}: {_STEPS[idx][0]}")

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _install_no_wheel_global(self):
        from faceswap.gui_app.gui_utils import install_no_wheel
        from PyQt6.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox
        for child in self.findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
            install_no_wheel(child)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.position().y() < 48:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            else:
                self._drag_pos = None

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            if self.isMaximized():
                self.showNormal()
                ratio = event.position().x() / self.width()
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
