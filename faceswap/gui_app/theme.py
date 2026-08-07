from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtWidgets import QApplication

PRIMARY = "#0078D4"
PRIMARY_HOVER = "#106EBE"
PRIMARY_PRESSED = "#005A9E"
DANGER = "#D13438"
DANGER_HOVER = "#A4262C"
WARNING = "#CA5010"
WARNING_HOVER = "#9B3A07"
NEUTRAL = "#7A7574"
NEUTRAL_HOVER = "#5C5654"
SUCCESS = "#0F8B4D"
BG_WINDOW = "#F3F3F3"
BG_SIDEBAR = "#E8E8E8"
BG_PANEL = "#FFFFFF"
BG_INPUT = "#FFFFFF"
BORDER = "#D6D6D6"
TEXT_PRIMARY = "#1B1B1B"
TEXT_SECONDARY = "#616161"
TEXT_ON_PRIMARY = "#FFFFFF"
TEXT_ON_DANGER = "#FFFFFF"

STYLESHEET = f"""
QMainWindow {{
    background-color: {BG_WINDOW};
}}
QWidget {{
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 13px;
    color: {TEXT_PRIMARY};
}}
#sidebar {{
    background-color: {BG_SIDEBAR};
    border-right: 1px solid {BORDER};
}}
#sidebar QPushButton {{
    background-color: transparent;
    border: none;
    text-align: left;
    padding: 10px 16px;
    font-size: 13px;
    color: {TEXT_PRIMARY};
    border-left: 3px solid transparent;
}}
#sidebar QPushButton:hover {{
    background-color: rgba(0,120,212,0.08);
}}
#sidebar QPushButton:checked {{
    background-color: rgba(0,120,212,0.12);
    border-left: 3px solid {PRIMARY};
    color: {PRIMARY};
    font-weight: 600;
}}
#contentArea {{
    background-color: {BG_WINDOW};
}}
QGroupBox {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {PRIMARY};
}}
QPushButton {{
    background-color: {PRIMARY};
    color: {TEXT_ON_PRIMARY};
    border: none;
    border-radius: 4px;
    padding: 7px 20px;
    font-weight: 600;
    min-height: 20px;
}}
QPushButton:hover {{
    background-color: {PRIMARY_HOVER};
}}
QPushButton:pressed {{
    background-color: {PRIMARY_PRESSED};
}}
QPushButton:disabled {{
    background-color: {BORDER};
    color: {TEXT_SECONDARY};
}}
QPushButton[danger="true"] {{
    background-color: {DANGER};
    color: {TEXT_ON_DANGER};
}}
QPushButton[danger="true"]:hover {{
    background-color: {DANGER_HOVER};
}}
QPushButton[danger="true"]:disabled {{
    background-color: {BORDER};
    color: {TEXT_SECONDARY};
}}
QPushButton[warning="true"] {{
    background-color: {WARNING};
    color: {TEXT_ON_PRIMARY};
}}
QPushButton[warning="true"]:hover {{
    background-color: {WARNING_HOVER};
}}
QPushButton[neutral="true"] {{
    background-color: {NEUTRAL};
    color: {TEXT_ON_PRIMARY};
}}
QPushButton[neutral="true"]:hover {{
    background-color: {NEUTRAL_HOVER};
}}
QPushButton[outline="true"] {{
    background-color: transparent;
    color: {PRIMARY};
    border: 1px solid {PRIMARY};
}}
QPushButton[outline="true"]:hover {{
    background-color: rgba(0,120,212,0.08);
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    min-height: 22px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    selection-background-color: rgba(0,120,212,0.15);
    selection-color: {PRIMARY};
}}
QCheckBox {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid {BORDER};
    background-color: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {PRIMARY};
    border-color: {PRIMARY};
}}
QTextEdit {{
    background-color: #1E1E1E;
    color: #CCCCCC;
    border: 1px solid {BORDER};
    border-radius: 4px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    padding: 4px;
}}
QTextEdit[readonly="true"] {{
    background-color: #1E1E1E;
}}
QLabel#stepTitle {{
    font-size: 18px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    padding: 4px 0;
}}
QLabel#stepDesc {{
    font-size: 12px;
    color: {TEXT_SECONDARY};
    padding: 2px 0 8px 0;
}}
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    background-color: {BG_INPUT};
    height: 20px;
}}
QProgressBar::chunk {{
    background-color: {PRIMARY};
    border-radius: 3px;
}}
QStatusBar {{
    background-color: {BG_SIDEBAR};
    border-top: 1px solid {BORDER};
    font-size: 12px;
    color: {TEXT_SECONDARY};
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
