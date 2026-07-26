import logging
from datetime import datetime
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QTextEdit


class GuiLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self._bridge: Optional[_LogBridge] = None

    def set_bridge(self, bridge: "_LogBridge"):
        self._bridge = bridge

    def emit(self, record: logging.LogRecord):
        if self._bridge is None:
            return
        msg = self.format(record)
        self._bridge.log_signal.emit(msg)


class _LogBridge(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self._te = text_edit
        self.log_signal.connect(self._append)

    def _append(self, msg: str):
        self._te.moveCursor(self._te.textCursor().MoveOperation.End)
        self._te.insertPlainText(msg + "\n")
        self._te.ensureCursorVisible()


_gui_handler: Optional[GuiLogHandler] = None


def get_gui_handler() -> GuiLogHandler:
    global _gui_handler
    if _gui_handler is None:
        _gui_handler = GuiLogHandler()
        _gui_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    return _gui_handler


def attach_gui_log(text_edit: QTextEdit):
    handler = get_gui_handler()
    bridge = _LogBridge(text_edit)
    handler.set_bridge(bridge)
    root = logging.getLogger()
    if handler not in root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)

    from DeepFaceLab.shared.logger import _loggers
    for name, logger in _loggers.items():
        logger.propagate = True
        for h in list(logger.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                logger.removeHandler(h)


def gui_log(msg: str):
    logging.info(msg)


def gui_error(msg: str):
    logging.error(msg)


def gui_warning(msg: str):
    logging.warning(msg)
