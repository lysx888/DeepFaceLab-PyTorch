import logging
import sys
from pathlib import Path
from typing import Optional, Callable

_log_format = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
_date_format = "%Y-%m-%d %H:%M:%S"

_loggers: dict[str, logging.Logger] = {}
_default_file_log: Optional[Path] = None
_gui_handler: Optional["GuiLogHandler"] = None
_log_bridge_cls = None


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    global _loggers, _default_file_log

    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True

    formatter = logging.Formatter(_log_format, datefmt=_date_format)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    file_path = log_file or _default_file_log
    if file_path is not None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


def set_default_log_file(log_file: Path) -> None:
    global _default_file_log
    _default_file_log = Path(log_file)


class GuiLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self._bridge = None

    def set_bridge(self, bridge):
        self._bridge = bridge

    def emit(self, record: logging.LogRecord):
        if self._bridge is None:
            return
        msg = self.format(record)
        self._bridge.log_signal.emit(msg)


def _get_log_bridge_cls():
    global _log_bridge_cls
    if _log_bridge_cls is None:
        from PyQt6.QtCore import QObject, pyqtSignal

        class _LogBridge(QObject):
            log_signal = pyqtSignal(str)

            def __init__(self, text_edit,
                         overwrite_getter: Optional[Callable[[], bool]] = None,
                         need_newline_setter: Optional[Callable[[bool], None]] = None):
                super().__init__()
                self._te = text_edit
                self._overwrite_getter = overwrite_getter
                self._need_newline_setter = need_newline_setter
                self.log_signal.connect(self._append)

            def _append(self, msg: str):
                te = self._te
                if self._overwrite_getter is not None and self._need_newline_setter is not None:
                    overwrite = self._overwrite_getter()
                    if overwrite:
                        cursor = te.textCursor()
                        cursor.beginEditBlock()
                        cursor.movePosition(cursor.MoveOperation.End)
                        cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
                        cursor.removeSelectedText()
                        cursor.insertText(msg)
                        cursor.endEditBlock()
                        te.setTextCursor(cursor)
                    else:
                        cursor = te.textCursor()
                        cursor.movePosition(cursor.MoveOperation.End)
                        if cursor.position() > 0:
                            cursor.insertText("\n")
                        cursor.insertText(msg)
                        te.setTextCursor(cursor)
                    self._need_newline_setter(False)
                else:
                    te.moveCursor(te.textCursor().MoveOperation.End)
                    te.insertPlainText(msg + "\n")
                te.ensureCursorVisible()

        _log_bridge_cls = _LogBridge
    return _log_bridge_cls


def attach_gui_handler(text_edit,
                       overwrite_getter: Optional[Callable[[], bool]] = None,
                       need_newline_setter: Optional[Callable[[bool], None]] = None):
    global _gui_handler

    bridge_cls = _get_log_bridge_cls()
    bridge = bridge_cls(text_edit, overwrite_getter, need_newline_setter)

    if _gui_handler is None:
        _gui_handler = GuiLogHandler()
        _gui_handler.setFormatter(logging.Formatter(_log_format, datefmt=_date_format))

    _gui_handler.set_bridge(bridge)

    root = logging.getLogger()
    if _gui_handler not in root.handlers:
        root.addHandler(_gui_handler)
    root.setLevel(logging.INFO)

    for name, logger in _loggers.items():
        logger.propagate = True
        for h in list(logger.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
