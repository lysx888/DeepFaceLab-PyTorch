import logging
import sys
from pathlib import Path
from typing import Optional

_log_format = "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
_date_format = "%Y-%m-%d %H:%M:%S"

_loggers: dict[str, logging.Logger] = {}
_default_file_log: Optional[Path] = None


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
