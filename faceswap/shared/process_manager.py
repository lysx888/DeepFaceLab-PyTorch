import subprocess
import sys
import signal
from pathlib import Path
from typing import Optional, Callable

from faceswap.shared.logger import get_logger

_logger = get_logger("process_manager")


class ProcessManager:
    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._on_output: Optional[Callable[[str], None]] = None
        self._on_error: Optional[Callable[[str], None]] = None
        self._on_exit: Optional[Callable[[int], None]] = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def set_callbacks(
        self,
        on_output: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._on_output = on_output
        self._on_error = on_error
        self._on_exit = on_exit

    def start(self, args: list[str], cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> None:
        if self.is_running:
            raise RuntimeError("A process is already running.")

        _logger.info(f"Starting process: {' '.join(args)}")

        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        self._reader_thread("stdout", self._process.stdout, self._on_output)
        self._reader_thread("stderr", self._process.stderr, self._on_error)
        self._wait_thread()

    def stop(self) -> None:
        if not self.is_running:
            return
        _logger.info("Stopping process...")
        try:
            if sys.platform == "win32":
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._process.terminate()
        except OSError:
            pass

    def kill(self) -> None:
        if not self.is_running:
            return
        _logger.info("Killing process...")
        try:
            self._process.kill()
        except OSError:
            pass

    def wait(self, timeout: Optional[float] = None) -> int:
        if self._process is None:
            return -1
        return self._process.wait(timeout=timeout)

    def _reader_thread(self, name: str, pipe, callback: Optional[Callable[[str], None]]) -> None:
        import threading

        def _read():
            try:
                for line in iter(pipe.readline, b""):
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if callback:
                        callback(text)
                    else:
                        _logger.info(f"[{name}] {text}")
            except ValueError:
                pass
            finally:
                pipe.close()

        t = threading.Thread(target=_read, daemon=True)
        t.start()

    def _wait_thread(self) -> None:
        import threading

        def _wait():
            if self._process is None:
                return
            retcode = self._process.wait()
            _logger.info(f"Process exited with code {retcode}")
            if self._on_exit:
                self._on_exit(retcode)

        t = threading.Thread(target=_wait, daemon=True)
        t.start()
