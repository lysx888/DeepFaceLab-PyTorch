import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("training_process_manager")


@dataclass
class ProcessInfo:
    pid: int
    command: list[str]
    start_time: str
    exit_code: Optional[int] = None


class TrainingProcessManager:

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._process_info: Optional[ProcessInfo] = None
        self._on_stdout: Optional[Callable[[str], None]] = None
        self._on_stderr: Optional[Callable[[str], None]] = None
        self._exit_event = threading.Event()

    def start_process(
        self,
        command: list[str],
        cwd: Optional[Path] = None,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
    ) -> ProcessInfo:
        if self.is_running():
            raise RuntimeError("A training process is already running.")

        _logger.info(f"Starting training process: {' '.join(command)}")

        self._on_stdout = on_stdout
        self._on_stderr = on_stderr
        self._exit_event.clear()

        try:
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
                creationflags=creationflags,
            )
        except FileNotFoundError:
            raise RuntimeError(f"Command not found: {command[0]}")
        except OSError as e:
            raise RuntimeError(f"Failed to start process: {e}")

        start_time = datetime.now().isoformat()
        self._process_info = ProcessInfo(
            pid=self._process.pid,
            command=list(command),
            start_time=start_time,
        )

        self._start_reader_thread("stdout", self._process.stdout, on_stdout)
        self._start_reader_thread("stderr", self._process.stderr, on_stderr)
        self._start_wait_thread()

        _logger.info(f"Training process started (PID: {self._process.pid})")
        return self._process_info

    def stop_process(self) -> Optional[int]:
        if not self.is_running():
            return None

        _logger.info("Stopping training process...")
        try:
            self._process.terminate()
        except OSError:
            pass

        try:
            exit_code = self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _logger.warning("Process did not terminate, killing...")
            try:
                self._process.kill()
            except OSError:
                pass
            try:
                exit_code = self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                exit_code = -1

        if self._process_info is not None:
            self._process_info.exit_code = exit_code

        _logger.info(f"Training process stopped (exit code: {exit_code})")
        return exit_code

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def get_process_info(self) -> Optional[ProcessInfo]:
        if self._process_info is None:
            return None
        if not self.is_running() and self._process_info.exit_code is None:
            self._process_info.exit_code = self._process.returncode
        return self._process_info

    def wait_for_completion(self, timeout: Optional[float] = None) -> int:
        if self._process is None:
            raise RuntimeError("No process has been started.")

        if not self.is_running():
            if self._process_info is not None and self._process_info.exit_code is None:
                self._process_info.exit_code = self._process.returncode
            return self._process.returncode

        timed_out = not self._exit_event.wait(timeout=timeout)
        if timed_out:
            raise TimeoutError(f"Process did not complete within {timeout} seconds.")

        if self._process_info is not None and self._process_info.exit_code is None:
            self._process_info.exit_code = self._process.returncode
        return self._process.returncode

    def _start_reader_thread(
        self,
        name: str,
        pipe,
        callback: Optional[Callable[[str], None]],
    ) -> None:
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

    def _start_wait_thread(self) -> None:
        def _wait():
            if self._process is None:
                return
            retcode = self._process.wait()
            if self._process_info is not None:
                self._process_info.exit_code = retcode
            self._exit_event.set()
            _logger.info(f"Training process exited with code {retcode}")

        t = threading.Thread(target=_wait, daemon=True)
        t.start()
