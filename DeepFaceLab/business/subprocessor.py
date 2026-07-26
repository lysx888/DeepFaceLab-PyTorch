from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from typing import TypeVar, Generic, Optional, Callable, Any
import sys

from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("subprocessor")

T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")

_CMD_START = "start"
_CMD_ITEM = "item"
_CMD_RESULT = "result"
_CMD_PROGRESS = "progress"
_CMD_ERROR = "error"
_CMD_DONE = "done"
_CMD_STOP = "stop"


class Subprocessor(ABC, Generic[T_Input, T_Output]):

    def __init__(
        self,
        name: str = "Subprocessor",
        num_workers: int = 0,
        worker_batch_size: int = 1,
    ) -> None:
        self._name = name
        self._num_workers = num_workers if num_workers > 0 else max(1, (self._cpu_count() - 1))
        self._worker_batch_size = worker_batch_size

    @staticmethod
    def _cpu_count() -> int:
        try:
            import os
            return os.cpu_count() or 1
        except Exception:
            return 1

    @abstractmethod
    def process_item(self, item: T_Input) -> T_Output:
        ...

    def on_client_startup(self) -> None:
        pass

    def on_client_finalize(self) -> None:
        pass

    def run(
        self,
        inputs: list[T_Input],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[T_Output]:
        if not inputs:
            return []

        total = len(inputs)
        input_queue: Queue = Queue()
        result_queue: Queue = Queue()

        for item in inputs:
            input_queue.put(item)

        for _ in range(self._num_workers):
            input_queue.put(None)

        workers: list[Process] = []
        for i in range(self._num_workers):
            p = Process(
                target=self._worker_loop,
                args=(input_queue, result_queue, i),
                daemon=True,
            )
            p.start()
            workers.append(p)

        results: list[T_Output] = []
        completed = 0
        active_workers = self._num_workers
        failed_items: list[T_Input] = []

        while active_workers > 0:
            try:
                msg_type, *msg_data = result_queue.get(timeout=300)
            except Exception:
                _logger.warning(f"{self._name}: Timeout waiting for worker results.")
                break

            if msg_type == _CMD_RESULT:
                result_item = msg_data[0]
                results.append(result_item)
                completed += 1
                if on_progress:
                    on_progress(completed, total)

            elif msg_type == _CMD_ERROR:
                worker_id = msg_data[0]
                item = msg_data[1]
                error_msg = msg_data[2]
                _logger.error(f"{self._name} worker {worker_id} error on item: {error_msg}")
                failed_items.append(item)

            elif msg_type == _CMD_DONE:
                active_workers -= 1

        for p in workers:
            if p.is_alive():
                p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        if failed_items:
            _logger.warning(f"{self._name}: {len(failed_items)} items failed, retrying on main process...")
            for item in failed_items:
                try:
                    self.on_client_startup()
                    result = self.process_item(item)
                    results.append(result)
                    completed += 1
                    if on_progress:
                        on_progress(completed, total)
                except Exception as e:
                    _logger.error(f"{self._name}: Failed to process item on main process: {e}")

        return results

    def _worker_loop(
        self,
        input_queue: Queue,
        result_queue: Queue,
        worker_id: int,
    ) -> None:
        try:
            self.on_client_startup()
        except Exception as e:
            result_queue.put((_CMD_ERROR, worker_id, None, str(e)))
            result_queue.put((_CMD_DONE,))
            return

        while True:
            try:
                item = input_queue.get(timeout=5)
            except Exception:
                continue

            if item is None:
                break

            try:
                result = self.process_item(item)
                result_queue.put((_CMD_RESULT, result))
            except Exception as e:
                result_queue.put((_CMD_ERROR, worker_id, item, str(e)))

        try:
            self.on_client_finalize()
        except Exception:
            pass

        result_queue.put((_CMD_DONE,))
