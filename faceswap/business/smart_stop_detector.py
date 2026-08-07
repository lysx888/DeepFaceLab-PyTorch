from collections import deque
from typing import Optional

from faceswap.shared.logger import get_logger

_logger = get_logger("smart_stop_detector")


class SmartStopDetector:
    def __init__(self, window: int = 500, threshold: float = 0.1,
                 enabled: bool = True,
                 min_iters: int = 10000) -> None:
        self._window = window
        self._threshold = threshold
        self._enabled = enabled
        self._min_iters = min_iters
        self._loss_history: deque[float] = deque(maxlen=window)
        self._ssim_history: deque[float] = deque(maxlen=window)
        self._converged = False
        self._notified = False

    def update(self, iter_count: int, loss_value: float,
               ssim_value: Optional[float] = None) -> None:
        if not self._enabled:
            return
        self._loss_history.append(loss_value)
        if ssim_value is not None:
            self._ssim_history.append(ssim_value)

        if len(self._loss_history) >= self._window:
            self._check_convergence(iter_count)

    def _check_convergence(self, iter_count: int) -> None:
        if iter_count < self._min_iters:
            return
        loss_converged = self._check_series(self._loss_history)
        ssim_converged = True
        if len(self._ssim_history) >= self._window:
            ssim_converged = self._check_series(self._ssim_history)

        if loss_converged and ssim_converged:
            if not self._notified:
                self._converged = True
                self._notified = True
                _logger.info(
                    f"SmartStop: 训练可能已收敛 (iter={iter_count}, "
                    f"loss改善<{self._threshold}%). 建议停止或降低学习率")

    def _check_series(self, series: deque) -> bool:
        if len(series) < self._window:
            return False
        vals = list(series)
        half = len(vals) // 2
        first_half_min = min(vals[:half])
        second_half_min = min(vals[half:])
        if first_half_min == 0:
            return second_half_min == 0
        improvement = abs(first_half_min - second_half_min) / abs(first_half_min) * 100
        return improvement < self._threshold

    def is_converged(self) -> bool:
        return self._converged

    @property
    def should_lower_lr(self) -> bool:
        return self._converged

    def reset(self) -> None:
        self._loss_history.clear()
        self._ssim_history.clear()
        self._converged = False
        self._notified = False
