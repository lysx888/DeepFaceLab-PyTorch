import torch
from faceswap.shared.logger import get_logger

_logger = get_logger("amp_utils")


class AMPManager:
    def __init__(self, device: torch.device, amp_mode: str = "fp32"):
        from faceswap.shared.config import is_gpu_device
        self._is_gpu = is_gpu_device(device)
        self._amp_mode = amp_mode
        self._device = device
        self._bf16_supported = False

        if amp_mode == "bf16" and self._is_gpu:
            if self._check_bf16_support():
                self._enabled = True
                self._dtype = torch.bfloat16
                self._scaler = torch.amp.GradScaler(device.type, enabled=False)
                self._bf16_supported = True
            else:
                _logger.warning("当前GPU不支持BF16，已自动降级为FP32")
                self._enabled = False
                self._dtype = torch.float32
                self._scaler = torch.amp.GradScaler(device.type, enabled=False)
        elif amp_mode == "fp16" and self._is_gpu:
            self._enabled = True
            self._dtype = torch.float16
            self._scaler = torch.amp.GradScaler(device.type, init_scale=4096.0)
            self._scaler.set_growth_factor(1.25)
            self._scaler.set_backoff_factor(0.75)
        else:
            self._enabled = False
            self._dtype = torch.float32
            self._scaler = torch.amp.GradScaler(device.type, enabled=False)

        if self._enabled:
            _logger.info(f"AMP enabled: mode={amp_mode}, dtype={self._dtype}, device={device}")
        else:
            _logger.info(f"AMP disabled (mode={amp_mode}, gpu={self._is_gpu})")

    def _check_bf16_support(self) -> bool:
        try:
            if self._device.type == 'cuda' and torch.cuda.is_available():
                cap = torch.cuda.get_device_capability(self._device)
                return cap[0] >= 8
        except Exception:
            pass
        return False

    @property
    def is_bf16_supported(self) -> bool:
        return self._bf16_supported

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def scaler(self) -> torch.amp.GradScaler:
        return self._scaler

    @property
    def device_type(self) -> str:
        return self._device.type

    def autocast(self):
        return torch.amp.autocast(device_type=self._device.type, dtype=self._dtype, enabled=self._enabled)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return self._scaler.scale(loss)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self._scaler.step(optimizer)

    def update(self) -> None:
        self._scaler.update()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self._scaler.unscale_(optimizer)
