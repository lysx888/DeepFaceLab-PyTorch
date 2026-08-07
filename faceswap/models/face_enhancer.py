from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("face_enhancer")

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights" / "facerestore"

MODEL_REGISTRY = {
    "gfpgan_1.4": {
        "file": "GFPGANv1.4.onnx",
        "size": 512,
    },
    "gpen_bfr_512": {
        "file": "GPEN-BFR-512.onnx",
        "size": 512,
    },
    "gpen_bfr_1024": {
        "file": "GPEN-BFR-1024.onnx",
        "size": 1024,
    },
    "gpen_bfr_2048": {
        "file": "GPEN-BFR-2048.onnx",
        "size": 2048,
    },
    "restoreformer_pp": {
        "file": "RestoreFormer_PP.onnx",
        "size": 512,
    },
}


def _get_providers(device: str = "auto") -> list[str]:
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class FaceEnhancer:
    def __init__(
        self,
        model_name: str = "gfpgan_1.4",
        device: str = "auto",
        blend: int = 80,
        weight: float = 0.5,
    ) -> None:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
            )
        self.model_name = model_name
        self.blend = blend
        self.weight = weight
        self._info = MODEL_REGISTRY[model_name]
        self._model_size = self._info["size"]

        model_path = WEIGHTS_DIR / self._info["file"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model weight not found: {model_path}")

        providers = _get_providers(device)
        self._session = ort.InferenceSession(
            str(model_path), providers=providers
        )

        input_names = {inp.name for inp in self._session.get_inputs()}
        self._has_weight_input = "weight" in input_names
        if self._has_weight_input:
            _logger.info(f"  Model supports 'weight' input (repair strength)")

        _logger.info(
            f"Loaded {model_name} from {model_path} (providers={providers})"
        )

    def enhance(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = img.shape[:2]
        crop = self._prepare(img)
        output = self._forward(crop)
        enhanced_crop = self._postprocess(output)

        if mask is not None:
            result = self._paste_back(img, enhanced_crop, mask)
        else:
            if h == w == self._model_size:
                result = enhanced_crop
            else:
                result = cv2.resize(enhanced_crop, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.blend < 100:
            blend_factor = self.blend / 100.0
            result = cv2.addWeighted(img, 1.0 - blend_factor, result, blend_factor, 0)

        return result

    def process_folder(
        self,
        aligned_dir: Path,
        mask: Optional[np.ndarray] = None,
    ) -> int:
        aligned_dir = Path(aligned_dir)
        count = 0
        for img_path in FileManager.find_images(aligned_dir):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            enhanced = self.enhance(img, mask)
            cv2.imwrite(str(img_path), enhanced)
            count += 1
        _logger.info(f"Enhanced {count} faces in {aligned_dir} with {self.model_name}")
        return count

    def _prepare(self, img: np.ndarray) -> np.ndarray:
        size = self._model_size
        if img.shape[0] != size or img.shape[1] != size:
            resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img
        rgb = resized[:, :, ::-1].copy()
        normalized = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        nchw = np.transpose(normalized, (2, 0, 1))
        return nchw[np.newaxis, :, :, :].astype(np.float32)

    def _forward(self, crop: np.ndarray) -> np.ndarray:
        inputs = {"input": crop}
        if self._has_weight_input:
            inputs["weight"] = np.array([self.weight], dtype=np.float64)
        outputs = self._session.run(None, inputs)
        return outputs[0][0]

    def _postprocess(self, output: np.ndarray) -> np.ndarray:
        result = np.clip(output, -1.0, 1.0)
        result = (result + 1.0) / 2.0
        result = np.transpose(result, (1, 2, 0))
        result = (result * 255.0).astype(np.uint8)
        return result[:, :, ::-1]

    def _paste_back(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        h, w = original.shape[:2]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 3:
            mask_resized = mask_resized[:, :, 0]
        mask_f = mask_resized.astype(np.float32) / 255.0
        enhanced_resized = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_3c = mask_f[:, :, np.newaxis]
        result = (original.astype(np.float32) * (1.0 - mask_3c)
                  + enhanced_resized.astype(np.float32) * mask_3c)
        return result.astype(np.uint8)

    def __del__(self) -> None:
        if hasattr(self, "_session"):
            del self._session
