from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from DeepFaceLab.core.metadata_manager import MetadataManager, FaceMetadata
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("face_enhancer")


class FaceEnhancer:
    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from DeepFaceLab.models.xseg_model import XSegModel
            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = XSegModel(resolution=256, base_ch=16).to(device)
            self._model.eval()
            _logger.info("FaceEnhancer model initialized (using lightweight XSeg-based enhancer).")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FaceEnhancer: {e}") from e

    def enhance(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        import torch
        self._ensure_model()
        from DeepFaceLab.shared.image_utils import ImageUtils
        tensor = ImageUtils.numpy_to_tensor(img, device=str(next(self._model.parameters()).device))
        with torch.inference_mode():
            enhanced = self._model(tensor.unsqueeze(0)).squeeze(0)
        enhanced_np = ImageUtils.tensor_to_numpy(enhanced.cpu())
        blended = cv2_add_weighted(img, enhanced_np, 0.5)
        return blended

    def process_folder(self, aligned_dir: Path, output_dir: Optional[Path] = None) -> int:
        aligned_dir = Path(aligned_dir)
        out_dir = Path(output_dir) if output_dir else aligned_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in FileManager.find_images(aligned_dir):
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            enhanced = self.enhance(img)
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), enhanced)
            count += 1

        _logger.info(f"Enhanced {count} faces in {aligned_dir}")
        return count


def cv2_add_weighted(img1: npt.NDArray[np.uint8], img2: npt.NDArray[np.uint8], alpha: float) -> npt.NDArray[np.uint8]:
    import cv2
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    blended = cv2.addWeighted(img1, 1.0 - alpha, img2, alpha, 0)
    return blended
