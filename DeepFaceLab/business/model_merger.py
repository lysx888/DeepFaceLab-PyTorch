from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from DeepFaceLab.core.insightface_adapter import InsightFaceAdapter, DetectedFace
from DeepFaceLab.core.metadata_manager import MetadataManager, FaceMetadata
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("model_merger")


class MaskMode(Enum):
    DST = "dst"
    XSEG = "xseg"
    LEARNED = "learned"


class ColorTransferMode(Enum):
    NONE = "none"
    LAB = "lab"
    RCT = "rct"


class SharpenMode(Enum):
    NONE = "none"
    BOX = "box"
    GAUSSIAN = "gaussian"


@dataclass
class MergeConfig:
    mask_mode: MaskMode = MaskMode.XSEG
    erode_mask_modifier: float = 0.0
    blur_mask_modifier: float = 0.0
    color_transfer_mode: ColorTransferMode = ColorTransferMode.NONE
    sharpen_mode: SharpenMode = SharpenMode.NONE
    use_inswapper: bool = False
    output_size: int = 256


class ModelMerger:
    def __init__(
        self,
        adapter: Optional[InsightFaceAdapter] = None,
        device: str = "auto",
    ) -> None:
        self._adapter = adapter
        self._device_str = device
        self._device = None

    def _resolve_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        if self._device_str == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self._device_str)
        return self._device

    def merge_auto(
        self,
        model_dir: Path,
        model_type: str,
        input_dir: Path,
        output_dir: Path,
        output_mask_dir: Path,
        aligned_dir: Path,
        config: MergeConfig,
        src_aligned_dir: Optional[Path] = None,
    ) -> int:
        device = self._resolve_device()
        model_dir = Path(model_dir)
        output_dir = Path(output_dir)
        output_mask_dir = Path(output_mask_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_mask_dir.mkdir(parents=True, exist_ok=True)

        encoder, decoder_src, decoder_dst, inter = self._load_model(model_dir, model_type, device)

        images = FileManager.find_images(input_dir)
        if not images:
            raise ValueError(f"No frames found in {input_dir}")

        aligned_meta = MetadataManager.load_all(aligned_dir)

        count = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            meta = aligned_meta.get(img_path.name)
            if meta is None:
                cv2.imwrite(str(output_dir / img_path.name), img)
                continue

            merged = self._merge_single(
                img, meta, encoder, decoder_src, decoder_dst, inter,
                device, config, aligned_dir, src_aligned_dir,
            )
            cv2.imwrite(str(output_dir / img_path.name), merged)

            mask = self._get_mask(img.shape[:2], meta, config, aligned_dir, img_path.name)
            if mask is not None:
                cv2.imwrite(str(output_mask_dir / img_path.name), mask)
            count += 1

        _logger.info(f"Merged {count} frames to {output_dir}")
        return count

    def merge_interactive(
        self,
        model_dir: Path,
        model_type: str,
        input_dir: Path,
        output_dir: Path,
        output_mask_dir: Path,
        aligned_dir: Path,
        config: MergeConfig,
        src_aligned_dir: Optional[Path] = None,
    ) -> int:
        _logger.info("Interactive merge - using auto merge as base")
        return self.merge_auto(model_dir, model_type, input_dir, output_dir, output_mask_dir, aligned_dir, config, src_aligned_dir=src_aligned_dir)

    def merge_inswapper(
        self,
        input_dir: Path,
        output_dir: Path,
        src_aligned_dir: Path,
        dst_aligned_dir: Path,
    ) -> int:
        if self._adapter is None:
            raise RuntimeError("InsightFaceAdapter required for INSwapper merge")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        src_images = FileManager.find_images(src_aligned_dir)
        if not src_images:
            raise ValueError(f"No source faces found in {src_aligned_dir}")

        src_img_path = src_images[0]
        src_img = cv2.imread(str(src_img_path))
        src_faces = self._adapter.detect_faces(src_img)
        if not src_faces:
            raise ValueError("No face detected in source image")
        source_face = src_faces[0]

        dst_images = FileManager.find_images(dst_aligned_dir)
        count = 0
        for img_path in dst_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dst_faces = self._adapter.detect_faces(img)
            if not dst_faces:
                cv2.imwrite(str(output_dir / img_path.name), img)
                continue

            result = img.copy()
            for dst_face in dst_faces:
                swapped = self._adapter.swap_face(result, dst_face, source_face, paste_back=True)
                result = swapped
            cv2.imwrite(str(output_dir / img_path.name), result)
            count += 1

        _logger.info(f"INSwapper merged {count} frames to {output_dir}")
        return count

    def _load_model(self, model_dir: Path, model_type: str, device: torch.device):
        from DeepFaceLab.models.saehd_model import SAEHDEncoder, SAEHDDecoder, SAEHDInter
        from DeepFaceLab.models.quick96_model import Quick96Encoder, Quick96Decoder
        from DeepFaceLab.models.amp_model import AMPEncoder, AMPDecoder

        config_path = model_dir / f"{model_type}_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        import json
        with open(str(config_path), "r") as f:
            cfg = json.load(f)

        resolution = cfg.get("resolution", 128)
        arch = cfg.get("architecture", "df")

        if model_type == "SAEHD":
            down_steps = 0
            r = resolution
            while r > 4:
                r //= 2
                down_steps += 1
            base_ch = cfg.get("base_ch", 64)
            use_liae = arch == "liae"

            encoder = SAEHDEncoder(in_ch=3, base_ch=base_ch, down_steps=down_steps).to(device)
            enc_out = encoder.out_ch
            decoder_src = SAEHDDecoder(enc_out, 3, down_steps, use_liae, enc_out if use_liae else 0).to(device)
            decoder_dst = SAEHDDecoder(enc_out, 3, down_steps, use_liae, enc_out if use_liae else 0).to(device)
            inter = SAEHDInter(enc_out, enc_out).to(device) if use_liae else None

            enc_path = model_dir / f"{model_type}_encoder.pt"
            if enc_path.exists():
                encoder.load_state_dict(torch.load(str(enc_path), map_location=device, weights_only=True))
            src_path = model_dir / f"{model_type}_decoder_src.pt"
            if src_path.exists():
                decoder_src.load_state_dict(torch.load(str(src_path), map_location=device, weights_only=True))
            dst_path = model_dir / f"{model_type}_decoder_dst.pt"
            if dst_path.exists():
                decoder_dst.load_state_dict(torch.load(str(dst_path), map_location=device, weights_only=True))
            if inter is not None:
                inter_path = model_dir / f"{model_type}_inter.pt"
                if inter_path.exists():
                    inter.load_state_dict(torch.load(str(inter_path), map_location=device, weights_only=True))

            encoder.eval()
            decoder_src.eval()
            decoder_dst.eval()
            if inter is not None:
                inter.eval()
            return encoder, decoder_src, decoder_dst, inter

        elif model_type == "Quick96":
            base_ch = cfg.get("base_ch", 32)
            encoder = Quick96Encoder(in_ch=3, base_ch=base_ch).to(device)
            enc_out = encoder.out_ch
            decoder_src = Quick96Decoder(enc_out, 3).to(device)
            decoder_dst = Quick96Decoder(enc_out, 3).to(device)
            return self._load_weights(model_dir, model_type, encoder, decoder_src, decoder_dst, device)

        elif model_type == "AMP":
            base_ch = cfg.get("base_ch", 64)
            down_steps = 0
            r = resolution
            while r > 4:
                r //= 2
                down_steps += 1
            encoder = AMPEncoder(in_ch=3, base_ch=base_ch, down_steps=down_steps).to(device)
            enc_out = encoder.out_ch
            decoder_src = AMPDecoder(enc_out, 3, down_steps).to(device)
            decoder_dst = AMPDecoder(enc_out, 3, down_steps).to(device)
            return self._load_weights(model_dir, model_type, encoder, decoder_src, decoder_dst, device)

        elif model_type == "TFM":
            return self._load_tfm_model(model_dir, device)

        raise ValueError(f"Unknown model type: {model_type}")

    def _load_weights(self, model_dir, model_type, encoder, decoder_src, decoder_dst, device):
        enc_path = model_dir / f"{model_type}_encoder.pt"
        if enc_path.exists():
            encoder.load_state_dict(torch.load(str(enc_path), map_location=device, weights_only=True))
        src_path = model_dir / f"{model_type}_decoder_src.pt"
        if src_path.exists():
            decoder_src.load_state_dict(torch.load(str(src_path), map_location=device, weights_only=True))
        dst_path = model_dir / f"{model_type}_decoder_dst.pt"
        if dst_path.exists():
            decoder_dst.load_state_dict(torch.load(str(dst_path), map_location=device, weights_only=True))
        encoder.eval()
        decoder_src.eval()
        decoder_dst.eval()
        return encoder, decoder_src, decoder_dst, None

    def _load_tfm_model(self, model_dir: Path, device: torch.device):
        from DeepFaceLab.models.tfm_model import TFMModel
        import json

        config_path = model_dir / "TFM_model_config.json"
        if config_path.exists():
            with open(str(config_path), "r") as f:
                cfg = json.load(f)
        else:
            cfg = {}

        model = TFMModel(
            resolution=cfg.get("resolution", 128),
            embed_dim=cfg.get("embed_dim", 96),
            depths=cfg.get("depths", [2, 2, 6, 2]),
            num_heads=cfg.get("num_heads", [3, 6, 12, 24]),
            window_size=cfg.get("window_size", 8),
            w_dim=cfg.get("w_dim", 512),
            identity_dim=cfg.get("identity_dim", 512),
            gan_power=0.0,
            base_channels=cfg.get("base_channels", 512),
        ).to(device)

        model.load(model_dir, device)
        model.eval()
        return model, None, None, None

    def _merge_single(self, img, meta, encoder, decoder_src, decoder_dst, inter, device, config, aligned_dir, src_aligned_dir=None):
        from DeepFaceLab.core.insightface_adapter import InsightFaceAdapter
        from DeepFaceLab.setting import FaceType, FACE_TYPE_SCALE
        from DeepFaceLab.models.tfm_model import TFMModel

        if isinstance(encoder, TFMModel):
            return self._merge_single_tfm(img, meta, encoder, device, config, aligned_dir, src_aligned_dir)

        adapter = self._adapter or InsightFaceAdapter()
        aligned = adapter.align_face(img, meta.landmarks_106, meta.face_type, config.output_size)

        img_t = torch.from_numpy(aligned.image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        img_t = img_t * 2.0 - 1.0

        with torch.inference_mode():
            enc = encoder(img_t)
            if inter is not None:
                latent = inter(enc)
                pred = decoder_dst(enc, latent)
            else:
                pred = decoder_dst(enc)

        pred_np = ((pred.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

        mask = self._get_mask(aligned.image.shape[:2], meta, config, aligned_dir, "")
        if mask is not None:
            mask_3ch = cv2.merge([mask, mask, mask])
            result = cv2.seamlessClone(pred_bgr, img, mask_3ch, (img.shape[1] // 2, img.shape[0] // 2), cv2.NORMAL_CLONE)
        else:
            result = pred_bgr
        return result

    def _merge_single_tfm(self, img, meta, model, device, config, aligned_dir, src_aligned_dir=None):
        from DeepFaceLab.core.insightface_adapter import InsightFaceAdapter

        adapter = self._adapter or InsightFaceAdapter()
        aligned = adapter.align_face(img, meta.landmarks_106, meta.face_type, config.output_size)

        img_rgb = cv2.cvtColor(aligned.image, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        img_t = img_t * 2.0 - 1.0

        identity = torch.zeros(1, 512, dtype=torch.float32, device=device)
        if meta.arcface_embedding is not None:
            identity = torch.from_numpy(meta.arcface_embedding.astype(np.float32)).unsqueeze(0).to(device)
        elif src_aligned_dir is not None:
            src_images = FileManager.find_images(Path(src_aligned_dir))
            if src_images:
                src_img = cv2.imread(str(src_images[0]))
                if src_img is not None:
                    src_faces = adapter.detect_faces(src_img, max_num=1)
                    if src_faces and src_faces[0].embedding is not None:
                        identity = torch.from_numpy(src_faces[0].embedding.astype(np.float32)).unsqueeze(0).to(device)

        with torch.inference_mode():
            pred = model(img_t, identity)

        pred_np = ((pred.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

        mask = self._get_mask(aligned.image.shape[:2], meta, config, aligned_dir, "")
        if mask is not None:
            mask_3ch = cv2.merge([mask, mask, mask])
            result = cv2.seamlessClone(pred_bgr, img, mask_3ch, (img.shape[1] // 2, img.shape[0] // 2), cv2.NORMAL_CLONE)
        else:
            result = pred_bgr
        return result

    def _get_mask(self, shape, meta, config, aligned_dir, filename):
        h, w = shape[:2]
        if config.mask_mode in (MaskMode.XSEG, MaskMode.LEARNED) and meta is not None and meta.seg_ie_polys is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly_data in meta.seg_ie_polys:
                pts_data = poly_data.get("pts", [])
                poly_type = poly_data.get("type", 1)
                if len(pts_data) < 3:
                    continue
                pts = np.array(pts_data, dtype=np.int32)
                if poly_type == 1:
                    cv2.fillPoly(mask, [pts], 255)
                else:
                    cv2.fillPoly(mask, [pts], 0)
            return mask
        return np.full((h, w), 255, dtype=np.uint8)
