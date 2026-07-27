from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from DeepFaceLab.shared.torch_config import get_dataloader_config, get_non_blocking, worker_init_fn

from DeepFaceLab.core.metadata_manager import MetadataManager
from DeepFaceLab.core.insightface_adapter import InsightFaceAdapter
from DeepFaceLab.models.saehd_model import SAEHDModel
from DeepFaceLab.models.quick96_model import Quick96Model
from DeepFaceLab.models.amp_model import AMPModel
from DeepFaceLab.models.faceset_dataset import FacesetDataset
from DeepFaceLab.shared.file_manager import FileManager
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("model_trainer")


class ModelType(Enum):
    SAEHD = "SAEHD"
    QUICK96 = "Quick96"
    AMP = "AMP"
    TFM = "TFM"


_SAEHD_PARAMS = [
    {"key": "resolution",       "label": "Resolution",              "type": int,   "default": 128,  "min": 64,  "max": 640,  "help": "Higher resolution = more VRAM. Adjusted to multiple of 16."},
    {"key": "face_type",        "label": "Face type",               "type": str,   "default": "whole_face", "choices": ["half", "mid_full", "full", "whole_face", "head"], "help": "half/mid_full/full/whole_face/head"},
    {"key": "architecture",     "label": "AE architecture",         "type": str,   "default": "df", "choices": ["df", "liae"], "help": "df = more identity, liae = can fix different face shapes"},
    {"key": "ae_dims",          "label": "AutoEncoder dimensions",  "type": int,   "default": 256,  "min": 32,  "max": 1024, "help": "More dims = better quality but more VRAM"},
    {"key": "e_dims",           "label": "Encoder dimensions",      "type": int,   "default": 64,   "min": 16,  "max": 256,  "help": "More dims = sharper result but more VRAM"},
    {"key": "d_dims",           "label": "Decoder dimensions",      "type": int,   "default": 64,   "min": 16,  "max": 256,  "help": "More dims = sharper result but more VRAM"},
    {"key": "batch_size",       "label": "Batch size",              "type": int,   "default": 4,    "min": 1,   "max": 64,   "help": "Higher = more VRAM. 4GB VRAM -> 4, 8GB+ -> 8"},
    {"key": "learning_rate",    "label": "Learning rate",           "type": float, "default": 1e-4, "min": 1e-6, "max": 1e-2, "help": "Typical: 1e-4"},
    {"key": "use_amp",          "label": "Use AMP (mixed precision)", "type": bool, "default": True, "help": "Faster training, less VRAM"},
    {"key": "random_warp",      "label": "Enable random warp",      "type": bool,  "default": True,  "help": "Required for generalization. Disable for extra sharpness late in training"},
    {"key": "gan_power",        "label": "GAN power",               "type": float, "default": 0.0,  "min": 0.0, "max": 5.0,  "help": "0=off. Typical fine value 0.1. Enable only when face is trained enough"},
    {"key": "random_hsv_power", "label": "Random hue/sat/light",    "type": float, "default": 0.0,  "min": 0.0, "max": 0.3,  "help": "Stabilizes color. Typical: 0.05"},
]

_QUICK96_PARAMS = [
    {"key": "batch_size",    "label": "Batch size",        "type": int,   "default": 4,   "min": 1,  "max": 64,  "help": "Higher = more VRAM"},
    {"key": "learning_rate", "label": "Learning rate",     "type": float, "default": 1e-4, "min": 1e-6, "max": 1e-2, "help": "Typical: 1e-4"},
    {"key": "use_amp",       "label": "Use AMP",           "type": bool,  "default": True,  "help": "Faster training, less VRAM"},
]

_AMP_PARAMS = [
    {"key": "resolution",   "label": "Resolution",     "type": int,   "default": 128,  "min": 64,  "max": 640,  "help": "Higher = more VRAM"},
    {"key": "batch_size",   "label": "Batch size",     "type": int,   "default": 4,    "min": 1,   "max": 64,   "help": "Higher = more VRAM"},
    {"key": "learning_rate","label": "Learning rate",  "type": float, "default": 1e-4, "min": 1e-6, "max": 1e-2, "help": "Typical: 1e-4"},
    {"key": "use_amp",      "label": "Use AMP",        "type": bool,  "default": True,  "help": "Faster training, less VRAM"},
    {"key": "src_src_mode", "label": "SRC-SRC mode",   "type": bool,  "default": False, "help": "Train with source-to-source mapping"},
]

_TFM_PARAMS = [
    {"key": "resolution",        "label": "Resolution",              "type": int,   "default": 128,  "min": 64,  "max": 256,  "help": "Must be multiple of 16"},
    {"key": "face_type",         "label": "Face type",               "type": str,   "default": "whole_face", "choices": ["half", "mid_full", "full", "whole_face", "head"], "help": "Face crop type"},
    {"key": "batch_size",        "label": "Batch size",              "type": int,   "default": 4,    "min": 1,   "max": 64,   "help": "Higher = more VRAM"},
    {"key": "learning_rate",     "label": "Learning rate",           "type": float, "default": 1e-4, "min": 1e-6, "max": 1e-2, "help": "Typical: 1e-4"},
    {"key": "use_amp",           "label": "Use AMP",                 "type": bool,  "default": True,  "help": "Mixed precision training"},
    {"key": "random_warp",       "label": "Random warp",             "type": bool,  "default": True,  "help": "Random geometric augmentation"},
    {"key": "gan_power",         "label": "GAN power",               "type": float, "default": 0.0,  "min": 0.0, "max": 5.0,  "help": "0=off, typical: 0.1"},
    {"key": "random_hsv_power",  "label": "Random HSV power",        "type": float, "default": 0.0,  "min": 0.0, "max": 0.3,  "help": "Color augmentation"},
    {"key": "lr_schedule",       "label": "LR schedule",             "type": str,   "default": "constant", "choices": ["constant", "cosine_annealing"], "help": "Learning rate schedule"},
    {"key": "gradient_clip",     "label": "Gradient clip",           "type": float, "default": 1.0,  "min": 0.0, "max": 10.0, "help": "0=off"},
    {"key": "random_flip",       "label": "Random flip",             "type": bool,  "default": True,  "help": "Random horizontal flip"},
    {"key": "color_transfer",    "label": "Color transfer",          "type": str,   "default": "none", "choices": ["none", "rct", "mkl"], "help": "Color transfer mode"},
    {"key": "model_preset",      "label": "Model preset",            "type": str,   "default": "medium", "choices": ["tiny", "small", "medium", "large"], "help": "Controls model size and VRAM"},
    {"key": "window_size",       "label": "Window size",             "type": int,   "default": 8,    "min": 4,   "max": 16,   "help": "Swin window attention size"},
    {"key": "skip_strength",     "label": "Skip strength",           "type": float, "default": 0.5,  "min": 0.0, "max": 1.0,  "help": "Encoder skip connection strength"},
    {"key": "gradient_checkpoint","label": "Gradient checkpoint",    "type": bool,  "default": False, "help": "Trade compute for VRAM"},
    {"key": "eye_priority",      "label": "Eye priority",            "type": float, "default": 1.0,  "min": 0.5, "max": 5.0,  "help": "Eye region loss weight"},
    {"key": "mouth_priority",    "label": "Mouth priority",          "type": float, "default": 1.0,  "min": 0.5, "max": 5.0,  "help": "Mouth region loss weight"},
    {"key": "nose_priority",     "label": "Nose priority",           "type": float, "default": 1.0,  "min": 0.5, "max": 3.0,  "help": "Nose region loss weight"},
    {"key": "jaw_priority",      "label": "Jaw priority",            "type": float, "default": 1.0,  "min": 0.5, "max": 3.0,  "help": "Jaw region loss weight"},
    {"key": "face_style_power",  "label": "Face style power",        "type": float, "default": 2.0,  "min": 0.0, "max": 5.0,  "help": "Swap face region loss weight"},
    {"key": "bg_style_power",    "label": "BG style power",          "type": float, "default": 1.0,  "min": 0.0, "max": 5.0,  "help": "Swap background region loss weight"},
    {"key": "perceptual_weight", "label": "Perceptual loss weight",   "type": float, "default": 0.0,  "min": 0.0, "max": 1.0,  "help": "VGG perceptual loss"},
    {"key": "identity_weight",   "label": "Identity loss weight",    "type": float, "default": 0.0,  "min": 0.0, "max": 1.0,  "help": "ArcFace identity preservation"},
    {"key": "uniform_yaw_sampling","label": "Uniform yaw sampling",  "type": bool,  "default": False, "help": "Uniform angle sampling"},
]

_MODEL_PARAMS = {
    ModelType.SAEHD: _SAEHD_PARAMS,
    ModelType.QUICK96: _QUICK96_PARAMS,
    ModelType.AMP: _AMP_PARAMS,
    ModelType.TFM: _TFM_PARAMS,
}


def _suggest_batch_size() -> int:
    if not torch.cuda.is_available():
        return 2
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        if vram_gb >= 8:
            return 8
        elif vram_gb >= 4:
            return 4
        else:
            return 2
    except Exception:
        return 4


class ModelTrainer:
    def __init__(self, device: str = "auto") -> None:
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

    def _create_model(self, model_type: ModelType, config: dict) -> "BaseModel":
        if model_type == ModelType.SAEHD:
            return SAEHDModel.from_config(config)
        elif model_type == ModelType.QUICK96:
            return Quick96Model.from_config(config)
        elif model_type == ModelType.AMP:
            return AMPModel.from_config(config)
        elif model_type == ModelType.TFM:
            from DeepFaceLab.models.tfm_model import TFMModel
            return TFMModel.from_preset(
                preset=config.get("model_preset", "medium"),
                resolution=config.get("resolution", 128),
                gan_power=config.get("gan_power", 0.0),
                window_size=config.get("window_size", 8),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _load_saved_config(model_type: ModelType, model_dir: Path) -> Optional[dict]:
        config_path = Path(model_dir) / f"{model_type.value}_training_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    @staticmethod
    def _save_training_config(model_type: ModelType, model_dir: Path, config: dict) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_dir / f"{model_type.value}_training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def configure_interactive(self, model_type: ModelType, model_dir: Path = None) -> dict:
        params = _MODEL_PARAMS.get(model_type, [])
        if not params:
            return {}

        saved = None
        if model_dir is not None:
            saved = self._load_saved_config(model_type, model_dir)

        suggest_bs = _suggest_batch_size()
        for p in params:
            if p["key"] == "batch_size" and saved is None:
                p["default"] = suggest_bs

        config = {}
        for p in params:
            key = p["key"]
            if saved and key in saved:
                default_val = saved[key]
            else:
                default_val = p["default"]
            config[key] = default_val

        print(f"\n===== {model_type.value} Training Configuration =====")
        if saved:
            print("  (Loaded from saved config)")
        print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
        if torch.cuda.is_available():
            vram_gb = getattr(torch.cuda.get_device_properties(0), 'total_memory', getattr(torch.cuda.get_device_properties(0), 'total_mem', 0)) / (1024 ** 3)
            print(f"  VRAM: {vram_gb:.1f} GB")
        print()

        for i, p in enumerate(params, 1):
            key = p["key"]
            val = config[key]
            label = p["label"]
            help_msg = p.get("help", "")

            choices = p.get("choices")
            ptype = p["type"]

            if ptype == bool:
                display_val = "y" if val else "n"
                prompt = f"  [{i}] {label} ({display_val})"
            elif ptype == float:
                prompt = f"  [{i}] {label} ({val})"
            else:
                prompt = f"  [{i}] {label} ({val})"

            if choices:
                prompt += f"  [{'/'.join(str(c) for c in choices)}]"
            if help_msg:
                prompt += f"  - {help_msg}"

            while True:
                try:
                    user_input = input(prompt + " : ").strip()
                except EOFError:
                    user_input = ""

                if not user_input:
                    break

                try:
                    if ptype == bool:
                        if user_input.lower() in ("y", "yes", "1", "true"):
                            config[key] = True
                            break
                        elif user_input.lower() in ("n", "no", "0", "false"):
                            config[key] = False
                            break
                        else:
                            print(f"    Please enter y/n")
                            continue
                    elif ptype == int:
                        new_val = int(user_input)
                        pmin = p.get("min")
                        pmax = p.get("max")
                        if pmin is not None and new_val < pmin:
                            print(f"    Minimum value: {pmin}")
                            continue
                        if pmax is not None and new_val > pmax:
                            print(f"    Maximum value: {pmax}")
                            continue
                        if key == "resolution":
                            new_val = (new_val // 16) * 16
                            if new_val < 64:
                                new_val = 64
                        config[key] = new_val
                        break
                    elif ptype == float:
                        new_val = float(user_input)
                        pmin = p.get("min")
                        pmax = p.get("max")
                        if pmin is not None and new_val < pmin:
                            print(f"    Minimum value: {pmin}")
                            continue
                        if pmax is not None and new_val > pmax:
                            print(f"    Maximum value: {pmax}")
                            continue
                        config[key] = new_val
                        break
                    elif ptype == str:
                        if choices and user_input not in choices:
                            print(f"    Valid choices: {choices}")
                            continue
                        config[key] = user_input
                        break
                except ValueError:
                    print(f"    Invalid input for {ptype.__name__}")
                    continue

        print("\n===== Configuration Summary =====")
        for p in params:
            key = p["key"]
            val = config[key]
            if p["type"] == bool:
                print(f"  {p['label']:30s}: {'Yes' if val else 'No'}")
            elif p["type"] == float and val < 0.01:
                print(f"  {p['label']:30s}: {val:.6f}")
            else:
                print(f"  {p['label']:30s}: {val}")
        print("=================================\n")

        try:
            confirm = input("Press Enter to start training, or 'q' to cancel: ").strip().lower()
        except EOFError:
            confirm = ""
        if confirm == "q":
            raise SystemExit(0)

        return config

    def train(
        self,
        model_type: ModelType,
        src_aligned_dir: Path,
        dst_aligned_dir: Path,
        model_dir: Path,
        resolution: int = 128,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        use_amp: bool = True,
        save_interval_min: float = 15.0,
        preview_interval_min: float = 15.0,
        num_workers: int = 0,
        architecture: str = "df",
        src_src_mode: bool = False,
        face_type: str = "whole_face",
        ae_dims: int = 256,
        e_dims: int = 64,
        d_dims: int = 64,
        random_warp: bool = True,
        gan_power: float = 0.0,
        random_hsv_power: float = 0.0,
    ) -> None:
        device = self._resolve_device()
        src_dir = Path(src_aligned_dir)
        dst_dir = Path(dst_aligned_dir)
        model_dir = Path(model_dir)

        src_dataset = FacesetDataset(src_dir, resolution=resolution, is_src=True)
        dst_dataset = FacesetDataset(dst_dir, resolution=resolution, is_src=False)

        if len(src_dataset) == 0:
            raise ValueError(f"No source faces found in {src_dir}")
        if len(dst_dataset) == 0:
            raise ValueError(f"No destination faces found in {dst_dir}")

        total_size = len(src_dataset) + len(dst_dataset)
        dl_cfg = get_dataloader_config("gpu_train" if device.type == "cuda" else "cpu_train", dataset_size=total_size)
        _wk_init = worker_init_fn if dl_cfg["num_workers"] > 0 else None
        src_loader = DataLoader(
            src_dataset, batch_size=batch_size, shuffle=True,
            num_workers=dl_cfg["num_workers"], pin_memory=dl_cfg["pin_memory"], drop_last=True,
            worker_init_fn=_wk_init,
        )
        dst_loader = DataLoader(
            dst_dataset, batch_size=batch_size, shuffle=True,
            num_workers=dl_cfg["num_workers"], pin_memory=dl_cfg["pin_memory"], drop_last=True,
            worker_init_fn=_wk_init,
        )

        config = {
            "resolution": resolution,
            "architecture": architecture,
            "src_src_mode": src_src_mode,
        }
        model = self._create_model(model_type, config)

        encoder = model.get_encoder().to(device)
        decoder_src = model.get_decoder_src().to(device)
        decoder_dst = model.get_decoder_dst().to(device)
        inter = model.get_inter()
        if inter is not None:
            inter = inter.to(device)

        all_params = list(encoder.parameters()) + list(decoder_src.parameters()) + list(decoder_dst.parameters())
        if inter is not None:
            all_params += list(inter.parameters())
        optimizer = torch.optim.Adam(all_params, lr=learning_rate)

        scaler = None
        if use_amp and device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()

        model.load(model_dir, device)

        training_config = {
            "resolution": resolution, "batch_size": batch_size,
            "learning_rate": learning_rate, "use_amp": use_amp,
            "architecture": architecture, "face_type": face_type,
            "ae_dims": ae_dims, "e_dims": e_dims, "d_dims": d_dims,
            "random_warp": random_warp, "gan_power": gan_power,
            "random_hsv_power": random_hsv_power, "src_src_mode": src_src_mode,
        }
        self._save_training_config(model_type, model_dir, training_config)

        criterion = torch.nn.L1Loss()
        iteration = 0
        import time
        start_time = time.time()
        last_save = start_time

        lock_file = model_dir / ".training_lock"
        lock_file.touch()

        try:
            _logger.info(f"Training {model_type.value} started. SRC: {len(src_dataset)}, DST: {len(dst_dataset)}")

            while True:
                for src_batch, dst_batch in zip(src_loader, dst_loader):
                    src_img = src_batch["image"].to(device)
                    dst_img = dst_batch["image"].to(device)

                    optimizer.zero_grad(set_to_none=True)

                    if use_amp and scaler is not None:
                        with torch.cuda.amp.autocast():
                            enc_src = encoder(src_img)
                            enc_dst = encoder(dst_img)

                            if inter is not None:
                                latent_src = inter(enc_src)
                                latent_dst = inter(enc_dst)
                                pred_src = decoder_src(enc_src, latent_src)
                                pred_dst = decoder_dst(enc_dst, latent_dst)
                            else:
                                pred_src = decoder_src(enc_src)
                                pred_dst = decoder_dst(enc_dst)

                            loss_src = criterion(pred_src, src_img)
                            loss_dst = criterion(pred_dst, dst_img)
                            loss = loss_src + loss_dst

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        enc_src = encoder(src_img)
                        enc_dst = encoder(dst_img)

                        if inter is not None:
                            latent_src = inter(enc_src)
                            latent_dst = inter(enc_dst)
                            pred_src = decoder_src(enc_src, latent_src)
                            pred_dst = decoder_dst(enc_dst, latent_dst)
                        else:
                            pred_src = decoder_src(enc_src)
                            pred_dst = decoder_dst(enc_dst)

                        loss_src = criterion(pred_src, src_img)
                        loss_dst = criterion(pred_dst, dst_img)
                        loss = loss_src + loss_dst
                        loss.backward()
                        optimizer.step()

                    iteration += 1
                    if iteration % 10 == 0:
                        elapsed = (time.time() - start_time) / 60.0
                        _logger.info(
                            f"[{model_type.value}] Iter: {iteration} | "
                            f"Loss SRC: {loss_src.item():.6f} | Loss DST: {loss_dst.item():.6f} | "
                            f"Elapsed: {elapsed:.1f} min"
                        )

                    now = time.time()
                    if (now - last_save) / 60.0 >= save_interval_min:
                        model._encoder = encoder
                        model._decoder_src = decoder_src
                        model._decoder_dst = decoder_dst
                        model._inter = inter
                        model.save(model_dir)

                        import io
                        opt_buf = io.BytesIO()
                        torch.save(optimizer.state_dict(), opt_buf)
                        FileManager.atomic_write(model_dir / f"{model_type.value}_optimizer.pt", opt_buf.getvalue())

                        last_save = now

        except KeyboardInterrupt:
            _logger.info("Training interrupted by user.")
        finally:
            if lock_file.exists():
                lock_file.unlink()
            model._encoder = encoder
            model._decoder_src = decoder_src
            model._decoder_dst = decoder_dst
            model._inter = inter
            model.save(model_dir)
            _logger.info(f"Model saved at iteration {iteration}")

    def export_dfm(self, model_type: ModelType, model_dir: Path) -> Path:
        _logger.info(f"DFM export for {model_type.value} - placeholder implementation")
        model_dir = Path(model_dir)
        output_path = model_dir / f"{model_type.value}.dfm"
        return output_path
