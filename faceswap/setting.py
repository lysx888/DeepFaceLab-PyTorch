from enum import IntEnum
from pathlib import Path
from sys import platform as _sys_platform
from typing import Final

_FFMPEG_SUFFIX = ".exe" if _sys_platform == "win32" else ""

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
_WORKSPACE_ROOT: Final[Path] = _PROJECT_ROOT.parent / "workspace"
_INSIGHTFACE_ROOT: Final[Path] = _PROJECT_ROOT / "insightface"
_FFMPEG_ROOT: Final[Path] = _PROJECT_ROOT.parent / "ffmpeg"
_WEIGHTS_ROOT: Final[Path] = _PROJECT_ROOT / "weights"

WORKSPACE_DIR: Final[Path] = _WORKSPACE_ROOT
INSIGHTFACE_DIR: Final[Path] = _INSIGHTFACE_ROOT
INSIGHTFACE_MODEL_DIR: Final[Path] = _WEIGHTS_ROOT
FFMPEG_DIR: Final[Path] = _FFMPEG_ROOT
FFMPEG_PATH: Final[Path] = _FFMPEG_ROOT / f"ffmpeg{_FFMPEG_SUFFIX}"
FFPROBE_PATH: Final[Path] = _FFMPEG_ROOT / f"ffprobe{_FFMPEG_SUFFIX}"
MODEL_DIR: Final[Path] = _WORKSPACE_ROOT / "model"

DATA_SRC_DIR: Final[Path] = _WORKSPACE_ROOT / "data_src"
DATA_DST_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst"
DATA_SRC_ALIGNED_DIR: Final[Path] = _WORKSPACE_ROOT / "data_src" / "aligned"
DATA_DST_ALIGNED_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "aligned"
DATA_DST_ALIGNED_DEBUG_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "aligned_debug"
DATA_DST_SWAPPED_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "swapped"
DATA_DST_MERGED_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "merged"
DATA_DST_MERGED_MASK_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "merged_mask"
DATA_SRC_ALIGNED_TRASH_DIR: Final[Path] = _WORKSPACE_ROOT / "data_src" / "aligned_trash"
DATA_DST_ALIGNED_TRASH_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "aligned_trash"
XSEG_MODEL_DIR: Final[Path] = _WORKSPACE_ROOT / "model"

SAEHD_MODEL_DIR: Final[Path] = _WORKSPACE_ROOT / "model" / "saehd"
PRETRAIN_DATA_DIR: Final[Path] = _WORKSPACE_ROOT / "pretrain_faces"

INSIGHTFACE_TRAIN_DIR: Final[Path] = _WORKSPACE_ROOT / "insightface_train"
INSIGHTFACE_MANUAL_ANNOTATED_DIR: Final[Path] = _WORKSPACE_ROOT / "insightface_train" / "manual_annotated"
INSIGHTFACE_TO_ANNOTATE_DIR: Final[Path] = _WORKSPACE_ROOT / "insightface_train" / "to_annotate"

DATA_SRC_VIDEO_PATTERN: Final[str] = "data_src.*"
DATA_DST_VIDEO_PATTERN: Final[str] = "data_dst.*"

INSIGHTFACE_MODEL_PACKAGE: Final[str] = "antelopev2"

DEFAULT_FACE_OUTPUT_SIZE: Final[int] = 512
DEFAULT_JPG_QUALITY: Final[int] = 100
DEFAULT_DET_THRESH: Final[float] = 0.5
DEFAULT_BATCH_SIZE: Final[int] = 4

LANDMARK_POINTS: Final[int] = 106
KPS5_POINTS: Final[int] = 5

VGG19_MODEL_PATH: Final[Path] = _WEIGHTS_ROOT / "vgg19-dcbb9e9d.pth"
FACE_PARSING_MODEL_PATH: Final[Path] = _WEIGHTS_ROOT / "face-parsing" / "model.onnx"
FACE_OCCLUDER_MODEL_DIR: Final[Path] = _WEIGHTS_ROOT / "face-occluder"
YOLO_MODEL_DIR: Final[Path] = _WEIGHTS_ROOT / "yolo"
YOLO_DEFAULT_MODEL: Final[str] = "yolo26n-seg.pt"
NSFW_MODEL_DIR: Final[Path] = _WEIGHTS_ROOT

SAM2_MODEL_DIR: Final[Path] = _WEIGHTS_ROOT / "sam2"
SAM2_CHECKPOINT_PATH: Final[Path] = SAM2_MODEL_DIR / "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG_PATH: Final[Path] = SAM2_MODEL_DIR / "sam2.1_hiera_b+.yaml"

SAM3_MODEL_DIR: Final[Path] = _WEIGHTS_ROOT / "sam3"
SAM3_CHECKPOINT_PATH: Final[Path] = SAM3_MODEL_DIR / "sam3.1_multiplex.pt"
SAM3_CONFIG_PATH: Final[Path] = SAM3_MODEL_DIR / "config.json"


class FaceType(IntEnum):
    HALF = 0
    MID_FULL = 1
    FULL = 2
    WHOLE_FACE = 3
    HEAD = 4


FACE_TYPE_SCALE: Final[dict[FaceType, float]] = {
    FaceType.HALF: 1.0,
    FaceType.MID_FULL: 1.0,
    FaceType.FULL: 1.15,
    FaceType.WHOLE_FACE: 1.35,
    FaceType.HEAD: 2.0,
}

FACE_TYPE_CHIN_OFFSET: Final[dict[FaceType, float]] = {
    FaceType.HALF: 0.0,
    FaceType.MID_FULL: 0.0,
    FaceType.FULL: 0.0,
    FaceType.WHOLE_FACE: 0.05,
    FaceType.HEAD: 0.35,
}

FACE_TYPE_MIN_RESOLUTION: Final[dict[FaceType, int]] = {
    FaceType.HALF: 64,
    FaceType.MID_FULL: 64,
    FaceType.FULL: 96,
    FaceType.WHOLE_FACE: 128,
    FaceType.HEAD: 192,
}

SUPPORTED_IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SUPPORTED_VIDEO_EXTENSIONS: Final[tuple[str, ...]] = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm")

IMAGE_EXTENSIONS_SET: Final[set[str]] = set(SUPPORTED_IMAGE_EXTENSIONS)
VIDEO_EXTENSIONS_SET: Final[set[str]] = set(SUPPORTED_VIDEO_EXTENSIONS)
