from enum import IntEnum
from pathlib import Path
from typing import Final

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
_WORKSPACE_ROOT: Final[Path] = _PROJECT_ROOT.parent / "workspace"
_INSIGHTFACE_ROOT: Final[Path] = _PROJECT_ROOT / "insightface"
_FFMPEG_ROOT: Final[Path] = _PROJECT_ROOT.parent / "ffmpeg"
_GUI_MODELS_ROOT: Final[Path] = _PROJECT_ROOT / "gui_app" / "models"

WORKSPACE_DIR: Final[Path] = _WORKSPACE_ROOT
INSIGHTFACE_DIR: Final[Path] = _INSIGHTFACE_ROOT
INSIGHTFACE_MODEL_DIR: Final[Path] = _INSIGHTFACE_ROOT / "models"
FFMPEG_DIR: Final[Path] = _FFMPEG_ROOT
FFMPEG_PATH: Final[Path] = _FFMPEG_ROOT / "ffmpeg.exe"
FFPROBE_PATH: Final[Path] = _FFMPEG_ROOT / "ffprobe.exe"
MODEL_DIR: Final[Path] = _WORKSPACE_ROOT / "model"

DATA_SRC_DIR: Final[Path] = _WORKSPACE_ROOT / "data_src"
DATA_DST_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst"
DATA_SRC_ALIGNED_DIR: Final[Path] = _WORKSPACE_ROOT / "data_src" / "aligned"
DATA_DST_ALIGNED_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "aligned"
DATA_DST_ALIGNED_DEBUG_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "aligned_debug"
DATA_DST_MERGED_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "merged"
DATA_DST_MERGED_MASK_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "merged_mask"
DATA_SRC_ALIGNED_TRASH_DIR: Final[Path] = _WORKSPACE_ROOT / "data_src" / "aligned_trash"
DATA_DST_ALIGNED_TRASH_DIR: Final[Path] = _WORKSPACE_ROOT / "data_dst" / "aligned_trash"
XSEG_MODEL_DIR: Final[Path] = _WORKSPACE_ROOT / "model"
TFM_MODEL_DIR: Final[Path] = _WORKSPACE_ROOT / "model"
VGG19_MODEL_PATH: Final[Path] = _GUI_MODELS_ROOT / "vgg19-dcbb9e9d.pth"

INSIGHTFACE_TRAIN_DIR: Final[Path] = _WORKSPACE_ROOT / "insightface_train"
INSIGHTFACE_MANUAL_ANNOTATED_DIR: Final[Path] = _WORKSPACE_ROOT / "insightface_train" / "manual_annotated"
INSIGHTFACE_TO_ANNOTATE_DIR: Final[Path] = _WORKSPACE_ROOT / "insightface_train" / "to_annotate"
INSIGHTFACE_SCRFD_DIR: Final[Path] = _WORKSPACE_ROOT / "insightface_train" / "scrfd"
INSIGHTFACE_SYNTHETICS_DIR: Final[Path] = _WORKSPACE_ROOT / "insightface_train" / "synthetics"
INSIGHTFACE_OUTPUT_DIR: Final[Path] = _WORKSPACE_ROOT / "insightface_train" / "output"

DATA_SRC_VIDEO_PATTERN: Final[str] = "data_src.*"
DATA_DST_VIDEO_PATTERN: Final[str] = "data_dst.*"

INSIGHTFACE_MODEL_PACKAGE: Final[str] = "antelopev2"
INSIGHTFACE_SWAP_MODEL: Final[str] = "inswapper_128.onnx"

DEFAULT_FACE_OUTPUT_SIZE: Final[int] = 512
DEFAULT_JPG_QUALITY: Final[int] = 100
DEFAULT_DET_THRESH: Final[float] = 0.5
DEFAULT_BATCH_SIZE: Final[int] = 4
DEFAULT_LEARNING_RATE: Final[float] = 1e-4
DEFAULT_SAVE_INTERVAL_MIN: Final[float] = 15.0
DEFAULT_PREVIEW_INTERVAL_MIN: Final[float] = 15.0

TFM_DEFAULT_RESOLUTION: Final[int] = 128
TFM_DEFAULT_BATCH_SIZE: Final[int] = 4
TFM_DEFAULT_LEARNING_RATE: Final[float] = 1e-4
TFM_DEFAULT_SAVE_INTERVAL_MIN: Final[float] = 15.0
TFM_DEFAULT_PREVIEW_INTERVAL_SEC: Final[float] = 60

LANDMARK_POINTS: Final[int] = 106
KPS5_POINTS: Final[int] = 5


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

SAM2_MODEL_DIR: Final[Path] = _GUI_MODELS_ROOT
SAM2_CONFIG_DIR: Final[Path] = _GUI_MODELS_ROOT
SAM2_DEFAULT_MODEL: Final[str] = "sam2.1_hiera_small.pt"
SAM2_DEFAULT_CONFIG: Final[str] = "sam2.1_hiera_s.yaml"

FACE_PARSING_MODEL_PATH: Final[Path] = _GUI_MODELS_ROOT / "model.onnx"

YOLO_MODEL_DIR: Final[Path] = _GUI_MODELS_ROOT
YOLO_DEFAULT_MODEL: Final[str] = "yolo26n-seg.pt"
