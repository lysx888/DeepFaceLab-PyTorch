from enum import Enum
from pathlib import Path
from typing import Optional

from faceswap.business.video_processor import VideoProcessor
from faceswap.shared.file_manager import FileManager
from faceswap.shared.logger import get_logger

_logger = get_logger("video_output")


class OutputFormat(Enum):
    AVI = "avi"
    MP4 = "mp4"
    MOV = "mov"


class VideoOutput:
    def __init__(self, video_processor: VideoProcessor) -> None:
        self._vp = video_processor

    def merged_to_video(
        self,
        merged_dir: Path,
        output_path: Path,
        format: OutputFormat = OutputFormat.MP4,
        reference_video: Optional[Path] = None,
        include_audio: bool = True,
        lossless: bool = False,
        override_fps: Optional[float] = None,
        stream_callback=None,
    ) -> Path:
        merged_dir = Path(merged_dir)
        output_path = Path(output_path)

        images = FileManager.find_images(merged_dir)
        if not images:
            raise ValueError(f"No merged frames found in {merged_dir}")

        fps = override_fps if override_fps is not None and override_fps > 0 else None
        audio_source = None

        if fps is None and reference_video is not None and reference_video.exists():
            try:
                fps = self._vp.get_fps(reference_video)
                if include_audio:
                    audio_source = reference_video
            except Exception as e:
                _logger.warning(f"Could not get video info from reference: {e}")

        if fps is None:
            fps = 25.0
            _logger.info(f"Could not detect fps, using default {fps}")

        if not output_path.suffix:
            output_path = output_path.with_suffix(f".{format.value}")

        result = self._vp.merge_frames_to_video(
            frames_dir=merged_dir,
            output_path=output_path,
            fps=fps,
            audio_source=audio_source,
            lossless=lossless,
            stream_callback=stream_callback,
        )

        merged_mask_dir = merged_dir.parent / "merged_mask"
        if merged_mask_dir.exists():
            mask_images = FileManager.find_images(merged_mask_dir)
            if mask_images:
                mask_output = output_path.with_stem(output_path.stem + "_mask")
                try:
                    self._vp.merge_frames_to_video(
                        frames_dir=merged_mask_dir,
                        output_path=mask_output,
                        fps=fps,
                        lossless=lossless,
                    )
                    _logger.info(f"Mask video saved to {mask_output}")
                except Exception as e:
                    _logger.warning(f"Failed to create mask video: {e}")

        if audio_source is None and reference_video is not None:
            _logger.info("No audio track found in reference video.")

        _logger.info(f"Output video saved to {result}")
        return result
