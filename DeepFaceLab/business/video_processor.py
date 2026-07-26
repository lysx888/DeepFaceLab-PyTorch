import subprocess
import re
from pathlib import Path
from typing import Optional

from DeepFaceLab.setting import FFMPEG_PATH, FFPROBE_PATH
from DeepFaceLab.shared.logger import get_logger

_logger = get_logger("video_processor")


class VideoProcessor:
    def __init__(self, ffmpeg_path: Optional[str] = None) -> None:
        self._ffmpeg = Path(ffmpeg_path) if ffmpeg_path else FFMPEG_PATH
        self._ffprobe = self._ffmpeg.parent / "ffprobe.exe"
        if not self._ffmpeg.exists():
            alt = Path("ffmpeg")
            try:
                subprocess.run([str(alt), "-version"], capture_output=True, check=True)
                self._ffmpeg = alt
                self._ffprobe = Path("ffprobe")
            except (FileNotFoundError, subprocess.CalledProcessError):
                raise FileNotFoundError(
                    f"ffmpeg not found at {self._ffmpeg}. "
                    f"Please install ffmpeg and set the correct path."
                )

    def _run_ffmpeg(self, args: list[str], silent: bool = False,
                     stream_callback=None) -> subprocess.CompletedProcess:
        cmd = [str(self._ffmpeg)] + args
        _logger.info(f"Running: {' '.join(cmd)}")
        if stream_callback is not None:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace',
                                    bufsize=1)
            for line in proc.stdout:
                line = line.rstrip('\n')
                if '\r' in line:
                    segments = line.split('\r')
                    last = segments[-1].strip()
                    if last:
                        stream_callback(last, overwrite=True)
                else:
                    if line.strip():
                        stream_callback(line, overwrite=False)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed (code {proc.returncode})")
            return subprocess.CompletedProcess(cmd, proc.returncode)
        if silent:
            result = subprocess.run(cmd, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (code {result.returncode}): {result.stderr}")
        return result

    def _run_ffprobe(self, args: list[str]) -> subprocess.CompletedProcess:
        cmd = [str(self._ffprobe)] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        return result

    def get_video_info(self, video_path: Path) -> dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        result = self._run_ffprobe([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of", "json",
            str(video_path),
        ])
        import json
        info = json.loads(result.stdout)
        return info.get("streams", [{}])[0] if info.get("streams") else {}

    def extract_frames_src(
        self,
        video_path: Path,
        output_dir: Path,
        fps: Optional[float] = None,
        output_format: str = "jpg",
        stream_callback=None,
    ) -> int:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        args = ["-i", str(video_path)]
        if fps is not None and fps > 0:
            args += ["-r", str(fps)]
        ext = "jpg" if output_format in ("jpg", "jpeg") else "png"
        args += ["-q:v", "2"] if ext == "jpg" else ["-compression_level", "3"]
        args += [str(output_dir / f"%05d.{ext}")]

        self._run_ffmpeg(args, stream_callback=stream_callback)
        count = len(list(output_dir.glob(f"*.{ext}")))
        _logger.info(f"Extracted {count} frames from {video_path.name} to {output_dir}")
        return count

    def extract_frames_dst(
        self,
        video_path: Path,
        output_dir: Path,
        fps: Optional[float] = None,
        stream_callback=None,
    ) -> int:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        args = ["-i", str(video_path)]
        if fps:
            args += ["-r", str(fps)]
        args += [
            "-compression_level", "3",
            str(output_dir / "%05d.png"),
        ]
        self._run_ffmpeg(args, stream_callback=stream_callback)
        count = len(list(output_dir.glob("*.png")))
        _logger.info(f"Extracted {count} frames from {video_path.name} to {output_dir}")
        return count

    def cut_video(
        self,
        video_path: Path,
        output_path: Path,
        start_time: str,
        end_time: str,
        stream_callback=None,
    ) -> None:
        video_path = Path(video_path)
        output_path = Path(output_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "-i", str(video_path),
            "-ss", start_time,
            "-to", end_time,
            "-c", "copy",
            str(output_path),
        ]
        self._run_ffmpeg(args, stream_callback=stream_callback)
        _logger.info(f"Cut video: {video_path.name} -> {output_path.name}")

    def denoise_frames(self, frames_dir: Path) -> None:
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        import cv2
        import numpy as np
        from DeepFaceLab.shared.file_manager import FileManager

        images = FileManager.find_images(frames_dir)
        if not images:
            _logger.warning(f"No images found in {frames_dir}")
            return

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            cv2.imwrite(str(img_path), denoised)

        _logger.info(f"Denoised {len(images)} frames in {frames_dir}")

    def merge_frames_to_video(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float = 0.0,
        audio_source: Optional[Path] = None,
        lossless: bool = False,
    ) -> Path:
        frames_dir = Path(frames_dir)
        output_path = Path(output_path)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        from DeepFaceLab.shared.file_manager import FileManager
        images = FileManager.find_images(frames_dir)
        if not images:
            raise ValueError(f"No frames found in {frames_dir}")

        if fps <= 0:
            fps = 25.0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            "-framerate", str(fps),
            "-i", str(frames_dir / "%05d.png"),
        ]

        if lossless:
            args += ["-c:v", "huffyuv"]
        else:
            ext = output_path.suffix.lower()
            if ext == ".avi":
                args += ["-c:v", "h264", "-crf", "18"]
            elif ext == ".mov":
                args += ["-c:v", "prores_ks", "-profile:v", "3"]
            else:
                args += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]

        if audio_source is not None and audio_source.exists():
            args += ["-i", str(audio_source), "-map", "0:v", "-map", "1:a?", "-c:a", "aac"]

        args += [str(output_path)]
        self._run_ffmpeg(args)
        _logger.info(f"Merged frames to video: {output_path}")
        return output_path

    def get_fps(self, video_path: Path) -> float:
        info = self.get_video_info(video_path)
        r_frame_rate = info.get("r_frame_rate", "25/1")
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            return float(num) / float(den) if float(den) != 0 else 25.0
        return float(r_frame_rate)
