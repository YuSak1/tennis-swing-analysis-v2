import cv2
import numpy as np
from pathlib import Path

from app.config import (
    ALLOWED_EXTENSIONS,
    MIN_VIDEO_DURATION,
    MAX_VIDEO_DURATION,
    ANALYSIS_DURATION,
)


class VideoService:
    """Handles video validation and frame extraction. No GIF conversion."""

    def validate(self, file_path: Path) -> tuple[bool, str]:
        """Validate video file format and duration."""
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return False, f"File format must be one of: {', '.join(ALLOWED_EXTENSIONS)}"

        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return False, "Could not open video file."

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps <= 0 or total_frames <= 0:
            return False, "Invalid video file."

        duration = total_frames / fps

        if duration < MIN_VIDEO_DURATION:
            return False, f"Video is too short (minimum {MIN_VIDEO_DURATION:.0f} seconds)."
        if duration > MAX_VIDEO_DURATION:
            return False, f"Video is too long (maximum {MAX_VIDEO_DURATION / 60:.0f} minutes)."

        return True, "OK"

    def extract_frames(
        self,
        file_path: Path,
        is_lefty: bool = False,
        max_seconds: float = ANALYSIS_DURATION,
    ) -> tuple[list[np.ndarray], float]:
        """
        Extract RGB frames from the first N seconds of a video.

        Returns:
            frames: list of RGB numpy arrays
            fps: frames per second of the video
        """
        cap = cv2.VideoCapture(str(file_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        max_frames = int(fps * max_seconds)
        frames = []

        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if is_lefty:
                frame = cv2.flip(frame, 1)  # horizontal mirror
            # Convert BGR (OpenCV default) to RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames, fps
