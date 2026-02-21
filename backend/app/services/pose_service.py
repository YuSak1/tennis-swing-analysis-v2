import mediapipe as mp
import numpy as np

from app.config import POSE_MODEL_PATH

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class PoseService:
    """Detects body pose landmarks using MediaPipe Pose Landmarker."""

    def __init__(self):
        self.landmarker = None

    def load(self):
        """Load the MediaPipe model. Called once at app startup."""
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def detect_sequence(
        self, frames: list[np.ndarray], fps: float
    ) -> list[list | None]:
        """
        Run pose detection on a sequence of RGB frames.

        Args:
            frames: list of RGB numpy arrays (H, W, 3)
            fps: frames per second (used to calculate timestamps)

        Returns:
            List of landmark lists (one per frame).
            Each element is a list of 33 landmarks with x, y, z, visibility,
            or None if no pose was detected in that frame.
        """
        if self.landmarker is None:
            raise RuntimeError("PoseService not loaded. Call load() first.")

        landmarks_sequence = []

        for idx, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(idx * 1000 / fps)

            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks_sequence.append(result.pose_landmarks[0])
            else:
                landmarks_sequence.append(None)

        return landmarks_sequence

    def landmarks_to_dict_list(
        self, landmarks_sequence: list
    ) -> list[list[dict]] | None:
        """
        Convert landmarks to serializable format for the frontend.
        Used for skeleton visualization on the client side.
        """
        result = []
        for landmarks in landmarks_sequence:
            if landmarks is None:
                result.append(None)
                continue
            frame_landmarks = [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                }
                for lm in landmarks
            ]
            result.append(frame_landmarks)
        return result
