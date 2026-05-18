import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# MoveNet keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

INPUT_SIZE = 256
MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"


class PoseService:
    """Detects body pose keypoints using MoveNet Thunder (TensorFlow Hub)."""

    def __init__(self):
        self.model = None

    def load(self):
        """Load the MoveNet model from TF Hub. Called once at app startup."""
        self.model = hub.load(MODEL_URL).signatures["serving_default"]

    def _run_inference(self, frame: np.ndarray) -> np.ndarray:
        """
        Run MoveNet on a single RGB frame.

        Args:
            frame: RGB numpy array (H, W, 3)

        Returns:
            keypoints: numpy array of shape (17, 3) — [y, x, confidence]
        """
        img = tf.image.resize_with_pad(
            tf.expand_dims(frame, axis=0), INPUT_SIZE, INPUT_SIZE
        )
        img = tf.cast(img, dtype=tf.int32)
        outputs = self.model(img)
        keypoints = outputs["output_0"].numpy()[0, 0, :, :]  # (17, 3)
        return keypoints

    def detect_sequence(
        self, frames: list[np.ndarray], fps: float
    ) -> list[np.ndarray | None]:
        """
        Run pose detection on a sequence of RGB frames.

        Args:
            frames: list of RGB numpy arrays (H, W, 3)
            fps: frames per second (unused for MoveNet, kept for API compat)

        Returns:
            List of keypoint arrays (17, 3) with [y, x, confidence],
            or None if no pose was detected in that frame.
        """
        if self.model is None:
            raise RuntimeError("PoseService not loaded. Call load() first.")

        keypoints_sequence = []

        for frame in frames:
            keypoints = self._run_inference(frame)

            # Check if enough keypoints were detected
            avg_confidence = np.mean(keypoints[:, 2])
            if avg_confidence > 0.1:
                keypoints_sequence.append(keypoints)
            else:
                keypoints_sequence.append(None)

        return keypoints_sequence

    def keypoints_to_dict_list(
        self, keypoints_sequence: list,
    ) -> list[list[dict] | None]:
        """
        Convert keypoints to serializable format for the frontend.
        Used for skeleton visualization on the client side.
        """
        result = []
        for keypoints in keypoints_sequence:
            if keypoints is None:
                result.append(None)
                continue
            frame_kps = [
                {
                    "name": KEYPOINT_NAMES[i],
                    "x": float(keypoints[i, 1]),
                    "y": float(keypoints[i, 0]),
                    "confidence": float(keypoints[i, 2]),
                }
                for i in range(17)
            ]
            result.append(frame_kps)
        return result
