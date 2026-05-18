import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# MoveNet keypoint indices (17 keypoints)
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def _compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle (degrees) at point b, formed by points a-b-c."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


class FeatureService:
    """Extracts biomechanical features and detects swing phases."""

    def extract_frame_features(self, keypoints: np.ndarray) -> dict | None:
        """
        Extract biomechanical features from a single frame's keypoints.

        Args:
            keypoints: numpy array of shape (17, 3) — [y, x, confidence]

        Returns:
            dict of feature_name -> value, or None if keypoints are missing
        """
        if keypoints is None:
            return None

        # MoveNet outputs [y, x, confidence] — convert to [x, y] for calculations
        pts = np.zeros((17, 2))
        pts[:, 0] = keypoints[:, 1]  # x
        pts[:, 1] = keypoints[:, 0]  # y

        features = {}

        # --- Joint angles ---
        features["r_elbow_angle"] = _compute_angle(
            pts[R_SHOULDER], pts[R_ELBOW], pts[R_WRIST]
        )
        features["l_elbow_angle"] = _compute_angle(
            pts[L_SHOULDER], pts[L_ELBOW], pts[L_WRIST]
        )
        features["r_shoulder_angle"] = _compute_angle(
            pts[R_HIP], pts[R_SHOULDER], pts[R_ELBOW]
        )
        features["l_shoulder_angle"] = _compute_angle(
            pts[L_HIP], pts[L_SHOULDER], pts[L_ELBOW]
        )
        features["r_knee_angle"] = _compute_angle(
            pts[R_HIP], pts[R_KNEE], pts[R_ANKLE]
        )
        features["l_knee_angle"] = _compute_angle(
            pts[L_HIP], pts[L_KNEE], pts[L_ANKLE]
        )
        features["r_hip_angle"] = _compute_angle(
            pts[R_SHOULDER], pts[R_HIP], pts[R_KNEE]
        )
        features["l_hip_angle"] = _compute_angle(
            pts[L_SHOULDER], pts[L_HIP], pts[L_KNEE]
        )

        # Torso rotation (angle between shoulder line and hip line)
        shoulder_vec = pts[R_SHOULDER] - pts[L_SHOULDER]
        hip_vec = pts[R_HIP] - pts[L_HIP]
        features["torso_rotation"] = float(
            np.degrees(
                np.arctan2(shoulder_vec[1], shoulder_vec[0])
                - np.arctan2(hip_vec[1], hip_vec[0])
            )
        )

        # --- Distances ---
        features["stance_width"] = float(
            np.linalg.norm(pts[R_ANKLE] - pts[L_ANKLE])
        )
        features["shoulder_width"] = float(
            np.linalg.norm(pts[R_SHOULDER] - pts[L_SHOULDER])
        )
        features["r_wrist_height"] = float(
            pts[R_SHOULDER][1] - pts[R_WRIST][1]
        )
        features["l_wrist_height"] = float(
            pts[L_SHOULDER][1] - pts[L_WRIST][1]
        )
        features["wrist_separation"] = float(
            np.linalg.norm(pts[R_WRIST] - pts[L_WRIST])
        )

        # --- Body center & balance ---
        hip_center = (pts[L_HIP] + pts[R_HIP]) / 2
        shoulder_center = (pts[L_SHOULDER] + pts[R_SHOULDER]) / 2
        features["body_lean"] = float(shoulder_center[0] - hip_center[0])
        features["forward_lean"] = float(shoulder_center[1] - hip_center[1])

        # --- Arm extension ---
        features["r_arm_extension"] = float(
            np.linalg.norm(pts[R_WRIST] - pts[R_SHOULDER])
        )
        features["l_arm_extension"] = float(
            np.linalg.norm(pts[L_WRIST] - pts[L_SHOULDER])
        )

        return features

    def extract_sequence_features(self, keypoints_sequence: list) -> pd.DataFrame:
        """
        Extract features from all frames, with smoothing.

        Args:
            keypoints_sequence: list of (17,3) numpy arrays from PoseService

        Returns:
            DataFrame with position features. Frames with no detection are skipped.
        """
        frames = []
        for kp in keypoints_sequence:
            if kp is not None:
                feat = self.extract_frame_features(kp)
                if feat is not None:
                    frames.append(feat)

        df = pd.DataFrame(frames)

        # Smooth noisy signals
        for col in df.columns:
            if len(df[col].dropna()) > 11:
                try:
                    df[col] = savgol_filter(df[col].values, window_length=11, polyorder=3)
                except Exception:
                    pass

        return df

    def detect_swing_phases(self, features_df: pd.DataFrame) -> dict:
        """
        Detect phases of a forehand swing using wrist velocity.

        Returns:
            dict with frame indices for each phase.
        """
        # Compute wrist velocity from position data
        if "r_wrist_height" not in features_df.columns:
            n = len(features_df)
            return self._default_phases(n)

        wrist_vel = features_df["r_wrist_height"].diff().fillna(0).values

        # Smooth
        if len(wrist_vel) > 11:
            try:
                wrist_vel = savgol_filter(wrist_vel, window_length=11, polyorder=3)
            except Exception:
                pass

        # Contact frame: peak absolute wrist velocity
        contact_frame = int(np.argmax(np.abs(wrist_vel)))

        # Find the active swing region based on velocity threshold
        threshold = 0.3 * np.max(np.abs(wrist_vel))
        above_threshold = np.where(np.abs(wrist_vel) > threshold)[0]

        n = len(features_df)

        if len(above_threshold) > 0:
            swing_start = int(above_threshold[0])
            swing_end = int(min(above_threshold[-1], n - 1))
        else:
            swing_start = 0
            swing_end = n - 1

        # Ensure logical ordering
        contact_frame = max(swing_start, min(contact_frame, swing_end))

        return {
            "preparation": (0, swing_start),
            "forward_swing": (swing_start, contact_frame),
            "contact": contact_frame,
            "follow_through": (contact_frame, swing_end),
            "recovery": (swing_end, n - 1),
        }

    def _default_phases(self, n: int) -> dict:
        """Fallback phases when detection fails."""
        q1, q2, q3 = n // 4, n // 2, 3 * n // 4
        return {
            "preparation": (0, q1),
            "forward_swing": (q1, q2),
            "contact": q2,
            "follow_through": (q2, q3),
            "recovery": (q3, n - 1),
        }
