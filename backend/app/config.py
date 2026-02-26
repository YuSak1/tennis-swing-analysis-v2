from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REFERENCES_DIR = DATA_DIR / "references"
REFERENCE_VIDEOS_DIR = BASE_DIR / "reference_videos"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"

# MoveNet
MOVENET_MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
MOVENET_INPUT_SIZE = 256

# Video constraints
ALLOWED_EXTENSIONS = {".mp4"}
MIN_VIDEO_DURATION = 3.0  # seconds
MAX_VIDEO_DURATION = 1800.0  # seconds
ANALYSIS_DURATION = 10.0  # only analyze first N seconds

# Players
PLAYERS = ["Federer", "Nadal", "Djokovic", "Murray"]

# Feature groups for comparison
FEATURE_GROUPS = {
    "Racket Arm": [
        "r_elbow_angle",
        "r_shoulder_angle",
        "r_wrist_height",
        "r_arm_extension",
    ],
    "Non-Racket Arm": [
        "l_elbow_angle",
        "l_shoulder_angle",
        "l_wrist_height",
        "l_arm_extension",
    ],
    "Torso & Rotation": [
        "torso_rotation",
        "body_lean",
        "forward_lean",
        "shoulder_width",
    ],
    "Lower Body": [
        "r_knee_angle",
        "l_knee_angle",
        "r_hip_angle",
        "l_hip_angle",
        "stance_width",
    ],
}
