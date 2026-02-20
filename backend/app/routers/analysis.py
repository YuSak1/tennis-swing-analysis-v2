import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request

from app.config import UPLOAD_DIR
from app.models.schemas import (
    AnalysisResponse,
    PlayerSimilarity,
    CoachingTip,
    SwingPhases,
    HealthResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", version="2.0.0")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_swing(
    request: Request,
    video: UploadFile = File(...),
    hand: str = Form("right"),
):
    """
    Analyze a tennis forehand swing.

    - Upload an MP4 video of a forehand swing
    - Select dominant hand (right/left)
    - Returns similarity comparison to pro players with coaching feedback
    """
    # Get services from app state
    video_service = request.app.state.video_service
    pose_service = request.app.state.pose_service
    feature_service = request.app.state.feature_service
    comparison_service = request.app.state.comparison_service
    feedback_service = request.app.state.feedback_service

    # Create a unique temp directory for this request
    request_id = str(uuid.uuid4())
    request_dir = UPLOAD_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Save uploaded file
        file_path = request_dir / "video.mp4"
        with open(file_path, "wb") as f:
            content = await video.read()
            f.write(content)

        # 2. Validate video
        is_valid, message = video_service.validate(file_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)

        # 3. Extract frames
        is_lefty = hand.lower() in ("left", "left_handed")
        frames, fps = video_service.extract_frames(file_path, is_lefty=is_lefty)

        if len(frames) < 10:
            raise HTTPException(
                status_code=400,
                detail="Could not extract enough frames from the video.",
            )

        # 4. Run pose detection
        landmarks_sequence = pose_service.detect_sequence(frames, fps)

        # Check we got enough valid detections
        valid_count = sum(1 for lm in landmarks_sequence if lm is not None)
        if valid_count < len(landmarks_sequence) * 0.5:
            raise HTTPException(
                status_code=400,
                detail="Could not detect body pose in enough frames. "
                "Make sure the full body is visible in the video.",
            )

        # 5. Extract biomechanical features
        features_df = feature_service.extract_sequence_features(landmarks_sequence)

        # 6. Detect swing phases
        phases = feature_service.detect_swing_phases(features_df)

        # 7. Compare to pro players via DTW
        comparison_results = comparison_service.compare(features_df)

        # 8. Generate coaching feedback
        coaching_tips = feedback_service.generate(comparison_results, features_df)

        # 9. Build response
        most_similar = max(
            comparison_results.items(),
            key=lambda x: x[1].get("overall_similarity", 0),
        )[0]

        similarities = [
            PlayerSimilarity(
                player=player,
                overall_similarity=round(data["overall_similarity"], 1),
                body_groups={
                    group: round(data["body_groups"][group]["similarity"], 1)
                    for group in data["body_groups"]
                },
            )
            for player, data in comparison_results.items()
        ]
        # Sort by similarity descending
        similarities.sort(key=lambda s: s.overall_similarity, reverse=True)

        coaching = [
            CoachingTip(
                type=tip["type"],
                body_part=tip["body_part"],
                message=tip["message"],
            )
            for tip in coaching_tips
        ]

        # Convert landmarks for frontend visualization
        landmarks_dict = pose_service.landmarks_to_dict_list(landmarks_sequence)

        return AnalysisResponse(
            most_similar_player=most_similar,
            similarities=similarities,
            coaching=coaching,
            phases=SwingPhases(**phases),
            landmarks=landmarks_dict,
            reference_video_url=f"/api/references/videos/{most_similar}",
        )

    finally:
        # Clean up temp files
        shutil.rmtree(request_dir, ignore_errors=True)
