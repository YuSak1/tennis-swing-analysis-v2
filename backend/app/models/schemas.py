from pydantic import BaseModel


class PlayerSimilarity(BaseModel):
    player: str
    overall_similarity: float
    body_groups: dict[str, float]


class CoachingTip(BaseModel):
    type: str  # "strength" or "improvement"
    body_part: str
    message: str


class SwingPhases(BaseModel):
    preparation: tuple[int, int]
    forward_swing: tuple[int, int]
    contact: int
    follow_through: tuple[int, int]
    recovery: tuple[int, int]


class AnalysisResponse(BaseModel):
    most_similar_player: str
    similarities: list[PlayerSimilarity]
    coaching: list[CoachingTip]
    phases: SwingPhases
    # Landmark data for frontend skeleton visualization
    landmarks: list[list[dict]] | None = None
    reference_video_url: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
