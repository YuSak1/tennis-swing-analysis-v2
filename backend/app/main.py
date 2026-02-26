from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR, UPLOAD_DIR, REFERENCE_VIDEOS_DIR
from app.routers.analysis import router as analysis_router
from app.services import (
    VideoService,
    PoseService,
    FeatureService,
    ComparisonService,
    FeedbackService,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and reference data once at startup."""
    print("Starting up — loading services...")

    # Ensure upload directory exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize services
    app.state.video_service = VideoService()

    app.state.pose_service = PoseService()
    app.state.pose_service.load()
    print("  ✓ MoveNet Thunder loaded")

    app.state.feature_service = FeatureService()
    print("  ✓ Feature service ready")

    app.state.comparison_service = ComparisonService()
    app.state.comparison_service.load()
    print("  ✓ Reference data loaded")

    app.state.feedback_service = FeedbackService()
    print("  ✓ Feedback service ready")

    print("All services loaded. Ready to analyze swings!")

    yield

    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="Tennis Swing Analysis",
    description="Analyze your tennis forehand by comparing it to pro players.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — permissive for development, tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(analysis_router, prefix="/api")

# Serve reference videos
if REFERENCE_VIDEOS_DIR.exists():
    app.mount(
        "/api/references/videos",
        StaticFiles(directory=str(REFERENCE_VIDEOS_DIR)),
        name="reference_videos",
    )

# Serve React static files (must be last — catches all routes)
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
