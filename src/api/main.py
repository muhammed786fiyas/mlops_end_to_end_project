"""
FastAPI backend for the football match outcome prediction service.

Loads the trained LightGBM model and the FeatureService at startup, then 
exposes three endpoints: /health, /teams, /predict.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for the interactive Swagger UI.
"""

import json
import os
import socket
import time
from contextlib import asynccontextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import joblib
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.feature_service import FeatureService
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError("Could not find project root")


PROJECT_ROOT = _project_root()
with open(PROJECT_ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_baseline.pkl"
DB_PATH = PROJECT_ROOT / CONFIG['data']['raw_db']
TEAM_LOOKUP_PATH = PROJECT_ROOT / CONFIG['data']['processed_dir'] / "team_lookup.csv"
METADATA_PATH = PROJECT_ROOT / "models" / "training_metadata.json"

CONTAINER_ID = socket.gethostname()  # A2 pattern — proves load balancing in viva

# Class label mapping (must match training)
INT_TO_OUTCOME = {0: 'H', 1: 'D', 2: 'A'}
OUTCOME_TO_LABEL = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}


# ---------------------------------------------------------------------------
# Application state — populated at startup
# ---------------------------------------------------------------------------

class AppState:
    model = None
    feature_service: Optional[FeatureService] = None
    model_version: str = "unknown"


state = AppState()


# ---------------------------------------------------------------------------
# Lifecycle: load model + feature service at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and feature service once at startup, reuse across requests."""
    logger.info("=" * 60)
    logger.info("API STARTUP — loading model and feature service")
    logger.info("=" * 60)

    # Load model
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise RuntimeError(f"Model file missing: {MODEL_PATH}")
    state.model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")

    # Load model version from metadata
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        state.model_version = f"baseline-{metadata.get('git_commit', 'unknown')}"
    else:
        state.model_version = "baseline-unknown"
    logger.info(f"Model version: {state.model_version}")

    # Load feature service
    state.feature_service = FeatureService(
        db_path=DB_PATH,
        team_lookup_path=TEAM_LOOKUP_PATH,
    )
    logger.info(f"Container ID: {CONTAINER_ID}")
    logger.info("API ready to serve requests")

    yield  # app runs here

    # Shutdown (nothing to clean up for now)
    logger.info("API shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Football Match Outcome Prediction API",
    description="Predicts H/D/A outcome for European football matches using a LightGBM classifier.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the frontend (on a different port) to call this API
# In production you'd restrict origins; for dev we allow all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models for request/response validation
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    container_id: str
    timestamp: str


class TeamInfo(BaseModel):
    team_api_id: int
    name: str
    short_name: Optional[str] = None


class TeamsResponse(BaseModel):
    count: int
    container_id: str
    teams: list[TeamInfo]


class PredictRequest(BaseModel):
    home_team_id: int = Field(..., description="team_api_id of home team")
    away_team_id: int = Field(..., description="team_api_id of away team")
    match_date: Optional[date] = Field(None, description="Match date (YYYY-MM-DD). Defaults to today.")


class PredictResponse(BaseModel):
    prediction: str
    prediction_label: str
    confidence: float
    probabilities: dict[str, float]
    home_team: str
    away_team: str
    match_date: str
    container_id: str
    model_version: str
    inference_latency_ms: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness check. Returns model-loaded status and container ID."""
    return HealthResponse(
        status="ok" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        container_id=CONTAINER_ID,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/teams", response_model=TeamsResponse, tags=["data"])
def list_teams():
    """Return all teams available for prediction. Used by the frontend dropdown."""
    if state.feature_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    teams = state.feature_service.list_teams()
    return TeamsResponse(
        count=len(teams),
        container_id=CONTAINER_ID,
        teams=[TeamInfo(**t) for t in teams],
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    """Predict the outcome for a match between two teams."""
    if state.model is None or state.feature_service is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded. Try again in a moment.")

    # Validate team IDs
    if not state.feature_service.team_exists(request.home_team_id):
        raise HTTPException(status_code=400, detail=f"Unknown home_team_id: {request.home_team_id}")
    if not state.feature_service.team_exists(request.away_team_id):
        raise HTTPException(status_code=400, detail=f"Unknown away_team_id: {request.away_team_id}")
    if request.home_team_id == request.away_team_id:
        raise HTTPException(status_code=400, detail="home_team_id and away_team_id must differ")

    # Default date to today
    match_date = request.match_date or date.today()
    match_datetime = datetime.combine(match_date, datetime.min.time())

    # Build features + predict
    start = time.perf_counter()
    try:
        features = state.feature_service.build_features(
            home_team_id=request.home_team_id,
            away_team_id=request.away_team_id,
            match_date=match_datetime,
        )
        proba = state.model.predict_proba(features)[0]
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    latency_ms = int((time.perf_counter() - start) * 1000)

    # Decode prediction
    predicted_class = int(proba.argmax())
    predicted_outcome = INT_TO_OUTCOME[predicted_class]
    probabilities = {INT_TO_OUTCOME[i]: float(proba[i]) for i in range(3)}

    home_name = state.feature_service.get_team_name(request.home_team_id)
    away_name = state.feature_service.get_team_name(request.away_team_id)

    logger.info(
        f"Prediction: {home_name} vs {away_name} → {predicted_outcome} "
        f"(confidence={proba[predicted_class]:.3f}, latency={latency_ms}ms)"
    )

    return PredictResponse(
        prediction=predicted_outcome,
        prediction_label=OUTCOME_TO_LABEL[predicted_outcome],
        confidence=float(proba[predicted_class]),
        probabilities=probabilities,
        home_team=home_name,
        away_team=away_name,
        match_date=str(match_date),
        container_id=CONTAINER_ID,
        model_version=state.model_version,
        inference_latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Entrypoint for `python -m src.api.main` (alternative to uvicorn CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)