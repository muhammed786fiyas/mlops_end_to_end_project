"""
C3 + C6 — Unit tests for FeatureService.

These tests instantiate FeatureService directly against the project's
real SQLite database (read-only). They are unit tests in the sense that
they don't go through HTTP, but they use real data because mocking
SQLite would add complexity without much value.
"""
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from src.api.feature_service import FeatureService
from src.training.train import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def feature_service():
    """Construct FeatureService against the real project database."""
    with open(PROJECT_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    db_path = PROJECT_ROOT / config["data"]["raw_db"]
    team_lookup_path = PROJECT_ROOT / config["data"]["processed_dir"] / "team_lookup.csv"

    if not db_path.exists() or not team_lookup_path.exists():
        pytest.skip(
            f"Project data not found ({db_path} or {team_lookup_path}). "
            f"Run `dvc pull` then `python -m src.features.build_features`."
        )

    return FeatureService(db_path=db_path, team_lookup_path=team_lookup_path)


def test_cold_start_team_returns_zero_form(feature_service, caplog):
    """TC-3.1 — Team with insufficient match history returns zeroed form features.

    We pick a far-past date (1990) so no team in the dataset has 5 prior matches.
    Real Kaggle data starts in 2008.
    """
    cold_date = datetime(1990, 1, 1)
    # Use any valid team_id from the lookup
    team_id = int(feature_service.team_lookup["team_api_id"].iloc[0])

    form = feature_service._compute_team_form(team_id, cold_date)

    assert form == {
        "form_wins": 0,
        "form_draws": 0,
        "form_losses": 0,
        "form_gs_avg": 0.0,
        "form_gc_avg": 0.0,
    }


def test_feature_columns_match_train_contract(feature_service):
    """TC-6.1 — FeatureService and train.py declare the same feature columns in the same order.

    This is the highest-impact test in the suite. If train.py adds a feature
    and feature_service.py is not updated, the model silently receives the
    wrong feature vector at inference time — no exception, just nonsense
    predictions. One assertion prevents a whole class of silent bugs.
    """
    assert feature_service._expected_columns() == FEATURE_COLUMNS, (
        "FeatureService._expected_columns() does not match train.FEATURE_COLUMNS — "
        "the runtime feature vector will not match what the model was trained on. "
        "Update both files to keep them in lockstep."
    )
