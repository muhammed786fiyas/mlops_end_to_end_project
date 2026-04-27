"""
Shared pytest fixtures for the football-mlops test suite.

The API tests assume the Docker stack is running:
    docker compose up -d
Backend must be reachable at http://localhost:8000.
"""
import os
import sys
from pathlib import Path

import httpx
import pytest

# Make `src` importable for unit tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Base URL for the running backend
API_BASE_URL = os.environ.get("FOOTBALL_API_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def api_url() -> str:
    """Base URL of the running backend."""
    return API_BASE_URL


@pytest.fixture(scope="session")
def http_client(api_url):
    """Reusable HTTP client for the test session."""
    with httpx.Client(base_url=api_url, timeout=10.0) as client:
        # Pre-flight: backend must be up. Skip the entire suite if not.
        try:
            r = client.get("/health")
            if r.status_code != 200:
                pytest.skip(f"Backend at {api_url} is not healthy ({r.status_code})")
        except httpx.ConnectError:
            pytest.skip(f"Backend at {api_url} is unreachable. Run `docker compose up -d`.")
        yield client


@pytest.fixture(scope="session")
def real_team_ids(http_client):
    """Two valid team IDs pulled from the live /teams endpoint."""
    r = http_client.get("/teams")
    teams = r.json()["teams"]
    assert len(teams) >= 2, "Need at least 2 teams for prediction tests"
    return teams[0]["team_api_id"], teams[1]["team_api_id"]
