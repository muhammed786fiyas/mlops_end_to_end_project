"""
C2 — Input validation tests against the running backend.

Verifies bad inputs are rejected with the right HTTP status codes and
helpful error messages, not silent 500s or successful predictions on
nonsense data.
"""


def test_unknown_home_team_returns_400(http_client, real_team_ids):
    """TC-2.1 — Unknown home_team_id returns 400 with a helpful message."""
    _, valid_away = real_team_ids
    r = http_client.post(
        "/predict",
        json={"home_team_id": 99999, "away_team_id": valid_away},
    )

    assert r.status_code == 400
    assert "Unknown home_team_id" in r.json()["detail"]


def test_same_team_for_home_and_away_returns_400(http_client, real_team_ids):
    """TC-2.2 — Identical home and away team_ids returns 400."""
    home, _ = real_team_ids
    r = http_client.post(
        "/predict",
        json={"home_team_id": home, "away_team_id": home},
    )

    assert r.status_code == 400
    assert "must differ" in r.json()["detail"]


def test_missing_required_field_returns_422(http_client, real_team_ids):
    """TC-2.3 — Pydantic catches missing required field with 422."""
    _, valid_away = real_team_ids
    r = http_client.post(
        "/predict",
        json={"away_team_id": valid_away},  # missing home_team_id
    )

    assert r.status_code == 422  # FastAPI's automatic validation


def test_valid_request_returns_prediction(http_client, real_team_ids):
    """Sanity check: a well-formed request must succeed.

    Not formally in the test plan, but anchors the validation tests —
    confirms 400/422 results above aren't just because /predict is broken.
    """
    home, away = real_team_ids
    r = http_client.post(
        "/predict",
        json={"home_team_id": home, "away_team_id": away},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("H", "D", "A")
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {"H", "D", "A"}
    assert body["model_version"]
    assert body["container_id"]
