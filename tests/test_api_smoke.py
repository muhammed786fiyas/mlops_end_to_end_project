"""
C1 — Smoke tests against the running backend.

Verifies the service started cleanly and the three primary endpoints
respond with the expected shape.
"""


def test_health_endpoint_returns_ok(http_client):
    """TC-1.1 — Health endpoint returns 200 with model loaded."""
    r = http_client.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["container_id"]  # non-empty string


def test_teams_endpoint_returns_full_lookup(http_client):
    """TC-1.2 — Teams endpoint returns the expected number of teams."""
    r = http_client.get("/teams")

    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 299  # known size of team_lookup.csv
    assert len(body["teams"]) == 299

    first = body["teams"][0]
    assert "team_api_id" in first
    assert "name" in first
    assert "short_name" in first


def test_metrics_endpoint_exposes_prometheus_format(http_client):
    """TC-1.3 — /metrics exposes the custom metrics in Prometheus text format."""
    r = http_client.get("/metrics")

    assert r.status_code == 200
    text = r.text

    # All 5 custom metrics must be registered (values may be 0 if no traffic yet)
    assert "football_predictions_total" in text
    assert "football_prediction_errors_total" in text
    assert "football_prediction_latency_seconds_bucket" in text
    assert "football_model_info" in text
    assert "football_model_load_total" in text
