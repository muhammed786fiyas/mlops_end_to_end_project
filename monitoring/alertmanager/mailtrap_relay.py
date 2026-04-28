"""
MailTrap relay for AlertManager webhooks.

Receives webhook POSTs from AlertManager (which doesn't natively support
MailTrap's REST API) and forwards them as emails via MailTrap's HTTP API.

This is a sidecar pattern: AlertManager fires a webhook, this Flask app
translates the JSON payload into a MailTrap email send.

Env vars required:
- MAILTRAP_API_TOKEN: MailTrap REST API token
- MAILTRAP_INBOX_ID: numeric ID of the sandbox inbox
- MAILTRAP_TO_EMAIL: where alerts get delivered (any address — MailTrap captures all)
- MAILTRAP_FROM_EMAIL: sender address
- MAILTRAP_FROM_NAME: sender display name

Adapted from Assignment 5; production-grade additions: env-var-driven
inbox ID + recipient, structured logging, request validation, /health
endpoint for Docker healthchecks.
"""
import logging
import os
import sys

import requests
from flask import Flask, jsonify, request

# Logging — single-line format so it interleaves cleanly with other services in `docker compose logs -f`
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | mailtrap_relay | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# Required env vars — fail loudly at startup if any are missing
def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required env var {name} is not set")
    return value


MAILTRAP_API_TOKEN = _require_env("MAILTRAP_API_TOKEN")
MAILTRAP_INBOX_ID = _require_env("MAILTRAP_INBOX_ID")
TO_EMAIL = _require_env("MAILTRAP_TO_EMAIL")
FROM_EMAIL = os.getenv("MAILTRAP_FROM_EMAIL", "alerts@football-mlops.example.com")
FROM_NAME = os.getenv("MAILTRAP_FROM_NAME", "Football MLOps AlertManager")

MAILTRAP_API_URL = f"https://sandbox.api.mailtrap.io/api/send/{MAILTRAP_INBOX_ID}"

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Healthcheck endpoint for Docker."""
    return jsonify({"status": "ok"}), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    """Receive AlertManager webhook payload and relay each alert via MailTrap REST API."""
    data = request.get_json(force=True, silent=True)
    if not data:
        logger.error("Received webhook with no JSON body")
        return jsonify({"error": "no JSON body"}), 400

    alerts = data.get("alerts", [])
    if not alerts:
        logger.warning("Webhook received with empty alerts list")
        return jsonify({"status": "ok", "alerts_processed": 0}), 200

    successes = 0
    failures = 0

    for alert in alerts:
        alertname = alert.get("labels", {}).get("alertname", "Unknown")
        status = alert.get("status", "unknown").upper()
        severity = alert.get("labels", {}).get("severity", "unknown")
        summary = alert.get("annotations", {}).get("summary", "No summary")
        description = alert.get("annotations", {}).get("description", "")

        try:
            response = requests.post(
                MAILTRAP_API_URL,
                headers={
                    "Authorization": f"Bearer {MAILTRAP_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": {"email": FROM_EMAIL, "name": FROM_NAME},
                    "to": [{"email": TO_EMAIL}],
                    "subject": f"[{status}] Football MLOps: {alertname}",
                    "html": f"""
                        <h2>{alertname}</h2>
                        <p><b>Status:</b> {status}</p>
                        <p><b>Severity:</b> {severity}</p>
                        <p><b>Summary:</b> {summary}</p>
                        <p><b>Description:</b> {description}</p>
                        <hr>
                        <p style="color:#888;font-size:12px;">
                            Football MLOps AlertManager → MailTrap relay sandbox
                        </p>
                    """,
                },
                timeout=10,
            )
            if response.status_code in (200, 201, 202):
                logger.info(f"Sent alert: {alertname} ({status}, {severity})")
                successes += 1
            else:
                logger.error(
                    f"MailTrap API rejected alert {alertname}: "
                    f"HTTP {response.status_code} — {response.text[:200]}"
                )
                failures += 1
        except requests.RequestException as e:
            logger.error(f"Failed to relay alert {alertname}: {e}")
            failures += 1

    return jsonify({
        "status": "ok",
        "alerts_processed": len(alerts),
        "succeeded": successes,
        "failed": failures,
    }), 200


if __name__ == "__main__":
    # For local dev. In Docker we use gunicorn (see Dockerfile.relay).
    app.run(host="0.0.0.0", port=5001, debug=False)
