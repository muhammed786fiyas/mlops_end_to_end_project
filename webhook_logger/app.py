"""
Webhook receiver that prints incoming AlertManager payloads to stdout.

In a production setup this would be replaced by a Slack webhook URL or
PagerDuty events API endpoint. The contract is identical: AlertManager
POSTs a JSON envelope describing the firing/resolved alerts.

This stub deliberately writes to stdout so the alerts show up in
`docker compose logs webhook-logger`.
"""
import json
import sys
import time
from fastapi import FastAPI, Request

app = FastAPI(title="Webhook Logger Stub")


def _log(channel: str, payload: dict) -> None:
    """Pretty-print one alert envelope, flush immediately so logs show up live."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    n_alerts = len(payload.get("alerts", []))
    status = payload.get("status", "?")
    receiver = payload.get("receiver", "?")

    print("=" * 72, flush=True)
    print(f"[{timestamp}] {channel.upper()} | status={status} | alerts={n_alerts} | receiver={receiver}", flush=True)
    print("=" * 72, flush=True)

    for a in payload.get("alerts", []):
        labels = a.get("labels", {})
        annotations = a.get("annotations", {})
        print(f"  alert     : {labels.get('alertname', '?')}", flush=True)
        print(f"  severity  : {labels.get('severity', '?')}", flush=True)
        print(f"  status    : {a.get('status', '?')}", flush=True)
        print(f"  starts_at : {a.get('startsAt', '?')}", flush=True)
        print(f"  summary   : {annotations.get('summary', '')}", flush=True)
        print(f"  desc      : {annotations.get('description', '')}", flush=True)
        print("-" * 72, flush=True)
    sys.stdout.flush()


@app.post("/alerts")
async def receive_default(request: Request):
    payload = await request.json()
    _log("default", payload)
    return {"received": True}


@app.post("/critical")
async def receive_critical(request: Request):
    payload = await request.json()
    _log("critical", payload)
    return {"received": True}


@app.get("/health")
def health():
    return {"status": "ok"}
