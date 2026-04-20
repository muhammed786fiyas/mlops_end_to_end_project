# Low-Level Design — REST API Endpoints

## Service
- **Name:** football-prediction-api
- **Port:** 8000 (configurable via env `API_PORT`)
- **Framework:** FastAPI
- **Auto-generated docs:** `/docs` (Swagger UI), `/redoc`

## Endpoint: GET /health

**Purpose:** Liveness check for Docker, monitoring, manual debugging.

**Request:** No body, no params.

**Response 200:**
| Field | Type | Description |
|---|---|---|
| status | string | "ok" or "degraded" |
| model_loaded | boolean | Whether model is in memory |
| container_id | string | Hostname of container handling request |
| timestamp | string | ISO 8601 UTC |

## Endpoint: GET /teams

**Purpose:** Provide team list for frontend dropdown.

**Request:** No body, no params.

**Response 200:**
| Field | Type | Description |
|---|---|---|
| count | int | Number of teams returned |
| container_id | string | Hostname of container |
| teams | array | List of team objects |
| teams[].team_api_id | int | Unique team ID (used in /predict) |
| teams[].name | string | Full team name (e.g., "Manchester United") |
| teams[].short_name | string | 3-letter abbreviation |

## Endpoint: POST /predict

**Purpose:** Predict match outcome for two teams.

**Request body:**
| Field | Type | Required | Description |
|---|---|---|---|
| home_team_id | int | yes | team_api_id of home team |
| away_team_id | int | yes | team_api_id of away team |
| match_date | string | no | "YYYY-MM-DD" — defaults to today |

**Response 200:**
| Field | Type | Description |
|---|---|---|
| prediction | string | "H", "D", or "A" |
| prediction_label | string | "Home Win", "Draw", "Away Win" |
| confidence | float | Probability of predicted class (0.0–1.0) |
| probabilities | object | Class → probability mapping (sums to 1.0) |
| home_team | string | Resolved team name |
| away_team | string | Resolved team name |
| match_date | string | Effective match date used |
| container_id | string | Hostname of container |
| model_version | string | Model identifier |
| inference_latency_ms | int | End-to-end inference time |

**Response 400 (invalid team ID):**
```json
{"detail": "Unknown home_team_id: 99999"}
```

**Response 503 (model not loaded):**
```json
{"detail": "Model not yet loaded. Try again in a moment."}
```

## Loose coupling guarantee

Frontend and backend communicate ONLY via these REST endpoints. Backend has no knowledge of frontend technology. Either can be replaced independently.