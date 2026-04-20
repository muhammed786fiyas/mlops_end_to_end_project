# Day 2 — FastAPI Backend + Frontend + Docker Compose
**Date:** 2026-04-21  
**Time spent:** ~5 hours  
**Rubric mapping:** Demonstration (Web App Front-end UI/UX), Software Engineering (loose coupling, implementation, design), MLOps Implementation (Software Packaging — Docker + Docker Compose)

## What I shipped today

- **API design (LLD)** — `docs/lld/api_endpoints.md` with full request/response specs for 3 endpoints
- **FastAPI backend** — `src/api/main.py` (~200 lines) with:
  - `GET /health` — liveness check with model-loaded status and container ID
  - `GET /teams` — list of 299 teams for frontend dropdown
  - `POST /predict` — match outcome with probabilities + latency + container ID
  - Swagger UI auto-generated at `/docs`
  - Pydantic models for request/response validation
  - CORS middleware for cross-origin frontend calls
  - Lifespan handler loading model + feature service once at startup
- **Live feature service** — `src/api/feature_service.py` (~220 lines) computing the same 32 features as the training pipeline, but for a single (home_team, away_team, date) tuple at inference time
- **Frontend** — vanilla HTML/CSS/JS in `frontend/`:
  - `templates/index.html` — structure with dropdowns, date picker, result panel
  - `static/style.css` — dark theme with color-coded probability bars
  - `static/app.js` — loads teams, calls /predict, renders result
- **Dockerization:**
  - `docker/Dockerfile.backend` — Python 3.11-slim, libgomp1, pip install, copy code+model+data
  - `docker/Dockerfile.frontend` — nginx:1.27-alpine with custom nginx.conf
  - `docker/nginx.conf` — static routing, gzip, 1-hour cache headers
  - `docker-compose.yml` — 2 services on `football-net` bridge network, healthcheck on backend
- **Updated `.dockerignore`** — broad-exclude + selective-include pattern for model/data files

## Key design decisions (viva-ready)

- **Two containers, not one monolith**
  - Alternatives considered: single container serving both API and frontend via uvicorn static mount
  - Why separate: project rubric explicitly requires "two separate services via docker-compose." Loose coupling means either can be replaced without touching the other. Frontend image is ~40 MB (Nginx alpine); backend is ~500 MB (Python + ML stack). Independent scaling possible.

- **Nginx for frontend, not Python http.server**
  - Alternatives considered: Python's `http.server`, serving static from FastAPI
  - Why Nginx: industry-standard static file server, gzip + caching out of the box, 40 MB alpine image, mirrors real-world deployment patterns

- **Vanilla JS, no React/Vue**
  - Alternatives considered: React, Vue, Svelte
  - Why vanilla: 4 interactive elements do not justify 50+ MB of node_modules and a build pipeline. 150 lines of vanilla code meets requirements.

- **IDs in API, names in UI**
  - Names can be ambiguous (multiple "Real Madrid") and change over time; IDs are stable. Frontend translates names↔IDs via /teams endpoint. Clean separation of concerns.

- **Pydantic models for request/response**
  - Alternatives: plain dict returns with manual validation
  - Why Pydantic: FastAPI uses them for auto-generated OpenAPI/Swagger UI + automatic validation + IDE autocomplete. Free tier of professionalism.

- **Lifespan handler for startup**
  - Alternatives: load model on first request
  - Why lifespan: avoids cold-start latency on the first /predict call. Model loads once, reused across all requests.

- **`host="0.0.0.0"` in uvicorn command**
  - Alternatives: default 127.0.0.1
  - Why 0.0.0.0: container's internal localhost is unreachable from outside the container. 0.0.0.0 binds to all interfaces so host traffic routes in.

- **`libgomp1` apt install in Dockerfile**
  - Why: LightGBM's C++ code depends on OpenMP, not included in `python:3.11-slim`. Without this, `import lightgbm` fails in the container.

## Problems faced & how I solved them

- **Problem:** `docker compose build` failed with "COPY failed" for model and data files
  - **Root cause:** `.dockerignore` broadly excluded `data/` and `models/` directories
  - **Fix:** Changed to broad-exclude + selective-include pattern with `!` negation: ignore the whole directory, then un-ignore the specific files we need (e.g., `!data/raw/database.sqlite`)

## Results / metrics

- **Backend image:** football-backend:latest, ~500 MB
- **Frontend image:** football-frontend:latest, ~40 MB
- **Backend startup time:** ~1.5 seconds (model load + feature service init)
- **End-to-end prediction latency:** ~50ms (well under the 200ms SLO)
- **Healthcheck:** passes 30s after start, repeats every 30s
- **Port mapping:** host:8000 → backend:8000, host:8080 → frontend:80
- **Container IDs visible:** e.g., `c87eccb15693` returned in /health and /predict responses

## What I deferred (and why)

- **HTTPS/TLS** — not needed for local demo; would add Let's Encrypt + nginx-proxy in a real deployment
- **Multi-replica backend** — `docker-compose.yml` supports `deploy.replicas: N` but requires Swarm mode; out of scope for Day 2
- **Frontend service worker / PWA** — overkill for this UI
- **Authentication** — not required by rubric

## Commits

- `a1ba894` — docs(lld): add REST API endpoint specifications
- `11696e3` — feat(api): add FastAPI backend with /health, /teams, /predict endpoints  
- `<hash>` — feat(frontend): add vanilla-JS UI with team dropdowns, date picker, probability bars
- `<hash>` — feat(docker): add Dockerfiles and docker-compose for backend+frontend services

## For the viva

**Q: Why two containers instead of one?**
A: Loose coupling — explicitly in the rubric. Frontend and backend can be developed, deployed, scaled, and replaced independently. The only contract between them is the REST API. I could rewrite the backend in Go tomorrow and the frontend would not need any changes.

**Q: How do the two containers find each other?**
A: They don't need to — the browser talks to both, via the host's port forwarding. `localhost:8080` maps to the frontend container's port 80; `localhost:8000` maps to the backend container's port 8000. The Docker `football-net` bridge network is there for future container-to-container communication (e.g., Prometheus scraping the backend on Day 5).

**Q: What does the `container_id` field prove?**
A: It's the container's hostname, which is a random hash like `c87eccb15693`. It proves the API request was served from inside a container, not from my host. If I scaled to 3 replicas, different requests would show different hashes, proving load balancing.

**Q: Why `host="0.0.0.0"` instead of `127.0.0.1`?**
A: 127.0.0.1 inside a container is only reachable by processes inside that container. 0.0.0.0 means "listen on all network interfaces" — which is what makes the port mapping from the host actually work.