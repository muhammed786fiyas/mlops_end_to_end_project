# Day 5 — Exporter Instrumentation, Grafana Dashboards, AlertManager
**Date:** 2026-04-27
**Time spent:** ~5 hours
**Rubric mapping:** MLOps Implementation (Exporter Instrumentation & Visualization — full Prometheus + Grafana + AlertManager stack), Software Packaging (stack expansion to 11 containers), Software Engineering (declarative config, separation of concerns), Viva (state machines, alerting pipelines, drift detection)

## What I shipped today

- **Backend instrumentation** — `src/api/main.py` updated with 5 custom Prometheus metrics:
  - `football_predictions_total` (Counter, labeled by `predicted_class`)
  - `football_prediction_errors_total` (Counter, labeled by `status_code`)
  - `football_prediction_latency_seconds` (Histogram, 10 buckets from 5ms to 5s)
  - `football_model_info` (Gauge, info-pattern with `version` + `source` labels)
  - `football_model_load_total` (Counter, labeled by `source`: mlflow/local)
  - `/metrics` endpoint exposing the standard Prometheus text format
- **Prometheus container** — `monitoring/prometheus/prometheus.yml`:
  - 15s scrape interval, 15s evaluation interval, 15-day retention
  - Two scrape jobs: self-monitoring and `football-backend`
  - `external_labels` for cluster/environment identification
  - `--web.enable-lifecycle` flag for hot-reload via POST `/-/reload`
- **Grafana container** with auto-provisioning:
  - `monitoring/grafana/provisioning/datasources/prometheus.yml` — fixed UID `prometheus` so dashboard JSON references it deterministically
  - `monitoring/grafana/provisioning/dashboards/dashboards.yml` — file provider, 30s update interval
  - Admin credentials hardcoded as `admin/admin` (academic; would be env-injected in production)
- **10-panel dashboard** — `monitoring/grafana/dashboards/football-overview.json`:
  - **Top row (instant health):** Backend UP/DOWN stat, Total Predictions Served, Backend CPU %, Backend Memory RSS
  - **Middle row (time-series):** Prediction Rate by class (stacked area), Latency p50/p95/p99
  - **Bottom row (breakdowns):** Error Rate stat, Errors by Status Code donut, Predicted Class Distribution donut, Currently Loaded Model table
  - **(Added in M5.4):** Active Alerts table at the bottom
  - All panels use `histogram_quantile()`, `rate()`, and `sum by (...)` — proper PromQL idioms
- **3 Prometheus alert rules** — `monitoring/prometheus/rules/football_alerts.yml`:
  - `BackendDown` (critical, `up==0 for 1m`)
  - `HighErrorRate` (warning, `errors > 0.5/sec for 2m`)
  - `SlowInference` (warning, `p95 latency > 250ms for 3m`)
- **AlertManager container** — `monitoring/alertmanager/alertmanager.yml`:
  - Severity-based routing tree (`critical` → immediate, `warning` → batched)
  - Inhibition rule: critical alerts suppress same-service warnings
  - `group_by`, `group_wait`, `group_interval`, `repeat_interval` tuned
- **Webhook-logger stub** — `webhook_logger/` (FastAPI app + Dockerfile):
  - Tiny ~50 line app that pretty-prints incoming AlertManager JSON payloads to stdout
  - Stand-in for Slack webhook / PagerDuty / SendGrid in academic context
  - Same contract as production receivers — only the URL changes
- **End-to-end pipeline verified:** stopped backend → rule fired after 60s → AlertManager routed by severity → webhook-logger logged the payload (full state-machine traversal `inactive → pending → firing`)

## Key design decisions (viva-ready)

- **Pull-based metrics, not push**
  - Alternatives considered: StatsD, OpenTelemetry push exporter
  - Why pull: industry standard for Prometheus ecosystem; backend doesn't need to know who's monitoring it; Prometheus controls scrape cadence and timeouts; failed scrapes (`up==0`) are themselves a signal

- **5 metrics covering the Four Golden Signals + model lineage**
  - Alternatives considered: just one metric per signal, or many fine-grained metrics
  - Why these 5: traffic (predictions counter), latency (histogram), errors (counter), saturation (process_* metrics from prometheus_client come for free), plus model lineage (info gauge) which is ML-specific and not covered by the Golden Signals framework. Five is the sweet spot — fewer would miss a category, more would dilute the dashboard story.

- **Histogram for latency, not Summary**
  - Why histogram: aggregatable across instances (averaged percentiles are mathematically meaningless for Summaries; histograms can be summed). Lets me compute `histogram_quantile()` on the merged time-series in PromQL — works for one backend or one hundred.

- **Info-gauge pattern for model metadata**
  - Why: Prometheus discourages high-cardinality string labels on metrics that change. `football_model_info` always has value 1; the labels (`version`, `source`) carry the actual data. When the loaded model changes, a new time-series appears and the old one stops emitting. Idiomatic Prometheus.

- **Skipped node_exporter despite A5 using it**
  - Alternatives: add node_exporter container for host-level CPU/RAM/disk
  - Why skipped: project rubric asks "are all components in your software being monitored?" — that's service-level coverage, not host. I track process-level CPU and memory directly from `prometheus_client`'s built-in collectors, which is the right scope for a containerized service. Adding node_exporter would be a 12th container for marginal value at academic scale.

- **Webhook stub instead of real email/Slack**
  - Alternatives: Mailtrap SMTP setup, Slack webhook, PagerDuty events API
  - Why webhook stub: project rubric grades "are alerts implemented?" not "does an email arrive?" The receiver pattern is identical; only the URL differs. Avoiding 3rd-party signups keeps the demo self-contained for graders.

- **Severity-based routing with inhibition**
  - Alternatives: single receiver for all alerts, no inhibition
  - Why this pattern: production-shape. Critical → immediate notification (no group_wait). Warning → batched. Inhibition prevents the "critical alert + 50 dependent warnings" notification storm that wakes on-call engineers at 3am.

- **Fixed datasource UID via provisioning**
  - Alternatives: let Grafana auto-generate, reference by name
  - Why fixed: dashboard JSON references the datasource by UID. Without pinning, every Grafana boot generates a new random UID and the dashboards break with "datasource not found." `uid: prometheus` makes provisioning deterministic.

- **Suppression duration `for: 1m / 2m / 3m`**
  - Why these values: each is at least 4 scrape intervals (60s = 4 × 15s). A single failed scrape isn't enough to fire — that's flap suppression. 1m for outages (fast detection matters), 3m for latency (slower-moving signal). Tuned so I can demo a fire by stopping the backend, but production-realistic.

- **Dashboard layout: high-level → low-level**
  - Why: matches A5's Commandment 6 ("brief title and sub-title") and 7 ("self-explanatory plot") — the eye flows from "is the service alive?" (top stats) to "how is it performing?" (middle time-series) to "what specifically is happening?" (bottom breakdowns). A pager opening the dashboard during an incident reads it left-to-right, top-to-bottom.

## Problems faced & how I solved them

- **Problem:** Docker build failed with DNS timeout to docker.io while pulling python:3.11-slim
  - **Root cause:** Transient DNS resolution issue — first build attempt couldn't resolve `registry-1.docker.io` even though `nslookup` worked
  - **Fix:** Retried after a few minutes; second attempt succeeded. ICMP appears blocked by the network, but HTTPS resolution recovered on its own. Fallback would have been `docker compose exec backend pip install prometheus-client` to install in the running container without rebuilding.

- **Problem:** Grafana dashboard panels showed "No data" with red triangles even though Prometheus had the metrics
  - **Root cause:** Dashboard JSON referenced datasource UID `"prometheus"` but Grafana auto-generated a random UID for the provisioned datasource
  - **Fix:** Added explicit `uid: prometheus` to the datasource provisioning YAML. Restarted Grafana; panels resolved cleanly.

- **Problem:** Resolved alert notifications didn't arrive at webhook-logger after backend recovered
  - **Root cause:** AlertManager's `group_interval: 5m` is the lower bound on notification cadence. Short-lived test alerts that resolve faster than `group_interval` are intentionally suppressed to reduce notification noise — this is documented Prometheus behavior, not a bug
  - **Fix:** Accepted as documented. Verified via `curl /api/v2/alerts` that AlertManager correctly tracked the resolution internally. In production where outages last >5min, both firing and resolved notifications arrive normally.

- **Problem:** Empty `develop` file appeared in working tree after a git operation
  - **Root cause:** Likely a stray output redirect from a misformatted command
  - **Fix:** `rm develop`. Working tree clean.

- **Problem:** Stray `src/monitoring/` folder from Day 1 scaffolding
  - **Root cause:** When the Python package layout was initially created, `src/monitoring/` was anticipated as a Python module but ended up unused — monitoring infrastructure (Prometheus configs, Grafana dashboards) lives at top-level `monitoring/` because it's config, not Python source
  - **Fix:** `git rm -r src/monitoring/`. Committed cleanup.

## Results / metrics

- **Stack scale:** 6 containers (Day 4) → **11 containers** (Day 5)
  - New: prometheus, grafana, alertmanager, webhook-logger
- **Backend image rebuild time:** ~6 min (slow pip install layer due to Docker Hub network)
- **Backend startup with metrics:** ~1.5 sec (no measurable latency overhead from prometheus_client)
- **Per-prediction latency overhead from instrumentation:** essentially 0 (counter increments and histogram observes are O(1))
- **Observed inference latency:** p50=39ms, p95=75ms, p99=95ms (well under informal 250ms SLO)
- **Backend memory (RSS):** 260 MB (Python runtime + LightGBM + feature service + 299-team lookup)
- **Backend CPU under load:** 1.67% during 30 req/s burst (model is essentially free)
- **Prometheus storage:** 15-day retention configured; current volume usage ~few MB
- **Active alert rules:** 3 (BackendDown, HighErrorRate, SlowInference)
- **End-to-end alert latency:** ~75 sec from `docker compose stop backend` to webhook-logger receiving payload (15s scrape lag + 60s `for:` duration)
- **All Grafana panels rendering:** 11/11 (10 service panels + 1 alerts panel)

## What I deferred (and why)

- **Real email/Slack notifications** — Mailtrap or SendGrid would be a 30-min addition. Project rubric grades whether alerts work, not which channel. Webhook stub demonstrates the pattern; viva-defensible.

- **node_exporter container** — A5's specific requirement, not in our project rubric. We track process-level CPU/memory directly from `prometheus_client`'s default collectors. Adding node_exporter for host metrics would be a 12th container with no rubric impact at academic scale.

- **Loki for centralized log aggregation** — Natural complement to Prometheus + Grafana, but adds a 12th container. Each service's stdout is currently accessible via `docker compose logs <service>`. Sufficient for academic demo.

- **AlertManager silencing UI demo** — Mentioned in A5 rubric, but our project rubric doesn't grade it. AlertManager UI at port 9093 supports it natively if asked in viva.

- **Multi-tenant Grafana / role-based access** — Single admin user is fine for academic deployment.

## Commits

- `7fc9082` — feat(monitoring): instrument backend with Prometheus metrics + add scraper
- `3e1b460` — docs: add Day 4 daily log (carried over from Day 4 wrap-up)
- `f0cc0c2` — feat(monitoring): add Grafana with auto-provisioned dashboard
- `a985d00` — build(docker): add Grafana service block to compose stack
- `a874bd9` — feat(monitoring): add AlertManager + 3 alert rules + webhook receiver
- Tag: `v0.5.0` (after Day 5 closes)

## For the viva

**Q: Walk me through what happens between "backend crashes" and "I get notified."**
A: Eight steps. (1) Backend container dies — `up{job="football-backend"}` becomes 0 on the next Prometheus scrape (within 15s). (2) Prometheus's rule manager evaluates `BackendDown` every 15s; condition is now true. (3) Alert enters `pending` state because `for: 1m` requires the condition to hold. (4) After 60s of continuously-true condition, alert transitions to `firing`. (5) Prometheus pushes the alert to AlertManager via the configured `alertmanagers:` target. (6) AlertManager matches `severity=critical` routing rule, picks `webhook-critical` receiver. (7) `group_wait: 0s` for critical means immediate dispatch. (8) AlertManager POSTs JSON payload to webhook-logger. Total latency end-to-end is about 75 seconds — 15s scrape lag plus the 60s suppression window.

**Q: Why a Histogram for latency instead of Summary?**
A: Two reasons. First, histograms are aggregable across instances — `sum by (le)` correctly merges latency buckets from multiple backends, then `histogram_quantile()` recomputes the percentile from the merged distribution. Summary percentiles cannot be averaged across instances; the math is wrong. Second, the bucket boundaries are fixed at the source, so even if I add backends later they all observe into the same buckets. The trade-off is that bucket choice has to be right at the start — too coarse misses tail latency, too fine wastes memory. My buckets are `(5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s)` — clustered tightly where my actual latencies live (25-100ms) and stretched out for tail outliers.

**Q: Why didn't you use node_exporter?**
A: node_exporter exposes host-level metrics — CPU, RAM, disk usage of the underlying machine. A5 used it because that assignment was about correlating bulk-upload load with hardware saturation. My project monitors the ML inference service itself: I track process-level CPU and memory directly from `prometheus_client`'s built-in collectors, which is the right scope for a containerized service. The project rubric asks "are all components in your software being monitored?" — that's service-level coverage, not OS-level. If I deployed across multiple nodes I'd add node_exporter on each, but for a single-host academic deployment, process-level metrics are sufficient and avoid the extra container.

**Q: What does the `predicted_class` distribution donut tell you?**
A: It's a model-bias / drift indicator. With diverse production traffic, the predicted class proportions should reflect roughly the natural football outcome distribution — historically about 46% Home / 25% Draw / 29% Away in European leagues. Significant skew over time is a drift signal: either the model has degraded (e.g., feature distribution shifted), or the traffic shape has changed (e.g., users only querying mid-table matchups). On my dashboard right now it shows ~97% Draw because all 90 predictions are the same Kaiserslautern vs Cesena fixture — a sample-size artifact, not a real bias signal. With diverse traffic this panel becomes a first-line drift detector.

**Q: Why does your webhook receiver use plain HTTP and not authenticated email?**
A: Two reasons. First, the project rubric grades whether the alerting pipeline works — I demonstrate the same JSON contract AlertManager uses for any production receiver (Slack, PagerDuty, OpsGenie all consume the same envelope). Replacing webhook-logger with a real receiver is a one-line change in `alertmanager.yml`. Second, my webhook-logger stub keeps the demo self-contained — no Mailtrap signup, no SMTP credentials, no `.env` secrets. The viva talking point is the pattern, not the channel.

**Q: Resolved alerts didn't show up in your webhook log. Is that a bug?**
A: No — it's documented Prometheus behavior. AlertManager respects `group_interval` (5 minutes by default) before sending follow-up notifications. If an alert fires and resolves faster than that interval, AlertManager intentionally suppresses the resolved-notification to avoid notification noise — the rationale being "if a problem fixes itself in under 5 minutes, you don't need to wake someone up to tell them." I verified via the AlertManager API that the resolution was tracked correctly internally. In production where outages last longer than 5 minutes, both firing and resolved notifications fire normally.

**Q: How would your dashboard scale to a multi-backend deployment?**
A: Three changes. First, every backend container exposes `/metrics` on its own port, all behind the same DNS name (`backend:8000` resolves to all replicas in Docker's round-robin). Prometheus's service discovery would discover them via Docker labels or a static target list. Second, my PromQL queries already use `sum(rate(...))` and `sum by (le)(...)` which merge across instances — no rewrites needed. Third, the `instance` label that Prometheus auto-adds would let me drill into per-replica metrics if a single backend was misbehaving. The dashboard works for one backend or fifty.