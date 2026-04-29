# ADR-0013: Drift detection + auto-promotion mechanics

**Status:** Accepted
**Date:** 2026-04-29
**Deciders:** Dmitrii Velibekov

## Context

Phase 7 closes the GitOps + retraining loop introduced in Phases 5–6:
- Production traffic must become observable (no logging today)
- Drift must trigger retraining (cron-only retraining today)
- Promoted MLflow versions must reach predictor pods (manual SHA bumps today)

## Decision

1. **Predictor sampling:** 1% of `/predict` inputs flushed to `s3://predictions/<date>/<family>/`. Decoupled write path; sampling buffer flushes on row-count or thread-safe.
2. **Drift baseline:** training set at deployed `data_version`. KS-test on length, χ² on cyrillic ratio, mean cosine on cheap char-feature embeddings.
3. **Drift workflow:** weekly Flyte `LaunchPlan` per family (Mon 06/07/08 UTC); emits `ru_jailbreak_drift_detected` gauge to Pushgateway; Prometheus alert fires if drift seen in last 8 days.
4. **Pushgateway as separate ArgoCD app** (`gitops/apps/pushgateway/`) — kube-prometheus-stack chart 84.x doesn't bundle it.
5. **Auto-promote:** hourly GitHub Actions cron reads MLflow `@production` per family, diffs against `modelVersion` in `gitops/apps/model-*/values.yaml` (NOT `MODEL_VERSION_PIN` — that's the rendered env var; the Helm input key is `modelVersion`), opens auto-mergeable PR if mismatch.
6. **Disagreement counter:** Streamlit UI emits `ru_jailbreak_disagreement_total{champion_label, other_family, other_label}` from `_view_comparison` after all 3 predictions land. UI exposes `/metrics` on port 9100 via `prometheus_client.start_http_server`.

## Consequences

- Auto-promote requires MLflow be reachable from GitHub-hosted runners. Local-first install has no public MLflow; the workflow no-ops cleanly with a single log line. Phase 8 cloud install populates `vars.MLFLOW_TRACKING_URI`.
- Sampled traffic accumulates in MinIO; bucket lifecycle deferred to Phase 8.
- Pushgateway is single-replica, ephemeral storage — sufficient for weekly drift cadence.
- Streamlit's hot-reload re-imports the module; the metrics server bind is idempotent (catches `OSError` on rebind).
