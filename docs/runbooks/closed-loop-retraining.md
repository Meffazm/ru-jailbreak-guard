# Runbook: Closed-loop retraining

## Verifying the loop

1. **Sampling alive**: drive predict traffic, check MinIO:
   ```
   kubectl -n model-tfidf port-forward svc/ru-jailbreak-tfidf-predictor 8001:80 &
   for i in $(seq 200); do
     curl -sX POST http://localhost:8001/predict -H 'Content-Type: application/json' \
       -d '{"text":"тестовая строка"}' > /dev/null
   done
   make port-forward-minio
   uv run python -c "
   import boto3
   s3 = boto3.client('s3', endpoint_url='http://localhost:9000',
       aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin', region_name='us-east-1')
   for o in s3.list_objects_v2(Bucket='predictions').get('Contents', []):
       print(o['Key'], o['Size'])
   "
   ```
   Expected: ~2 parquet files (1% of 200 = ~2 samples).

2. **Drift workflow manual run**:
   ```
   uv run --group flyte pyflyte run --remote \
     --image $(TRAINER_IMAGE) \
     flyte/workflows/drift.py drift_detect \
     --data_version $(DATA_VERSION) \
     --family tfidf_logreg
   ```

3. **Drift metric in Prometheus**:
   ```
   make port-forward-prometheus
   # Browser http://localhost:9090 → query: ru_jailbreak_drift_detected
   ```
   Returns 0 or 1 per family.

4. **Auto-promote**: trigger manually via GitHub Actions UI or `gh workflow run auto-promote`.
   - With `MLFLOW_TRACKING_URI` unset (local-first scope): logs "no-op" and exits.
   - With `MLFLOW_TRACKING_URI` set + a registered `@production` alias diff: opens a PR with `modelVersion: "<new>"` updates.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Sampling never triggers | Buffer's `_SAMPLE_RATE` is 0.01 — drive ≥100 requests; or temporarily set `sample_rate=1.0` in `make_default_buffer()` for verification |
| Drift workflow `insufficient_samples` skip | Need ≥100 production samples — drive more traffic, or shorten `days` argument range |
| Pushgateway unreachable from drift pod | `kubectl -n pushgateway get pods` — if Pushgateway pod isn't Running, check ArgoCD `pushgateway` app status |
| Auto-promote opens PR but CI fails | PR sits open; operator notices via watchlist; can manually fix by updating script + retrying |
| Auto-promote runs but `modelVersion` regex doesn't match | Verify the gitops file uses `modelVersion: "<n>"` (Helm input), NOT `MODEL_VERSION_PIN` (rendered env var) |

## Components

- **Sampling:** `src/ru_jailbreak_guard/serve/sampling.py` — `SamplingBuffer` invoked from `Predictor.predict()`
- **Drift logic:** `src/ru_jailbreak_guard/evaluate/drift.py` — KS, χ², cosine; `compute_drift()` returns `DriftReport`
- **Drift workflow:** `flyte/workflows/drift.py` — 4 tasks + `drift_detect` workflow, 3 weekly LaunchPlans in `launchplans.py`
- **Pushgateway:** `gitops/apps/pushgateway/pushgateway.yaml` — Deployment + Service + ServiceMonitor
- **Drift alert:** `gitops/apps/monitoring/alerts.yaml` group `ru-jailbreak-drift`
- **UI disagreement:** `src/ui/streamlit_app.py` — Counter emitted from `_view_comparison`; UI Service exposes port 9100 named `metrics`; ServiceMonitor in `gitops/apps/monitoring/servicemonitors.yaml`
- **Auto-promote:** `scripts/auto_promote.py` + `.github/workflows/promote.yaml`

## Phase 7 known limitations

- Auto-promote no-ops without `vars.MLFLOW_TRACKING_URI` set (local-first scope intentionally has no public MLflow)
- Drift baseline is training set only (not rolling window) — gradual drift detected; sudden traffic spikes look like drift but resolve when traffic normalizes
- Embeddings used for drift's cosine distance are cheap char-feature proxies (length, alpha-count, cyrillic-count, digit-count), NOT real ruBERT embeddings — keeps drift pod light. Trade-off: less semantic drift sensitivity.
- MinIO `predictions/` bucket lacks lifecycle rules — defer to Phase 8 cloud bucket lifecycle config
