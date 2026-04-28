# Runbook: Manual Retrain

**When to use:** Operator wants to retrain models on a new data version, or kick
off a one-off retrain outside the cron schedule.

**Prereqs:**
- `kubectl config current-context` returns `docker-desktop` (or your cloud cluster)
- ArgoCD shows `flyte` Healthy + Synced
- Trainer image present in cluster (built via CI on push to main, or via local
  `docker buildx build -f docker/trainer.Dockerfile -t ghcr.io/meffazm/ru-jailbreak-guard/trainer:sha-$(git rev-parse --short HEAD) --load .` then `docker save ... | docker exec -i desktop-control-plane ctr -n=k8s.io images import -`)
- Port-forwards open in separate terminals as needed:
  - `make port-forward-mlflow` → http://localhost:5000
  - `make port-forward-minio` → http://localhost:9001
  - `make port-forward-flyte` → http://localhost:8088/console
  - `kubectl -n flyte port-forward svc/flyte-binary-grpc 8089:8089` (for `pyflyte` to reach FlyteAdmin)

## Steps

1. **Refresh data with DVC**
   ```
   uv run dvc repro
   ```
   Re-runs only stages whose inputs changed. Produces fresh
   `data/splits/{train,val,test}.parquet` if anything upstream moved.

2. **Publish splits to MinIO**
   ```
   make publish-splits
   ```
   Idempotent. Computes `data_version` from `dvc.lock` (sha256 of the split
   stage out md5s, 12 chars), uploads to `s3://splits/<data_version>/` if not
   already there.

3. **Configure flytectl to point at local Flyte**
   ```
   mkdir -p ~/.flyte
   cat > ~/.flyte/config.yaml <<'EOF'
   admin:
     endpoint: dns:///localhost:8089
     insecure: true
   EOF
   ```

4. **Update LaunchPlan defaults to the new data_version**
   ```
   make register-workflows
   ```
   Re-registers the workflows + LaunchPlans with the current data_version
   baked into their default inputs. Cron schedules pick up the new value
   on the next tick.

5. **(Optional) Trigger an immediate run**
   ```
   make trigger-cheap        # cheap_train_pipeline (tfidf + lgbm)
   ```

6. **Watch in Flyte console**
   `make port-forward-flyte` → http://localhost:8088/console → flytesnacks/development.

7. **Verify in MLflow**
   New runs appear in the relevant experiments, tagged with
   `flyte_execution_id`, `data_version`, `model_family`.

8. **Promote (optional, manual)**
   ```
   uv run --group flyte pyflyte run --remote \
     --image ghcr.io/meffazm/ru-jailbreak-guard/trainer:sha-$(git rev-parse --short HEAD) \
     flyte/workflows/pipelines.py evaluate_and_promote \
     --data_version <data_version>
   ```
   Updates `@production` per family + `@champion` overall in the MLflow registry.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `download_splits` 404 | Forgot `make publish-splits` after `dvc repro` | Run step 2, retry |
| `train_*` OOMKilled | LightGBM + PyTorch OpenMP collision | Already handled via `OMP_NUM_THREADS=1` env var (set in `gitops/helm-values/flyte.yaml`) |
| `evaluate` raises `No successful runs` | No runs match `tags.data_version` | Verify training tasks tagged correctly; check MLflow UI |
| Trainer image pull stuck | First-time pull from GHCR is slow | Pre-pull on host: `docker pull ... && docker save ... \| ctr import` |
| Cron tick missed | Flyte LaunchPlan paused | In Flyte console → LaunchPlan → Schedule → Activate |
| `flyte-binary-release` image not found | Docker Desktop registry mirror returned 401 on `cr.flyte.org` redirect | Already worked around — values.yaml uses `ghcr.io/flyteorg/flyte-binary-release` directly |

## Phase 5 known limitations

- Aliases (`@production`, `@champion`) are set in MLflow but **not propagated to
  predictor `MODEL_VERSION_PIN`** — that's Phase 7's automation. If you want a
  newly promoted version to actually serve, manually bump the SHA in
  `gitops/apps/model-<family>/values.yaml`.
- Drift-triggered retraining is Phase 7. Phase 5 only supports cron + manual.
- Cron retrains on smoke-sized data (~1300 rows). MinHash dedup O(N) refactor
  for full ~300K data is a separate follow-up PR.
