# ru-jailbreak-guard

End-to-end MLOps pipeline for a Russian-language jailbreak classifier. Course-defense
project for the OTUS MLOps program. The classifier itself is intentionally small;
the focus is the platform around it: data versioning, experiment tracking, model
registry, GitOps deploy, batch retraining, drift detection and observability.

The model takes a Russian prompt and returns a probability that it contains a
jailbreak attempt (system-prompt override, role-play coercion, token smuggling,
context switching, etc.). Three model families are trained and compared in a
single MLflow registry; the highest-scoring one is promoted to `@champion`.

| Family            | Validation F1 | Test F1 | Latency (CPU) |
|-------------------|--------------:|--------:|--------------:|
| TF-IDF + LogReg   | 0.980         | 0.978   | ~5 ms         |
| LightGBM + ruBERT | 0.996         | 0.997   | ~25 ms        |
| ruBERT-tiny2 (FT) | 0.999         | 0.998   | ~80 ms        |

Trained on a 29.5K-row deduplicated split derived from `dmtrdr/russian_prompt_injections`
(Apache-2.0), HiveTraceRed adversarial templates and `hivetracered`-generated variations,
plus a benign Russian Wikipedia sample.

## Architecture

```
┌────────────────┐  DVC      ┌─────────────┐  Flyte    ┌─────────────┐
│ Raw datasets   │──pipeline─▶│  Splits S3 │──@task───▶│  Trainer    │
│ (HF hub + WG)  │           │  (parquet)  │           │  (3 fams)   │
└────────────────┘           └─────────────┘           └──────┬──────┘
                                                              │
                                                              ▼
                                              ┌──────────────────────────┐
                                              │  MLflow registry         │
                                              │  @production / @champion │
                                              └──────────┬───────────────┘
                                                         │ alias
                                                         ▼
┌──────────────────┐   GitHub Actions   ┌─────────────────────────────┐
│  gitops/apps/    │◀──auto-promote PR──│  ArgoCD ApplicationSet      │
│  values.yaml     │                    │  (sync ⇒ k8s)               │
└──────────────────┘                    └─────────────┬───────────────┘
                                                      │
              ┌───────────────────────────────────────┼──────────────────────────┐
              ▼                                       ▼                          ▼
    ┌─────────────────┐                  ┌─────────────────────┐    ┌──────────────────┐
    │ KServe ISVCs    │                  │ Streamlit UI        │    │ Prometheus +     │
    │ tfidf, lgbm,    │── /predict ─────▶│ 3-way comparison    │    │ Grafana +        │
    │ rubert-ft       │                  │                     │    │ Alertmanager     │
    └────────┬────────┘                  └─────────────────────┘    └────────┬─────────┘
             │                                                               │
             │ 1% sample                                                     │ scrape
             ▼                                                               │
    ┌─────────────────┐  Flyte cron     ┌─────────────────────┐              │
    │ predictions S3  │────drift────────▶ Pushgateway gauge   │──────────────┘
    └─────────────────┘                  └─────────────────────┘
```

## Tech stack

| Layer              | Tooling                                                        |
|--------------------|----------------------------------------------------------------|
| Language / build   | Python 3.12, `uv`, `ruff`, `ty`, `pytest`                      |
| Data versioning    | DVC + S3 remote                                                |
| Datasets           | HuggingFace Hub, `hivetracered`, WildGuardMix                  |
| Models             | scikit-learn, LightGBM, transformers (ruBERT-tiny2)            |
| Tracking + registry| MLflow 3.11 (Postgres backend, S3 artifact store)              |
| Orchestration      | Flyte 1.16 (`@task` / `@workflow` / `LaunchPlan`)              |
| Serving            | KServe 0.18 (Standard / RawDeployment), FastAPI custom predictor |
| UI                 | Streamlit 1.57 (3-way model comparison, prediction history)    |
| Containerization   | Docker, GitHub Container Registry (single-arch linux/amd64)    |
| Cluster            | Kubernetes 1.33 (Yandex managed), local Docker Desktop k8s     |
| GitOps             | ArgoCD ApplicationSet (one app per `gitops/apps/*` directory)  |
| CI/CD              | GitHub Actions (lint + tests, image build, auto-promote)       |
| IaC                | Terraform (Yandex Cloud k8s + node group + VPC + IAM)          |
| Observability      | kube-prometheus-stack 84.5, Pushgateway, custom dashboards     |

## Quick start (local development, no cluster)

```bash
git clone https://github.com/Meffazm/ru-jailbreak-guard.git
cd ru-jailbreak-guard
make setup                   # uv sync (dev + flyte groups via `uv run --group flyte`)
make all                     # lint + type-check + tests

# Build splits via the DVC pipeline (requires HF login for WildGuardMix)
huggingface-cli login
dvc repro

# Train each family locally
make train-tfidf
make train-lgbm
make train-rubert
```

`make help` lists every target.

## Cloud deploy (Yandex Cloud)

```bash
yc init                          # one-time: pick cloud + folder
make cloud-versions              # show available REGULAR-channel k8s versions
make cloud-up                    # terraform apply + bootstrap CRDs + ArgoCD
make cloud-down                  # terraform destroy + verify no dangling disks
```

`make cloud-up` provisions a 1-node managed-k8s cluster (8 vCPU / 32 GB / 128 GB),
auto-detects the latest REGULAR-channel k8s version, and installs the
prometheus-operator CRDs at the version pinned in the rendered chart manifest.
The same `gitops/` tree drives both local and cloud deploys; the cluster
context picks the realm.

After ArgoCD reports all 14 apps healthy:

```bash
make publish-splits             # upload data/splits/ to s3://splits/<data_version>/
make register-workflows         # register Flyte workflows at the current SHA
make trigger-cheap              # run cheap_train_pipeline on the cluster
python scripts/auto_promote.py  # set @production + @champion aliases
make port-forward-ui            # http://localhost:8501 for the Streamlit UI
```

## Screenshots

ArgoCD ApplicationSet with all 14 apps healthy:

![ArgoCD apps](screenshots/argocd-apps.png)

MLflow registry, all three families with `@production` and `@champion` aliases:

![MLflow models](screenshots/mlflow-models.png)

Flyte execution graph for `cheap_train_pipeline`:

![Flyte execution](screenshots/flyte-execution.png)

Streamlit 3-way model comparison UI:

![Streamlit UI](screenshots/streamlit.png)

Grafana ML-metrics dashboard (predict latency, throughput, family breakdown):

![Grafana ML](screenshots/grafana-ml.png)

Prometheus alerts (predictor down, drift detected, error rate, latency):

![Prometheus alerts](screenshots/prometheus-alerts.png)

Additional captures live in [`screenshots/`](screenshots/): MinIO buckets,
Postgres schema, GitHub Actions runs, Terraform apply, Flyte launch plans,
KServe `kubectl` view, Grafana service-health, MLflow experiments.

## Layout

```
ru-jailbreak-guard/
├── src/ru_jailbreak_guard/   # data pipeline, models, drift, predictors, sampling
├── src/ui/                   # Streamlit app
├── tests/                    # pytest, mirrors src/ + flyte/ + scripts/
├── flyte/workflows/          # tasks.py, pipelines.py, launchplans.py, drift.py
├── gitops/                   # ArgoCD watches this tree
│   ├── apps/                 # k8s manifests per app (postgres, mlflow, kserve, ...)
│   └── helm-values/          # Helm value inputs rendered into apps/
├── infra/                    # Terraform: Yandex Cloud k8s + VPC + IAM
├── docker/                   # Dockerfiles: trainer, 3 predictors, ui-streamlit
├── scripts/                  # bootstrap_cloud.sh, teardown_cloud.sh, auto_promote.py ...
├── notebooks/                # eda.ipynb (EDA + 3-model comparison) + builder
├── .github/workflows/        # ci.yaml, images.yaml, promote.yaml
├── dvc.yaml + dvc.lock       # data pipeline (fetch -> merge -> dedup -> split)
├── params.yaml               # DVC pipeline parameters
├── pyproject.toml + uv.lock  # locked dependencies (pinned, ==)
└── Makefile                  # all developer entry points
```

## Notable design decisions

- **Three-family registry, single workflow.** `cheap_train_pipeline` retrains
  TF-IDF + LightGBM in parallel; `gpu_train_pipeline` retrains ruBERT separately.
  `evaluate_and_promote` reads MLflow runs filtered by `data_version`, picks the
  best run per family, sets `@production`, then sets `@champion` on the overall
  winner.
- **Drift detection in one task, not three.** The split tasks would have serialised
  30K texts as a Flyte output, exceeding the 2 MB inter-task cap. Both fetches
  happen inside `compute_drift_task`; only a small report dict crosses task
  boundaries. Sampling caps training-side at 5K texts.
- **GitOps without overlays.** Every app under `gitops/apps/<name>/` is plain
  YAML or a tiny Helm chart whose values come from `gitops/helm-values/<name>.yaml`.
  No kustomize, no environment branches; the cluster context picks the realm.
- **`@production` is set; `@champion` is not cleared.** `promote()` writes the
  winning family's `@champion` alias but does not unset stale `@champion` on
  losing families. MLflow does not support cross-model atomic alias swaps.
- **Single-arch images, rolling `:main` tag.** `linux/amd64` only. Push to `main`
  also overwrites `:main`, so YAML-only commits to `gitops/` can reference a stable
  pointer when no new image was built. Production task pods register with the
  explicit `:sha-<commit>` tag because kubelet caches by tag.
- **Dependencies pinned with `==`.** Drift between local and CI is the single
  biggest source of "works on my machine" failures in MLOps; every dependency
  in `pyproject.toml` uses an exact version.

## License

MIT. See [LICENSE](LICENSE).
