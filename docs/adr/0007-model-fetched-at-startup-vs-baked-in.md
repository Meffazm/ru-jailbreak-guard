# ADR-0007: Fetch registered model at pod startup via MLflow client

**Status:** Accepted
**Date:** 2026-04-26
**Deciders:** Dmitrii Velibekov

## Context

Custom predictor images (ADR-0006) need to load a trained model artifact before
serving traffic. Two approaches:

- **Bake the model into the image at build time** — `COPY model.joblib /app/model/`
  during `docker build`. Image is self-contained; cold start is instant.
- **Fetch the model at pod startup** — container reads a `MODEL_VERSION_PIN`
  environment variable, calls `mlflow.artifacts.download_artifacts(...)` on the
  in-cluster MLflow server, and loads the artifact before marking itself ready.

Constraints:
- We want GitOps to be the single source of truth for what model version is live.
  Bumping a version should be a diff in `gitops/apps/model-tfidf/values.yaml`,
  not a CI image rebuild.
- Models are stored in MLflow's artifact store (MinIO), which is already in-cluster.
  The download path is fast (LAN speed, no internet hop).
- Cold-start latency must be acceptable: TF-IDF joblib artifact is ~5 MB
  (download + load: ~5–15 s); LightGBM requires ruBERT-tiny2 (~45 MB) +
  LightGBM model file (~2 MB) (total: ~30–90 s including HF Hub download on
  first pod start, subsequent starts use the node's Docker layer cache or a
  pre-pulled image).

## Decision

Predictor containers fetch the registered model at startup via `mlflow.client.MlflowClient`.
The exact version is pinned by the `MODEL_VERSION_PIN` environment variable
(an integer version number, not an MLflow stage alias).
The variable is set in the KServe `InferenceService` manifest rendered by
the Helm chart in `gitops/apps/model-<family>/values.yaml`.

A new model promotion is a two-step GitOps operation:
1. Promote the run to Production in the MLflow UI (registers the version).
2. Update `MODEL_VERSION_PIN` in `values.yaml`, commit, push → ArgoCD rolls a
   new pod that fetches the pinned version.

## Consequences

### Positive

- Image is stable across model versions. A new training run does not require
  a CI image build; only a config change is needed. Image build frequency
  drops from "every training run" to "every dependency change."
- Exact version is reproducible: `MODEL_VERSION_PIN=7` always loads model
  version 7, regardless of what stage aliases point to at the time the pod starts.
  Using stage aliases (`Staging`, `Production`) instead would make pod restarts
  non-deterministic if an alias is re-pointed between the restart and the load.
- GitOps diff is a meaningful audit trail: the commit that bumps
  `MODEL_VERSION_PIN` records who promoted which model version and when.

### Negative

- Every pod start hits the MLflow server. A MLflow outage blocks new pod starts
  (and therefore rollouts). Existing running pods are unaffected.
- Cold start is slower than baked-in: 5–15 s for TF-IDF, 30–90 s for LightGBM
  (first start with ruBERT download; subsequent starts are faster if the HF cache
  is warm on the node or the model is pre-pulled into the cluster).
  KServe's readiness probe delays traffic until the pod signals ready; no traffic
  is dropped during load. Cold start latency is acceptable at this scale.
- The predictor startup code adds a dependency on `mlflow` in the serving image.
  This is already present as a tracking client; no new dependency introduced.

### Neutral

- `MODEL_VERSION_PIN` is validated at startup; if the version does not exist in
  the registry, the pod crashes with a clear error rather than silently serving
  a wrong model.
- Pre-pulling ruBERT-tiny2 into the cluster (via the `docker save | ctr images import`
  pattern documented in `docs/local-setup.md`) eliminates the HF Hub network hop
  for LightGBM cold start in airgapped or rate-limited environments.

## Alternatives considered

- **Bake model into image at build time** — rejected because it couples the CI
  pipeline to every training run. Promotes a heavyweight workflow (build + push +
  deploy) for what is fundamentally a config change (which version is live).
- **Use MLflow stage alias (`Production`) instead of pinned version** — rejected
  because alias re-pointing is not atomic with pod restart; a pod restarting
  mid-promotion could load the old or new model depending on timing. Pinned
  version is deterministic.
- **KServe model storage initializer (init container)** — rejected because it
  requires the model to be stored in a format KServe's built-in downloaders
  understand (S3 URI pattern). Our custom startup code is equivalent and lets
  us add validation logic (schema check, warm-up inference) before marking ready.
