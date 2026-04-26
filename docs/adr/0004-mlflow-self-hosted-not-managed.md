# ADR-0004: Self-host MLflow tracking and registry

**Status:** Accepted
**Date:** 2026-04-26
**Deciders:** Dmitrii Velibekov

## Context

Phase 2 introduces experiment tracking and model registry. The options are:

- **Managed service** — Databricks-managed MLflow, Comet ML, Weights & Biases, etc.
- **Self-hosted** — deploy the open-source MLflow server ourselves, backed by
  a relational store and an artifact store.

Constraints:
- Yandex Cloud has no managed MLflow offering. The closest analog (Yandex DataSphere)
  does not expose an MLflow-compatible API.
- Databricks-managed MLflow (the canonical managed path) is priced per compute unit;
  even the free tier requires a Databricks workspace setup that is out of scope for
  this project.
- Budget is 1000 RUB total (YC grant). Any managed SaaS that charges per experiment
  log would consume that budget before the training pipelines are even built.
- The project is portfolio-first: the Helm wiring and ArgoCD app definition are
  themselves demonstrable artifacts.
- We already run Postgres (ADR-0008) and MinIO for other components; MLflow can
  share both with no additional infrastructure.

## Decision

Self-host **MLflow** (tracking server + model registry) in-cluster via the
`community-charts/mlflow` Helm chart, managed by ArgoCD like every other app.
Artifact store is MinIO (S3-compatible). Metadata store is the shared Postgres instance.

## Consequences

### Positive

- Zero ongoing cost: runs locally for free; on Yandex Managed K8s it shares the
  Postgres and MinIO pods already present.
- The ArgoCD app definition, Helm values, and Postgres/MinIO wiring are
  demonstrable portfolio artifacts — concrete evidence of "I can deploy and maintain
  an ML platform stack."
- MLflow tracking client code (`mlflow.set_tracking_uri(...)`, `mlflow.log_metric(...)`)
  is 100% portable; migrating to a managed MLflow later requires only a URI change,
  not a client rewrite.
- Unified artifact store: the same MinIO bucket that holds DVC-managed datasets
  can hold MLflow model artifacts under a different prefix, reducing the number
  of storage systems to operate.

### Negative

- We maintain the MLflow server. Upgrades, certificate rotation, Postgres
  migrations, and MinIO bucket policies are our responsibility.
  For a single-team loop this is low operational overhead; estimate ~1 h/quarter.
- MLflow's Helm chart (`community-charts/mlflow`) is community-maintained, not
  official. Chart quality varies; pinning an exact chart version in ArgoCD
  mitigates surprise upgrades.
- Cold-start ordering matters: MLflow fails its startup probe if Postgres is not
  yet ready. ArgoCD sync waves handle this, but the dependency must be declared
  explicitly in the Application manifest.

### Neutral

- MLflow UI is accessible via `make port-forward-mlflow` locally, and via
  a `NodePort` or `Ingress` in cloud — same pattern as ArgoCD and other UIs.
- Model registry aliases (Staging / Production) are used by the predictor
  startup logic in ADR-0007; the registry feature is not an optional add-on.

## Alternatives considered

- **Databricks-managed MLflow** — rejected because it requires a Databricks
  workspace and charges per compute unit; out of budget.
- **Weights & Biases** — rejected because it is not MLflow-compatible; switching
  would require rewriting tracking calls and the registry lookup in ADR-0007.
- **Comet ML free tier** — rejected because the free tier limits experiment
  retention and lacks a model registry API compatible with MLflow client code.
- **No experiment tracking (plain CSV logs)** — rejected because the course
  rubric explicitly requires an ML experiment tracker, and the portfolio defense
  would be weaker without a live registry demonstrating model promotion.
