# ADR-0001: Use Flyte for ML pipeline orchestration

**Status:** Accepted
**Date:** 2026-04-25
**Deciders:** Dmitrii Velibekov

## Context

The project requires an orchestrator for data preprocessing and ML training pipelines.
Two main candidates from the CNCF / k8s-native ecosystem:

- **Argo Workflows** — general-purpose YAML-defined DAGs, paired with ArgoCD
  (which we already use for GitOps)
- **Flyte** — ML-native orchestrator with typed Python tasks, automatic caching,
  and built-in ML metadata

Constraints:
- Local Docker Desktop k8s with 16 GB allocation (after bumping from 8 GB)
- Yandex Cloud budget cap of 1000 RUB; aggressive teardown between dev sessions
- Author is adopting Flyte at work; project doubles as work-context validation

## Decision

Use **Flyte** (single-binary Helm chart locally, full chart in cloud) as the
orchestrator for both the data pipeline and training pipelines.

## Consequences

### Positive

- Typed Python task definitions; refactor safety + IDE support
- Automatic task-level caching keyed on input hashes; saves recompute on retraining
- Native ML metadata (datasets, models) integrate cleanly with MLflow
- Distinctive choice for the portfolio defense vs. the more common Airflow / Argo Workflows
- Validates a tooling decision the author is making at work simultaneously

### Negative

- ~2.5 GB RAM idle vs. ~1 GB for Argo Workflows; pushes Docker Desktop to 16 GB minimum
- Requires Postgres backend (paired with MLflow's Postgres in a single managed
  cluster — see ADR-0008)
- Doubles cloud cluster cost vs. Argo Workflows (~10 RUB/h vs. ~5 RUB/h);
  acceptable given local-first dev pattern and aggressive teardown
- More components to bootstrap (FlytePropeller + FlyteAdmin + FlyteConsole +
  DataCatalog) vs. Argo Workflows' single controller

### Neutral

- Pairs naturally with KServe — both ML-native — even if not "all from the Argo family"
- Slightly steeper defense narrative ("why not Argo Workflows?") — but defensible
  on technical merits

## Alternatives considered

- **Argo Workflows** — rejected because ML-native primitives (typed tasks,
  caching, metadata) outweigh the resource savings for this project's portfolio goals
- **Apache Airflow (managed YC)** — rejected because the smallest managed cluster
  is ~10K RUB/month, blowing the entire budget on the orchestrator alone
- **Kubeflow Pipelines** — rejected because it's heavier than Flyte without
  commensurate ML-native upside, and tightly couples to Kubeflow's broader
  ecosystem we don't need
- **Plain GitHub Actions cron + Python scripts** — rejected because it doesn't
  satisfy the "ML pipeline orchestrator" expectation of the course rubric and
  lacks lineage/caching
