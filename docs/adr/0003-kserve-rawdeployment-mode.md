# ADR-0003: Use KServe in RawDeployment mode

**Status:** Accepted
**Date:** 2026-04-26
**Deciders:** Dmitrii Velibekov

## Context

KServe supports two installation modes:

- **Serverless mode** — the default; requires Knative Serving + Istio (or Kourier)
  as prerequisites. Knative adds scale-to-zero and traffic-split primitives;
  Istio handles ingress and mTLS.
- **RawDeployment mode** — KServe deploys `InferenceService` resources as plain
  Kubernetes `Deployment` + `Service` objects. No Knative, no Istio.

Constraints:
- Local Docker Desktop k8s with 16 GB allocated. Knative + Istio add ~1.5 GB
  idle RAM and ~15 additional pods. That headroom is needed for Flyte, MLflow,
  and Postgres rather than serving infrastructure.
- Cloud installs (Yandex Managed K8s) should mirror local exactly to avoid
  environment drift. Keeping the same mode gives identical manifests.
- Scale-to-zero is not a goal: we tear down the entire local cluster between
  sessions, and cloud sessions are short. We are not paying for idle serving pods.
- Traffic splitting between model versions is a Phase 4 concern; it will be
  handled at the UI / MLflow experiment level, not via Knative traffic rules.

## Decision

Deploy KServe in **RawDeployment mode** by setting
`kserve.controller.deploymentMode: RawDeployment` in the Helm values.
No Knative or Istio components are installed.

## Consequences

### Positive

- ~1.5 GB lower idle RAM; Docker Desktop stays within 16 GB with headroom.
- Identical installation locally and on Yandex Managed K8s; no environment
  divergence for CI validation or the course portfolio defense.
- Simpler dependency graph: one Helm chart (`kserve`) instead of three
  (kserve + knative-serving + istio). Faster bootstrap, smaller failure surface.
- `InferenceService` pods are visible as ordinary `Deployment` objects —
  easier to inspect, debug, and resource-constrain without Istio/Envoy sidecar overhead.

### Negative

- No scale-to-zero. Serving pods run continuously while deployed.
  Acceptable — we destroy clusters rather than idling them.
- No built-in Knative traffic splitting between revisions. A/B comparison
  between TF-IDF and LightGBM ISVCs will use independent services + UI
  routing, not a single weighted `InferenceService`.
- Some KServe documentation and examples assume Serverless mode; RawDeployment
  manifest differences (no `knative.dev/serving` annotations) require
  reading mode-specific docs carefully.

### Neutral

- Ingress is handled by a plain Kubernetes `Ingress` or `NodePort` rather than
  an Istio `Gateway`. Consistent with how ArgoCD and MLflow UIs are exposed locally.

## Alternatives considered

- **Serverless mode (Knative + Istio)** — rejected because the RAM cost (~1.5 GB)
  trades a feature (scale-to-zero) we don't need for headroom we do need on 16 GB.
- **Serverless mode (Knative + Kourier)** — lighter than Istio but still adds
  Knative's ~1 GB overhead and does not fit the "mirror cloud infra locally" goal,
  since Yandex Managed K8s would require Istio separately.
- **Raw Kubernetes Deployment without KServe** — rejected because KServe provides
  the `InferenceService` CRD, model storage initializer, and health-check conventions
  that Phase 4 (fine-tuned ruBERT) reuses without code changes.
