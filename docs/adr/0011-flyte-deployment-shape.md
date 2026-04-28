# ADR-0011: Deploy Flyte as pre-rendered helm template YAML

**Status:** Accepted
**Date:** 2026-04-28
**Deciders:** Dmitrii Velibekov

## Context

Phase 5 introduces Flyte to the cluster as the orchestrator for training pipelines.
Three deployment shapes were considered:

- **Live Helm chart via ArgoCD** — ArgoCD's native Helm support installs from `flyteorg/flyte-binary` directly
- **Hand-rolled minimal YAML** — author Deployment + Service + ConfigMap by hand, matching `gitops/apps/postgres` and `gitops/apps/mlflow`
- **Pre-rendered helm template YAML** — run `helm template` once, commit the rendered output, ArgoCD applies as plain manifests

Project context: ADR-0003 (KServe RawDeployment), the cert-manager and KServe apps,
and the Bitnami pivot in CLAUDE.md decisions all establish a "raw YAML in
`gitops/apps/`" pattern for this repo.

## Decision

Render `helm template flyte-binary flyteorg/flyte-binary --version v1.16.6
--values gitops/helm-values/flyte.yaml` once locally, commit the rendered file
as `gitops/apps/flyte/flyte-binary.yaml`, and let ArgoCD apply it as plain
manifests. The values input lives at `gitops/helm-values/flyte.yaml` (outside
`gitops/apps/` so ArgoCD's ApplicationSet doesn't try to apply it as a
manifest), and re-rendering is documented in the leading comment of
`flyte-binary.yaml`.

## Consequences

### Positive
- Consistent with the existing pattern: `cert-manager.yaml` v1.18.0 and
  `kserve.yaml` v0.14.0 are also static manifests from upstream.
- ArgoCD does pure `kubectl apply`; no Helm rendering inside the cluster.
- Diff at chart-version-upgrade time is auditable in PR review.
- No need to hand-author Flyte's complex topology (server, RBAC, volumes,
  etc.) — the chart maintainers do it for us.

### Negative
- Chart upgrades require a manual local re-render step.
- Two files must stay in sync (`gitops/helm-values/flyte.yaml` +
  `gitops/apps/flyte/flyte-binary.yaml`); a header comment in the rendered
  file references the re-render command.
- The `cr.flyte.org/flyteorg/flyte-binary` image reference in the chart's
  default values doesn't work behind Docker Desktop's registry mirror
  (returns 401 on the redirect to GHCR). The values input overrides the
  image to `ghcr.io/flyteorg/flyte-binary-release` directly.

### Neutral
- This pattern is increasingly common ("GitOps with rendered manifests")
  and supported by tools like `helmfile template` and `kustomize` for
  similar use cases.

## Alternatives considered

- **Live Helm via ArgoCD** — rejected because it deviates from the repo's
  established raw-YAML pattern (see CLAUDE.md decisions about Bitnami).
- **Hand-rolled minimal YAML** — rejected because Flyte's binary chart
  ships ~500 lines of intermixed RBAC, ConfigMap, Deployment, Service,
  and Job manifests; re-deriving these by hand is error-prone and offers
  no benefit over the rendered chart output.
