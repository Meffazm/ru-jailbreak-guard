# ADR-0014: Phase 8 cloud demo deployment shape

**Status:** Accepted
**Date:** 2026-04-29
**Deciders:** Dmitrii Velibekov

## Context

Phase 8 brings the project to Yandex Cloud managed Kubernetes for the OTUS final
defense. The defense is a **static presentation** with screenshots, tables, and
data captures — not a live demo. The author has a 1000 RUB YC grant; aggressive
teardown between sessions is mandatory.

The original spec (§3.1) called for managed PostgreSQL replacing the in-cluster
pod and Object Storage replacing in-cluster MinIO. Once the defense format was
clarified as screenshots-only, those production-parity decisions become
unnecessary cost.

## Decision

1. **Single `infra/main.tf`** brings up minimum viable YC infra: VPC + subnet +
   service account + IAM bindings + zonal k8s cluster + 1-node group.
2. **One node, 4 vCPU / 16 GB RAM** — fits the entire stack (ArgoCD,
   cert-manager, KServe, in-cluster Postgres, in-cluster MinIO, MLflow, Flyte,
   monitoring + Pushgateway, 3 predictors, UI). Verified locally on Docker
   Desktop's 16 GB allocation.
3. **Reuse existing `gitops/applicationset.yaml` unchanged.** Same 12 apps run
   on YC as run locally. No env-specific values overrides.
4. **No managed PostgreSQL, no Object Storage migration.** In-cluster pods
   (postgres + minio) use PVCs backed by YC Compute disks via the YC CSI driver.
5. **No public ingress, no TLS.** Defense screenshots taken via
   `kubectl port-forward` from the local laptop. Auto-promote stays no-op in
   cloud (no public MLflow URL).

## Consequences

### Positive
- Fastest possible bring-up: single `terraform apply` + bootstrap script.
- Lowest cost: ~250 RUB/day; a 4-hour capture session costs ~40 RUB.
- No environment skew between local dev and cloud capture — same gitops, same
  manifests.
- Tear-down is a single `terraform destroy`.

### Negative
- No production-parity for the defense (managed services + ingress are
  documented as Phase 9 stretch).
- Single-zone cluster has no HA — fine for a one-shot capture, would not be
  acceptable for a real production deployment.
- In-cluster MinIO state is lost on `terraform destroy`. MLflow run history
  archived only as Defense screenshots.

### Neutral
- The ApplicationSet's git revision points at `main` — same as local default.
  No branch switching needed for cloud bring-up.
- IAM service account is `editor` on the folder — broader than strictly
  necessary, but the cluster is transient.

## Alternatives considered

- **Managed PostgreSQL + Object Storage** — rejected because the defense
  presentation doesn't show those production niceties and they would consume
  ~600 RUB / month from a 1000 RUB total budget.
- **Public ingress with cert-manager + Let's Encrypt** — rejected because
  port-forward is sufficient for screenshot capture and skipping ingress
  removes ~50 RUB / month + ~30 minutes of TLS-cert wrangling per bring-up.
- **Multi-zone HA cluster** — rejected as obvious overkill for screenshots.
- **DataSphere GPU integration** — never adopted (deferred from Phase 3 since
  CPU training fits the smoke-data scope).

## Stretch (Phase 9 if budget allows)

Add a `gitops/envs/cloud/values.yaml` overlay that points apps at managed
PostgreSQL + Object Storage + a public ingress. Brings true production parity
but requires additional Terraform for the managed services and ~600 RUB/month
running cost. Out of scope for v1.0.0.
