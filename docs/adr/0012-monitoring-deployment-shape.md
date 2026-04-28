# ADR-0012: Deploy kube-prometheus-stack as pre-rendered helm template YAML

**Status:** Accepted
**Date:** 2026-04-28
**Deciders:** Dmitrii Velibekov

## Context

Phase 6 adds Prometheus + Grafana + Alertmanager + kube-state-metrics +
node-exporter via the `kube-prometheus-stack` chart. ADR-0011 (Phase 5)
established the chart-template-rendered YAML pattern for `gitops/apps/<app>/`.

## Decision

Reuse the same pattern: render `helm template kube-prometheus-stack` once,
commit the rendered output as `gitops/apps/monitoring/kube-prometheus-stack.yaml`,
keep the values input at `gitops/helm-values/monitoring.yaml` (outside
`gitops/apps/` so ArgoCD's ApplicationSet doesn't try to apply it as a
manifest). ServiceMonitors, PrometheusRule, and the Grafana dashboards
ConfigMap are committed alongside as small hand-authored YAML files.

## Consequences

- Same trade-offs as ADR-0011: manual re-render on chart upgrade, auditable
  diff in PR, no Helm in cluster.
- kube-prometheus-stack renders to ~6800 lines (chart 84.x); review-time diff
  is the only way to catch chart-side regressions.
- ServiceMonitor + PrometheusRule + ConfigMap dashboards live alongside the
  rendered chart so the full observability story is in one directory.
- Alertmanager runs with the default `null` receiver — alerts fire but go
  nowhere external. Phase 8 (cloud) will wire Slack/email receivers.
