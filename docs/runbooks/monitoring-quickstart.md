# Runbook: Monitoring Quickstart

**Goal:** Reach Grafana, Prometheus, and Alertmanager UIs in the local cluster.

## Port-forwards

```
make port-forward-grafana       # http://localhost:3000  (admin/admin)
make port-forward-prometheus    # http://localhost:9090
make port-forward-alertmanager  # http://localhost:9093
```

## Dashboards

Two dashboards auto-discovered from the `ru-jailbreak-dashboards` ConfigMap
in the `monitoring` namespace (Grafana sidecar mode, label `grafana_dashboard: "1"`):

- **ru-jailbreak — Service Health** — request rate, latency p50/p95/p99, pod CPU/mem
- **ru-jailbreak — ML Observability** — label rate, confidence distribution,
  token count, cyrillic ratio, deployed-version table

## Alerts

`http://localhost:9090/alerts` lists 5 rules (2 service-health + 3 ML):
- `RuJailbreakHighErrorRate` — 5xx rate > 5% for 5 min (critical)
- `RuJailbreakHighLatency` — p95 inference latency > 500ms for 10 min (warning)
- `RuJailbreakConfidenceCollapse` — median confidence < 0.6 for 1h (warning)
- `RuJailbreakClassImbalanceHigh` — jailbreak rate > 80% for 1h (warning)
- `RuJailbreakClassImbalanceLow` — jailbreak rate < 5% for 1h (warning)

Alertmanager uses the `null` receiver — alerts fire in the UI but go nowhere
external. Phase 8 (cloud) wires Slack/email receivers.

## Verifying scrape targets

`http://localhost:9090/targets` should list 3 ServiceMonitor targets in `UP`
state (one per predictor namespace). If any is `DOWN`:
- Check predictor pod is Running: `kubectl -n model-<family> get pods`
- Check `/metrics` is reachable: `kubectl -n model-<family> port-forward svc/<predictor>-predictor 8001:80 && curl http://localhost:8001/metrics`
- Check ServiceMonitor selector matches the predictor's Service labels:
  `kubectl -n model-<family> get svc --show-labels`

## Generating traffic

```
kubectl -n model-tfidf port-forward svc/ru-jailbreak-tfidf-predictor 8001:80 &
for i in $(seq 100); do
  curl -sX POST http://localhost:8001/predict -H 'Content-Type: application/json' \
    -d '{"text":"тестовая строка для проверки"}' > /dev/null
done
```

In Prometheus: `rate(ru_jailbreak_prediction_label_total[1m])` should return non-zero.

## Re-rendering the chart

```
helm repo update prometheus-community
helm template monitoring prometheus-community/kube-prometheus-stack \
  --version <X.Y.Z> --namespace monitoring \
  --values gitops/helm-values/monitoring.yaml \
  > gitops/apps/monitoring/kube-prometheus-stack.yaml
```

Bump the version in the leading comment of the rendered file. Commit both
the values file (if changed) and the rendered output.
