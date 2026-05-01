#!/bin/bash
# Bootstrap a freshly-applied YC k8s cluster with ArgoCD + ApplicationSet.
# Run after `terraform apply` succeeds. Requires `yc` CLI configured + kubectl.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/6] Fetching cluster credentials from terraform output..."
KUBECONFIG_CMD=$(cd "$REPO_ROOT/infra" && terraform output -raw kubeconfig_command)
echo "  $ $KUBECONFIG_CMD"
eval "$KUBECONFIG_CMD"

CTX="$(kubectl config current-context)"
echo "  current context: $CTX"
case "$CTX" in
  yc-*|*"-cloud"*|*ru-jailbreak-guard*)
    echo "  ✓ context looks like a YC cluster"
    ;;
  *)
    echo "  ! WARNING: context does not look like a YC cluster — abort if wrong"
    read -r -p "  continue? type 'yes' to proceed: " confirm
    [ "$confirm" = "yes" ] || exit 1
    ;;
esac

# Auto-derive prometheus-operator CRD version from the rendered chart manifest
# so it stays in sync with the chart bump.
PROM_OP_TAG=$(grep -m1 -oE 'prometheus-operator/prometheus-operator:v[0-9]+\.[0-9]+\.[0-9]+' \
  "$REPO_ROOT/gitops/apps/monitoring/kube-prometheus-stack.yaml" | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+')
test -n "$PROM_OP_TAG" || { echo "ERROR: could not detect prometheus-operator tag from monitoring chart."; exit 1; }
echo "[2/6] Pre-installing prometheus-operator CRDs ($PROM_OP_TAG)..."
kubectl apply --server-side --force-conflicts -f \
  "https://github.com/prometheus-operator/prometheus-operator/releases/download/${PROM_OP_TAG}/stripped-down-crds.yaml"

echo "[3/6] Installing ArgoCD..."
kubectl create namespace argocd 2>/dev/null || true
kubectl apply --server-side -n argocd -f \
  https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml \
  --force-conflicts

echo "[4/6] Waiting for ArgoCD server to be ready..."
kubectl -n argocd rollout status deploy/argocd-server --timeout=5m

echo "[5/6] Applying project + ApplicationSet..."
kubectl apply --server-side -f "$REPO_ROOT/gitops/project.yaml"
kubectl apply --server-side -f "$REPO_ROOT/gitops/applicationset.yaml" --force-conflicts

echo "[6/6] Done. ArgoCD is bootstrapping. Useful next steps:"
echo ""
echo "  ArgoCD admin password:"
ARGO_PW=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' 2>/dev/null | base64 -d || echo "(not yet generated, retry in 30s)")
echo "    $ARGO_PW"
echo ""
echo "  Port-forward ArgoCD UI:"
echo "    kubectl -n argocd port-forward svc/argocd-server 8080:443"
echo "    # then open https://localhost:8080  (admin / $ARGO_PW)"
echo ""
echo "  Watch app sync:"
echo "    kubectl get application -n argocd -w"
echo ""
echo "  Tear down: ./scripts/teardown_cloud.sh"
