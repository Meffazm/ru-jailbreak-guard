#!/bin/bash
# Bootstrap a freshly-applied YC k8s cluster with ArgoCD + ApplicationSet.
# Run after `terraform apply` succeeds. Requires `yc` CLI configured + kubectl.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/5] Fetching cluster credentials from terraform output..."
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

echo "[2/5] Installing ArgoCD..."
kubectl create namespace argocd 2>/dev/null || true
kubectl apply --server-side -n argocd -f \
  https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml \
  --force-conflicts

echo "[3/5] Waiting for ArgoCD server to be ready..."
kubectl -n argocd rollout status deploy/argocd-server --timeout=5m

echo "[4/5] Applying project + ApplicationSet..."
kubectl apply --server-side -f "$REPO_ROOT/gitops/project.yaml"
kubectl apply --server-side -f "$REPO_ROOT/gitops/applicationset.yaml" --force-conflicts

echo "[5/5] Done. ArgoCD is bootstrapping. Useful next steps:"
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
