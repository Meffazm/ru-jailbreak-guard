#!/bin/bash
# Phase 8 — destroy the YC infra brought up by infra/main.tf.
#
# Run after capturing defense screenshots — every hour the cluster runs costs.
#
#   ./scripts/teardown_cloud.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "WARNING: this will destroy the YC k8s cluster + node group + VPC + IAM bindings."
echo "All in-cluster data (PVCs, MLflow runs, MinIO buckets, predictions) will be lost."
echo "If you want to keep MLflow run history, export it via mlflow experiments csv first."
echo ""
read -r -p "Type 'destroy' to confirm: " confirm
[ "$confirm" = "destroy" ] || { echo "Aborted."; exit 1; }

cd "$REPO_ROOT/infra"
terraform destroy -auto-approve

echo ""
echo "Cluster destroyed. Verify zero charges accruing:"
echo "  yc compute disk list      # should be empty"
echo "  yc managed-kubernetes cluster list  # should be empty"
echo "  yc vpc network list       # should be empty"
echo ""
echo "Final billing check: https://console.cloud.yandex.net/billing"
