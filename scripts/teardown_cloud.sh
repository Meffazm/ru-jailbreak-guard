#!/bin/bash
# Destroy the YC infra brought up by infra/main.tf.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "WARNING: this will destroy the YC k8s cluster + node group + VPC + IAM bindings."
echo "All in-cluster data (PVCs, MLflow runs, MinIO buckets, predictions) will be lost."
echo ""
read -r -p "Type 'destroy' to confirm: " confirm
[ "$confirm" = "destroy" ] || { echo "Aborted."; exit 1; }

command -v yc >/dev/null 2>&1 || { echo "ERROR: yc CLI not installed."; exit 1; }
yc iam create-token >/dev/null 2>&1 || { echo "ERROR: 'yc iam create-token' failed - run 'yc init' first."; exit 1; }

cd "$REPO_ROOT/infra"
TF_VAR_yc_token="$(yc iam create-token)" \
TF_VAR_yc_cloud_id="$(yc config get cloud-id)" \
TF_VAR_yc_folder_id="$(yc config get folder-id)" \
  terraform destroy -auto-approve

echo ""
echo "Cluster destroyed. Verify zero charges accruing:"
echo "  yc compute disk list      # should be empty"
echo "  yc managed-kubernetes cluster list  # should be empty"
echo "  yc vpc network list       # should be empty"
echo ""
echo "Final billing check: https://console.cloud.yandex.net/billing"
