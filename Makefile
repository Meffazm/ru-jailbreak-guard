SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help setup lint format type-check test all clean argocd-ui port-forward-mlflow port-forward-minio port-forward-flyte port-forward-grafana port-forward-prometheus port-forward-alertmanager port-forward-pushgateway port-forward-predictor-tfidf port-forward-predictor-lgbm port-forward-predictor-rubert-ft port-forward-ui train-tfidf train-lgbm train-rubert publish-splits register-workflows trigger-cheap cloud-versions cloud-up cloud-down

help:
	@echo "ru-jailbreak-guard — top-level commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        — uv sync (install all deps incl. dev)"
	@echo ""
	@echo "Quality:"
	@echo "  make lint         — ruff check"
	@echo "  make format       — ruff format (writes)"
	@echo "  make type-check   — ty check"
	@echo "  make test         — pytest"
	@echo "  make all          — lint + type-check + test"
	@echo ""
	@echo "Local k8s:"
	@echo "  make argocd-ui            — port-forward ArgoCD UI to localhost:8080"
	@echo "  make port-forward-mlflow  — port-forward MLflow UI to localhost:5000"
	@echo "  make port-forward-minio   — port-forward MinIO console to localhost:9001"
	@echo ""
	@echo "Training (run with port-forwards active):"
	@echo "  make train-tfidf  — train TF-IDF + LogReg, log to MLflow, register"
	@echo "  make train-lgbm   — train LightGBM on ruBERT-emb, log to MLflow, register"
	@echo "  make train-rubert — fine-tune ruBERT-tiny2, log to MLflow, register"

setup:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

type-check:
	uv run ty check src tests scripts flyte

test:
	uv run pytest

all: lint type-check test

clean:
	rm -rf .venv .ruff_cache .pytest_cache .ty_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

argocd-ui:
	@echo "ArgoCD UI: https://localhost:8080  (self-signed cert — accept browser warning)"
	@echo "Login: admin"
	@echo "Password (run separately):"
	@echo "  kubectl -n argocd get secret argocd-initial-admin-secret \\"
	@echo "    -o jsonpath='{.data.password}' | base64 -d"
	kubectl -n argocd port-forward svc/argocd-server 8080:443

port-forward-mlflow:
	@echo "MLflow UI: http://localhost:5001  (5000 is macOS AirPlay Receiver)"
	kubectl -n mlflow port-forward svc/mlflow 5001:5000

port-forward-minio:
	@echo "MinIO console: http://localhost:9001  (login: minioadmin/minioadmin)"
	kubectl -n minio port-forward svc/minio 9001:9001

train-tfidf:
	GIT_SHA=$$(git rev-parse --short HEAD) GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) \
	MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
	AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_DEFAULT_REGION=us-east-1 \
	uv run python -m ru_jailbreak_guard.models.tfidf_logreg \
	  --mlflow-uri http://localhost:5000 \
	  --data-version $$(grep -A 2 'split:' dvc.lock | grep 'md5:' | head -1 | awk '{print $$2}')

# OMP_NUM_THREADS=1 + KMP_DUPLICATE_LIB_OK=TRUE: OpenMP conflict between PyTorch
# and LightGBM on macOS — without these, SIGSEGV at startup.
train-lgbm:
	OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
	GIT_SHA=$$(git rev-parse --short HEAD) GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) \
	MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
	AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_DEFAULT_REGION=us-east-1 \
	uv run python -m ru_jailbreak_guard.models.lgbm_emb \
	  --mlflow-uri http://localhost:5000 \
	  --data-version $$(grep -A 2 'split:' dvc.lock | grep 'md5:' | head -1 | awk '{print $$2}')

# Same OpenMP fix as train-lgbm.
train-rubert:
	OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
	GIT_SHA=$$(git rev-parse --short HEAD) GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) \
	MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
	AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_DEFAULT_REGION=us-east-1 \
	uv run python -m ru_jailbreak_guard.models.rubert_ft \
	  --mlflow-uri http://localhost:5000 \
	  --data-version $$(grep -A 2 'split:' dvc.lock | grep 'md5:' | head -1 | awk '{print $$2}')

publish-splits:
	@echo "Uploading data/splits/ to s3://splits/<data_version>/"
	MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
	AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_DEFAULT_REGION=us-east-1 \
	uv run python scripts/publish_splits.py --skip-existing

# Current data_version from dvc.lock — sha256 of split-stage output md5s, 12 chars.
DATA_VERSION = $(shell uv run python -c "import yaml,hashlib; d=yaml.safe_load(open('dvc.lock')); s=d['stages']['split']; print(hashlib.sha256('|'.join(sorted(o['md5'] for o in s['outs'])).encode()).hexdigest()[:12])")
SHA = $(shell git rev-parse --short HEAD)
TRAINER_IMAGE = ghcr.io/meffazm/ru-jailbreak-guard/trainer:sha-$(SHA)

# --copy none skips fast-register tarball upload. Flyte's signed URL points at
# in-cluster `minio.minio.svc.cluster.local:9000`, unreachable from the laptop.
register-workflows:
	@if ! git diff --quiet HEAD --; then echo "WARNING: working tree dirty"; fi
	@echo "Registering with data_version=$(DATA_VERSION) image=$(TRAINER_IMAGE)"
	uv run --python 3.12 --group flyte pyflyte register --copy none flyte/workflows \
	  --image $(TRAINER_IMAGE) \
	  --project flytesnacks \
	  --domain development \
	  --version $(SHA)

# Run an already-registered workflow by FQN; skips fast-register entirely.
trigger-cheap:
	uv run --python 3.12 --group flyte pyflyte run --remote \
	  remote-workflow flyte.workflows.pipelines.cheap_train_pipeline \
	  --data_version $(DATA_VERSION)

port-forward-flyte:
	@echo "Flyte console: http://localhost:8088/console"
	kubectl -n flyte port-forward svc/flyte-binary-http 8088:8088

port-forward-grafana:
	@echo "Grafana: http://localhost:3000  (admin/admin)"
	kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80

port-forward-prometheus:
	@echo "Prometheus: http://localhost:9090"
	kubectl -n monitoring port-forward svc/monitoring-prometheus 9090:9090

port-forward-alertmanager:
	@echo "Alertmanager: http://localhost:9093"
	kubectl -n monitoring port-forward svc/monitoring-alertmanager 9093:9093

port-forward-pushgateway:
	@echo "Pushgateway: http://localhost:9091"
	kubectl -n pushgateway port-forward svc/pushgateway 9091:9091

port-forward-predictor-tfidf:
	@echo "TF-IDF predictor: http://localhost:8001"
	kubectl -n model-tfidf port-forward svc/ru-jailbreak-tfidf-predictor 8001:80

port-forward-predictor-lgbm:
	@echo "LightGBM predictor: http://localhost:8002"
	kubectl -n model-lgbm port-forward svc/ru-jailbreak-lgbm-predictor 8002:80

port-forward-predictor-rubert-ft:
	@echo "ruBERT-tiny2 predictor: http://localhost:8003"
	kubectl -n model-rubert-ft port-forward svc/ru-jailbreak-rubert-ft-predictor 8003:80

port-forward-ui:
	@echo "Streamlit UI: http://localhost:8501"
	kubectl -n ui port-forward svc/ui-streamlit 8501:80

cloud-versions:
	@echo "Available k8s versions per release channel:"
	yc managed-kubernetes list-versions

cloud-up:
	@echo "Bringing up YC k8s cluster + bootstrapping ArgoCD..."
	@command -v yc >/dev/null 2>&1 || { echo "ERROR: yc CLI not installed. See docs/runbooks/cloud-defense-capture.md prereqs."; exit 1; }
	@yc iam create-token >/dev/null 2>&1 || { echo "ERROR: 'yc iam create-token' failed — run 'yc init' first."; exit 1; }
	$(eval K8S_VERSION := $(shell yc managed-kubernetes list-versions 2>/dev/null | grep "REGULAR" | grep -oE '[0-9]+\.[0-9]+' | sort -V | tail -1))
	@test -n "$(K8S_VERSION)" || { echo "ERROR: could not auto-detect k8s version from 'yc managed-kubernetes list-versions'. Pass TF_VAR_k8s_version=X.Y explicitly."; exit 1; }
	@echo "Using latest k8s version: $(K8S_VERSION)"
	cd infra && \
	  TF_VAR_yc_token="$$(yc iam create-token)" \
	  TF_VAR_yc_cloud_id="$$(yc config get cloud-id)" \
	  TF_VAR_yc_folder_id="$$(yc config get folder-id)" \
	  TF_VAR_k8s_version="$(K8S_VERSION)" \
	  terraform init && \
	  TF_VAR_yc_token="$$(yc iam create-token)" \
	  TF_VAR_yc_cloud_id="$$(yc config get cloud-id)" \
	  TF_VAR_yc_folder_id="$$(yc config get folder-id)" \
	  TF_VAR_k8s_version="$(K8S_VERSION)" \
	  terraform apply -auto-approve
	./scripts/bootstrap_cloud.sh

cloud-down:
	./scripts/teardown_cloud.sh
