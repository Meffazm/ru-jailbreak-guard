SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help setup lint format type-check test all clean argocd-ui port-forward-mlflow port-forward-minio port-forward-flyte port-forward-grafana port-forward-prometheus port-forward-alertmanager train-tfidf train-lgbm train-rubert publish-splits register-workflows trigger-cheap

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
	@echo "ArgoCD UI: http://localhost:8080"
	@echo "Login: admin"
	@echo "Password (run separately):"
	@echo "  kubectl -n argocd get secret argocd-initial-admin-secret \\"
	@echo "    -o jsonpath='{.data.password}' | base64 -d"
	kubectl -n argocd port-forward svc/argocd-server 8080:443

port-forward-mlflow:
	@echo "MLflow UI: http://localhost:5000"
	kubectl -n mlflow port-forward svc/mlflow 5000:5000

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

# OMP_NUM_THREADS=1 + KMP_DUPLICATE_LIB_OK=TRUE work around an OpenMP conflict
# between PyTorch and LightGBM on macOS (segfault otherwise).
train-lgbm:
	OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
	GIT_SHA=$$(git rev-parse --short HEAD) GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) \
	MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
	AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_DEFAULT_REGION=us-east-1 \
	uv run python -m ru_jailbreak_guard.models.lgbm_emb \
	  --mlflow-uri http://localhost:5000 \
	  --data-version $$(grep -A 2 'split:' dvc.lock | grep 'md5:' | head -1 | awk '{print $$2}')

# OMP_NUM_THREADS=1 + KMP_DUPLICATE_LIB_OK=TRUE: same OpenMP fix as train-lgbm.
# CPU fine-tune of ruBERT-tiny2 takes ~10–15 min on M-series macOS.
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

# Compute current data_version from dvc.lock (sha256 of split stage out md5s, 12 chars).
# Used by register-workflows + trigger-cheap.
DATA_VERSION = $(shell uv run python -c "import yaml,hashlib; d=yaml.safe_load(open('dvc.lock')); s=d['stages']['split']; print(hashlib.sha256('|'.join(sorted(o['md5'] for o in s['outs'])).encode()).hexdigest()[:12])")
SHA = $(shell git rev-parse --short HEAD)
TRAINER_IMAGE = ghcr.io/meffazm/ru-jailbreak-guard/trainer:sha-$(SHA)

register-workflows:
	@if ! git diff --quiet HEAD --; then echo "WARNING: working tree dirty"; fi
	@echo "Registering with data_version=$(DATA_VERSION) image=$(TRAINER_IMAGE)"
	uv run --group flyte pyflyte register flyte/workflows \
	  --image $(TRAINER_IMAGE) \
	  --project flytesnacks \
	  --domain development \
	  --version $(SHA)

trigger-cheap:
	uv run --group flyte pyflyte run --remote \
	  --image $(TRAINER_IMAGE) \
	  flyte/workflows/pipelines.py cheap_train_pipeline \
	  --data_version $(DATA_VERSION)

port-forward-flyte:
	@echo "Flyte console: http://localhost:8088/console"
	kubectl -n flyte port-forward svc/flyte-binary-http 8088:8088

port-forward-grafana:
	@echo "Grafana: http://localhost:3000  (admin/admin)"
	kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80

port-forward-prometheus:
	@echo "Prometheus: http://localhost:9090"
	kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090

port-forward-alertmanager:
	@echo "Alertmanager: http://localhost:9093"
	kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-alertmanager 9093:9093
