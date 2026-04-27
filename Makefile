SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help setup lint format type-check test all clean argocd-ui port-forward-mlflow port-forward-minio train-tfidf train-lgbm train-rubert

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
	uv run ty check src tests

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
