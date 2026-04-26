SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help setup lint format type-check test all clean argocd-ui port-forward-mlflow port-forward-minio

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
