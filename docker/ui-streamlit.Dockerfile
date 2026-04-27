# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_PROGRESS=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.11.7

WORKDIR /app
COPY pyproject.toml uv.lock /app/
COPY src /app/src
COPY README.md LICENSE /app/

# UI doesn't need any of the modeling stack — slim aggressively.
RUN uv sync --no-dev --frozen \
    && uv pip uninstall --quiet \
       scikit-learn lightgbm transformers torch accelerate joblib \
       matplotlib mlflow datasketch dvc datasets || true

EXPOSE 8501

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/ui/streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--server.headless=true", "--browser.gatherUsageStats=false"]
