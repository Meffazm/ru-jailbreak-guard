# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_PROGRESS=1 \
    UV_LINK_MODE=copy

# libgomp1: required by lightgbm at runtime
# git: required by some HF datasets resolvers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.11.7

WORKDIR /app
COPY pyproject.toml uv.lock /app/
COPY src /app/src
COPY flyte /app/flyte
COPY params.yaml /app/
COPY README.md LICENSE /app/

# Install with flyte group; trainer image needs flytekit at runtime
# because Flyte spawns pods that exec `pyflyte-execute` from this image.
RUN uv sync --frozen --no-dev --group flyte

# Default command is overridden by Flyte's task runner with `pyflyte-execute`.
# `sleep infinity` keeps an accidentally-launched container idle without
# triggering uv resolution; smoke tests should override CMD explicitly.
CMD ["sleep", "infinity"]
