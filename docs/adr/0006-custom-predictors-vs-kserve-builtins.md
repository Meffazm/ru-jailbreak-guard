# ADR-0006: Use custom FastAPI predictors instead of KServe built-in runtimes

**Status:** Accepted
**Date:** 2026-04-26
**Deciders:** Dmitrii Velibekov

## Context

KServe ships built-in serving runtimes for common model formats:

- `sklearn` runtime — loads a joblib/pickle model and wraps it in a V2 Inference
  Protocol endpoint.
- `huggingface` runtime — loads a HuggingFace `AutoModel` / `pipeline` and
  exposes a V2 endpoint.
- `lgbm` runtime — loads a LightGBM `.txt` model file.

An alternative is to write a custom predictor: a Docker image with a FastAPI
server that we fully control.

Phase 2 deploys two model families:
- **TF-IDF + LogisticRegression** — a sklearn pipeline serialised with joblib.
- **ruBERT embeddings + LightGBM** — ruBERT-tiny2 encodes the text at inference
  time; LightGBM classifies the embedding.

Phase 3 will add fine-tuned ruBERT-tiny2 (direct classification head).

Constraints:
- We want a single response schema across all families so the UI, Prometheus
  metrics, and downstream consumers do not need family-specific parsing:
  `{ label, confidence, model_family, model_version, data_version, latency_ms }`.
- KServe's built-in runtimes each expose their own V2 payload schema; combining
  them requires schema translation at the consumer layer.
- The ruBERT + LightGBM family does not map cleanly to any single built-in runtime
  (it is not a plain sklearn pipeline nor a plain LightGBM model).

## Decision

Implement a `Predictor` ABC in `src/serving/base.py` with `predict(text: str) -> PredictResponse`.
Each model family subclasses it. A shared FastAPI `app.py` wraps the ABC and
exposes `GET /health` and `POST /predict`. Each family has its own Docker image
(built in CI) with the family's dependencies installed.

KServe `InferenceService` manifests use `containers[].image` pointing to these
custom images, not `predictor.sklearn` / `predictor.lgbm` / `predictor.huggingface`.

## Consequences

### Positive

- Uniform response schema across all families. UI, Prometheus exporters, and
  integration tests are written once against `PredictResponse`.
- The ruBERT + LightGBM family (encode then classify) is implemented naturally
  in Python; no shoehorning into a runtime that expects a single model file.
- Per-family Prometheus metrics (`predict_latency_seconds{family="tfidf"}`,
  `predict_latency_seconds{family="lgbm"}`) are instrumented in the shared
  FastAPI app, not bolted on externally.
- Base class + shared app total ~150 lines. Concrete predictors are ~30–50 lines
  each. The maintenance burden is small and concentrated.

### Negative

- We build and push Docker images per family. CI pipeline (`images.yaml`) must
  build two images in Phase 2, three in Phase 3.
- We maintain the FastAPI servers; security patches (e.g. `fastapi` CVEs) require
  image rebuilds. Mitigated by dependabot and pinned base images.
- We lose KServe's built-in model storage initializer for the sklearn runtime.
  Model loading is handled instead by the startup logic in ADR-0007.

### Neutral

- KServe health checks (`/health`) and readiness probes still apply; we implement
  the same paths the built-in runtimes expose.
- The shared `app.py` is tested with `pytest` + `httpx.AsyncClient`; no KServe
  cluster needed for unit tests.

## Alternatives considered

- **KServe `sklearn` runtime for TF-IDF, `huggingface` for ruBERT** — rejected
  because each runtime has a different V2 payload shape, and the ruBERT + LightGBM
  family does not fit either runtime cleanly (it is a two-stage pipeline).
- **KServe `lgbm` runtime for LightGBM** — rejected for the same two-stage reason;
  the ruBERT encoding step requires Python code that the built-in runtime cannot execute.
- **Torchserve or Triton** — rejected because they add heavy infrastructure
  (handler archives, model repository conventions) that outweigh the benefit
  at this project's scale.
