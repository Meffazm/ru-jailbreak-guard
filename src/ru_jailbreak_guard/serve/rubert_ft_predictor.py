"""Concrete fine-tuned ruBERT predictor.

Loads a HF SequenceClassification model + tokenizer from MLflow registry's
artifact path. Exposes the same response schema as the other predictors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import torch

from ru_jailbreak_guard.serve.predictor import Predictor


class RubertFtPredictor(Predictor):
    family = "rubert_ft"

    def __init__(
        self,
        *,
        version: str,
        data_version: str,
        model_dir: Path,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        super().__init__(version=version, data_version=data_version)
        self._model_dir = model_dir
        self._device = device
        self._max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None

    @classmethod
    def from_local(
        cls,
        *,
        model_dir: Path,
        version: str,
        data_version: str,
    ) -> RubertFtPredictor:
        return cls(version=version, data_version=data_version, model_dir=model_dir)

    @classmethod
    def from_mlflow(cls) -> RubertFtPredictor:
        """Build from MLflow registry. Reads env vars."""
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        model_name = os.environ["MLFLOW_MODEL_NAME"]
        version = os.environ["MODEL_VERSION_PIN"]
        device = os.environ.get("PREDICTOR_DEVICE", "cpu")

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version(name=model_name, version=version)
        run_id = model_version.run_id
        data_version = client.get_run(run_id).data.tags.get("data_version", "unknown")  # ty: ignore[invalid-argument-type]
        local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        return cls(
            version=version,
            data_version=data_version,
            model_dir=Path(local_dir),
            device=device,
        )

    def load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_dir).to(
            self._device
        )
        self._model.eval()

    def _predict_one(self, text: str) -> tuple[str, float]:
        assert self._tokenizer is not None and self._model is not None, "predictor not loaded"
        with torch.no_grad():
            enc = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)
            logits = self._model(**enc).logits
            prob_jb = float(torch.softmax(logits, dim=-1)[0, 1].cpu().item())
        if prob_jb >= 0.5:
            return "jailbreak", prob_jb
        return "benign", 1.0 - prob_jb


def main() -> None:
    """Container entrypoint."""
    import uvicorn

    from ru_jailbreak_guard.serve.predictor import build_app

    predictor = RubertFtPredictor.from_mlflow()
    app = build_app(predictor=predictor)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
