"""Concrete LightGBM-on-ruBERT-embeddings predictor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np

from ru_jailbreak_guard.features.embeddings import Encoder
from ru_jailbreak_guard.serve.predictor import Predictor


class LgbmEmbPredictor(Predictor):
    family = "lgbm_emb"

    def __init__(
        self,
        *,
        version: str,
        data_version: str,
        booster_path: Path,
        encoder: Any,
    ) -> None:
        super().__init__(version=version, data_version=data_version)
        self._booster_path = booster_path
        self._encoder = encoder
        self._booster: lgb.Booster | None = None

    @classmethod
    def from_local(
        cls,
        *,
        booster_path: Path,
        encoder: Any,
        version: str,
        data_version: str,
    ) -> LgbmEmbPredictor:
        return cls(
            version=version,
            data_version=data_version,
            booster_path=booster_path,
            encoder=encoder,
        )

    @classmethod
    def from_mlflow(cls) -> LgbmEmbPredictor:
        """Build from MLflow registry. Reads env vars + builds Encoder."""
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        model_name = os.environ["MLFLOW_MODEL_NAME"]
        version = os.environ["MODEL_VERSION_PIN"]
        embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "cointegrated/rubert-tiny2")
        embedding_revision = os.environ.get("EMBEDDING_REVISION") or None

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version(name=model_name, version=version)
        run_id = model_version.run_id
        data_version = client.get_run(run_id).data.tags.get("data_version", "unknown")
        local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")

        encoder = Encoder(
            model_name=embedding_model_name,
            revision=embedding_revision,
            device="cpu",
        )
        return cls(
            version=version,
            data_version=data_version,
            booster_path=Path(local_dir) / "lgbm.txt",
            encoder=encoder,
        )

    def load(self) -> None:
        self._booster = lgb.Booster(model_file=str(self._booster_path))

    def _predict_one(self, text: str) -> tuple[str, float]:
        assert self._booster is not None, "predictor not loaded"
        emb: np.ndarray = self._encoder.encode([text])
        scores: np.ndarray = np.asarray(self._booster.predict(emb))
        prob_jb = float(scores[0])
        if prob_jb >= 0.5:
            return "jailbreak", prob_jb
        return "benign", 1.0 - prob_jb


def main() -> None:
    """Container entrypoint: load from MLflow + serve via uvicorn."""
    import uvicorn

    from ru_jailbreak_guard.serve.predictor import build_app

    predictor = LgbmEmbPredictor.from_mlflow()
    app = build_app(predictor=predictor)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
