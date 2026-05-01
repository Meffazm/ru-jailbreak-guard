"""Concrete TF-IDF + LogReg predictor.

Default load path: pull `model` artifact bundle from MLflow registry by
`MLFLOW_MODEL_NAME` + `MODEL_VERSION_PIN`. Falls back to `from_local` for tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import mlflow

from ru_jailbreak_guard.features.tfidf import TfidfWrapper
from ru_jailbreak_guard.serve.predictor import Predictor


class TfidfLogregPredictor(Predictor):
    family = "tfidf_logreg"

    def __init__(
        self,
        *,
        version: str,
        data_version: str,
        tfidf_path: Path,
        logreg_path: Path,
    ) -> None:
        super().__init__(version=version, data_version=data_version)
        self._tfidf_path = tfidf_path
        self._logreg_path = logreg_path
        self._tfidf: TfidfWrapper | None = None
        self._logreg: Any = None

    @classmethod
    def from_local(
        cls,
        *,
        artifacts_dir: Path,
        version: str,
        data_version: str,
    ) -> TfidfLogregPredictor:
        return cls(
            version=version,
            data_version=data_version,
            tfidf_path=artifacts_dir / "tfidf.joblib",
            logreg_path=artifacts_dir / "logreg.joblib",
        )

    @classmethod
    def from_mlflow(cls) -> TfidfLogregPredictor:
        """Build from MLflow registry. Reads env vars."""
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        model_name = os.environ["MLFLOW_MODEL_NAME"]
        version = os.environ["MODEL_VERSION_PIN"]
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version(name=model_name, version=version)
        run_id = model_version.run_id
        data_version = client.get_run(run_id).data.tags.get("data_version", "unknown")  # ty: ignore[invalid-argument-type]
        local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
        return cls(
            version=version,
            data_version=data_version,
            tfidf_path=Path(local_dir) / "tfidf.joblib",
            logreg_path=Path(local_dir) / "logreg.joblib",
        )

    def load(self) -> None:
        self._tfidf = TfidfWrapper.load(self._tfidf_path)
        self._logreg = joblib.load(self._logreg_path)

    def _predict_one(self, text: str) -> tuple[str, float]:
        assert self._tfidf is not None and self._logreg is not None, "predictor not loaded"
        x = self._tfidf.transform([text])
        prob_jb = float(self._logreg.predict_proba(x)[0, 1])
        if prob_jb >= 0.5:
            return "jailbreak", prob_jb
        return "benign", 1.0 - prob_jb


def main() -> None:
    """Container entrypoint: load from MLflow + serve via uvicorn."""
    import uvicorn

    from ru_jailbreak_guard.serve.predictor import build_app

    predictor = TfidfLogregPredictor.from_mlflow()
    app = build_app(predictor=predictor)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
