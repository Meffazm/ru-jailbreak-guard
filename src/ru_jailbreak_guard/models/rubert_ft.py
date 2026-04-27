"""Fine-tuned ruBERT-tiny2 trainer with MLflow logging.

Uses HF Trainer to fine-tune the binary head on Phase 1 splits. Logs to MLflow
and registers as `ru-jailbreak-rubert-ft`. Designed to run on local CPU
(slow but feasible for ruBERT-tiny2 + ~2K rows) or GPU (DataSphere).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from ru_jailbreak_guard.evaluate.metrics import (
    binary_metrics,
    confusion_matrix_png,
    latency_benchmark,
    per_source_f1,
)


@dataclass
class TrainConfig:
    train_path: Path
    val_path: Path
    test_path: Path
    artifacts_dir: Path
    backbone: str = "cointegrated/rubert-tiny2"
    revision: str | None = None
    max_length: int = 256
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    device: str = "cpu"
    mlflow_tracking_uri: str | None = None
    experiment_name: str = "rubert_ft"
    data_version: str = "unknown"
    register_as: str = "ru-jailbreak-rubert-ft"


class _TextDataset(Dataset):
    """Tokenizes (text, label) pairs lazily for HF Trainer."""

    def __init__(
        self,
        *,
        texts: list[str],
        labels: np.ndarray,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):  # ty: ignore[invalid-method-override]
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def _load_split(path: Path) -> tuple[list[str], np.ndarray, pl.DataFrame]:
    df = pl.read_parquet(path)
    return df["text"].to_list(), df["label"].to_numpy(), df


def _predict(
    model: Any, tokenizer: PreTrainedTokenizerBase, texts: list[str], cfg: TrainConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Run model over `texts`, return (preds, scores)."""
    model.eval()
    all_scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), cfg.batch_size):
            batch = texts[i : i + cfg.batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            ).to(cfg.device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_scores.extend(probs.cpu().numpy().tolist())
    scores = np.array(all_scores)
    preds = (scores >= 0.5).astype(int)
    return preds, scores


def train_rubert_ft(*, cfg: TrainConfig) -> dict[str, Any]:
    """Fine-tune ruBERT-tiny2 binary classifier. Logs to MLflow if URI set."""
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    train_texts, y_train, _ = _load_split(cfg.train_path)
    val_texts, y_val, _ = _load_split(cfg.val_path)
    test_texts, y_test, test_df = _load_split(cfg.test_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, revision=cfg.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.backbone,
        revision=cfg.revision,
        num_labels=2,
    ).to(cfg.device)

    train_ds = _TextDataset(
        texts=train_texts, labels=y_train, tokenizer=tokenizer, max_length=cfg.max_length
    )
    val_ds = _TextDataset(
        texts=val_texts, labels=y_val, tokenizer=tokenizer, max_length=cfg.max_length
    )

    args = TrainingArguments(
        output_dir=str(cfg.artifacts_dir / "hf_trainer"),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=cfg.seed,
        report_to=[],
        disable_tqdm=False,
        use_cpu=(cfg.device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()  # ty: ignore[unresolved-attribute]

    val_pred, val_score = _predict(model, tokenizer, val_texts, cfg)
    test_pred, test_score = _predict(model, tokenizer, test_texts, cfg)

    val_metrics = binary_metrics(y_true=y_val, y_pred=val_pred, y_score=val_score)
    test_metrics = binary_metrics(y_true=y_test, y_pred=test_pred, y_score=test_score)
    test_per_source = per_source_f1(df=test_df, y_pred=test_pred)

    cm_path = cfg.artifacts_dir / "confusion_matrix.png"
    confusion_matrix_png(y_true=y_test, y_pred=test_pred, out_path=cm_path, title="ruBERT-ft test")

    model_dir = cfg.artifacts_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    def _predict_batch(texts: list[str]) -> list[int]:
        preds, _ = _predict(model, tokenizer, texts, cfg)
        return [int(p) for p in preds]

    latency = latency_benchmark(
        predict_fn=_predict_batch,
        sample_texts=test_texts[: min(50, len(test_texts))],
        batch_size=1,
    )

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_params(
                {
                    "model_family": "rubert_ft",
                    "backbone": cfg.backbone,
                    "revision": cfg.revision or "latest",
                    "max_length": cfg.max_length,
                    "num_epochs": cfg.num_epochs,
                    "batch_size": cfg.batch_size,
                    "learning_rate": cfg.learning_rate,
                    "weight_decay": cfg.weight_decay,
                    "warmup_ratio": cfg.warmup_ratio,
                    "seed": cfg.seed,
                    "device": cfg.device,
                }
            )
            mlflow.set_tags(
                {
                    "data_version": cfg.data_version,
                    "git_sha": os.environ.get("GIT_SHA", "unknown"),
                    "git_branch": os.environ.get("GIT_BRANCH", "unknown"),
                    "model_family": "rubert_ft",
                }
            )
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            for k, v in latency.items():
                mlflow.log_metric(f"latency_{k}", v)
            for src, f1 in test_per_source.items():
                mlflow.log_metric(f"test_f1_source_{src}", f1)
            mlflow.log_artifact(str(cm_path))
            # The whole model dir gets logged + registered as a transformers
            # flavor; predictors use mlflow.transformers.load_model() if they
            # want, but our predictor goes via from_pretrained() on the
            # downloaded artifact path — both work.
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            client = mlflow.tracking.MlflowClient()
            with contextlib.suppress(mlflow.exceptions.RestException):
                client.create_registered_model(cfg.register_as)
            mv = client.create_model_version(
                name=cfg.register_as,
                source=f"{run.info.artifact_uri}/model",
                run_id=run.info.run_id,
            )
            run_id = run.info.run_id
            version = mv.version
    else:
        run_id = "no-mlflow"
        version = "no-mlflow"

    return {
        "run_id": run_id,
        "version": version,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_per_source_f1": test_per_source,
        "latency": latency,
        "model_dir": str(model_dir),
        "confusion_matrix_path": str(cm_path),
    }


def _params_from_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        return yaml.safe_load(fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ruBERT-tiny2")
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.parquet"))
    parser.add_argument("--val", type=Path, default=Path("data/splits/val.parquet"))
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.parquet"))
    parser.add_argument("--artifacts", type=Path, default=Path("data/artifacts/rubert_ft"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--mlflow-uri", type=str, default=None)
    parser.add_argument("--data-version", type=str, default="unknown")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-epochs", type=int, default=None)
    args = parser.parse_args()

    params = _params_from_yaml(args.params)
    emb = params.get("embedding", {})
    rubert = params.get("model_rubert", {})
    mlflow_uri = args.mlflow_uri or params["mlflow"]["tracking_uri"]

    cfg = TrainConfig(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        artifacts_dir=args.artifacts,
        backbone=emb.get("model_name", "cointegrated/rubert-tiny2"),
        revision=emb.get("revision"),
        max_length=emb.get("max_length", 256),
        num_epochs=args.num_epochs or rubert.get("num_epochs", 3),
        batch_size=rubert.get("batch_size", 16),
        learning_rate=rubert.get("learning_rate", 5e-5),
        weight_decay=rubert.get("weight_decay", 0.01),
        warmup_ratio=rubert.get("warmup_ratio", 0.1),
        seed=params.get("seed", 42),
        device=args.device,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name=params["mlflow"].get("experiment_rubert", "rubert_ft"),
        data_version=args.data_version,
    )
    result = train_rubert_ft(cfg=cfg)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
