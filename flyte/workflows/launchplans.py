"""LaunchPlans for cron-scheduled retraining and drift detection."""

from __future__ import annotations

from flytekit import CronSchedule, LaunchPlan

from flyte.workflows.drift import drift_detect
from flyte.workflows.pipelines import (
    cheap_train_pipeline,
    evaluate_and_promote,
    gpu_train_pipeline,
)

# data_version is hard-coded into LaunchPlan defaults at registration time.
# `make register-workflows` updates the default; manual triggers override.
DEFAULT_DATA_VERSION = "set-via-make-register-workflows"


cheap_weekly = LaunchPlan.get_or_create(
    name="cheap_weekly",
    workflow=cheap_train_pipeline,  # ty: ignore[invalid-argument-type]
    default_inputs={"data_version": DEFAULT_DATA_VERSION},
    schedule=CronSchedule(schedule="0 3 * * 0"),  # Sundays 03:00 UTC
)


gpu_monthly = LaunchPlan.get_or_create(
    name="gpu_monthly",
    workflow=gpu_train_pipeline,  # ty: ignore[invalid-argument-type]
    default_inputs={"data_version": DEFAULT_DATA_VERSION},
    schedule=CronSchedule(schedule="0 4 1 * *"),  # 1st of month 04:00 UTC
)


# Manual invocation via `pyflyte run`; no standalone schedule.
evaluate_promote_lp = LaunchPlan.get_or_create(
    name="evaluate_and_promote_default",
    workflow=evaluate_and_promote,  # ty: ignore[invalid-argument-type]
    default_inputs={"data_version": DEFAULT_DATA_VERSION},
)


# Per-family drift detection (weekly, staggered hours).
drift_weekly_tfidf = LaunchPlan.get_or_create(
    name="drift_weekly_tfidf",
    workflow=drift_detect,  # ty: ignore[invalid-argument-type]
    default_inputs={"data_version": DEFAULT_DATA_VERSION, "family": "tfidf_logreg"},
    schedule=CronSchedule(schedule="0 6 * * 1"),  # Mon 06:00 UTC
)


drift_weekly_lgbm = LaunchPlan.get_or_create(
    name="drift_weekly_lgbm",
    workflow=drift_detect,  # ty: ignore[invalid-argument-type]
    default_inputs={"data_version": DEFAULT_DATA_VERSION, "family": "lgbm_emb"},
    schedule=CronSchedule(schedule="0 7 * * 1"),
)


drift_weekly_rubert = LaunchPlan.get_or_create(
    name="drift_weekly_rubert",
    workflow=drift_detect,  # ty: ignore[invalid-argument-type]
    default_inputs={"data_version": DEFAULT_DATA_VERSION, "family": "rubert_ft"},
    schedule=CronSchedule(schedule="0 8 * * 1"),
)
