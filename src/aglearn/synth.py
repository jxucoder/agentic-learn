"""Synthetic datasets for Kaggle-style benchmarking.

Public datasets (Titanic, California Housing) are likely in LLM training data.
These generators produce fresh tabular tasks with realistic competition setup:
- train/test split with hidden test labels
- sample_submission template
- distribution shift between train and test
- missingness, outliers, high-cardinality categoricals, and noisy features
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SyntheticTask:
    """Describes a synthetic data generation task."""

    name: str
    task_type: str  # "classification" or "regression"
    metric: str
    n_samples: int = 2000
    noise_level: float = 0.1
    missing_frac: float = 0.05
    train_frac: float = 0.8
    seed: int = 42


def generate(task: SyntheticTask, output_dir: str = "data") -> str:
    """Generate a Kaggle-style synthetic dataset.

    Files written:
    - ``synth_<name>.csv``: full data (train + test with labels)
    - ``synth_<name>_train.csv``: train split with labels (returned path)
    - ``synth_<name>_test.csv``: test split without labels
    - ``synth_<name>_solution.csv``: hidden test labels for evaluation
    - ``synth_<name>_sample_submission.csv``: submission template
    - ``synth_<name>_meta.json``: generation metadata
    """
    _validate_task(task)

    rng = np.random.default_rng(task.seed)
    n = task.n_samples
    n_train = int(n * task.train_frac)
    if n_train <= 0 or n_train >= n:
        raise ValueError(
            f"train split must have at least 1 and at most n-1 rows; got n={n}, "
            f"train_frac={task.train_frac}"
        )

    split_index = np.arange(n)
    is_train = split_index < n_train
    time_idx = split_index.astype(float)
    time_norm = time_idx / max(float(n - 1), 1.0)

    # Temporal backbone (kept as regular features to mimic production logs)
    event_date = pd.Timestamp("2021-01-01") + pd.to_timedelta(split_index, unit="D")

    # Numeric features with realistic distributions and mild test drift.
    income = rng.lognormal(
        mean=10.2 + 0.12 * time_norm + 0.10 * (~is_train), sigma=0.45, size=n
    )
    age = rng.normal(loc=39 + 1.5 * time_norm, scale=11.5, size=n).clip(18, 78)
    tenure_months = rng.gamma(shape=2.2 + 0.5 * time_norm, scale=10.0, size=n).clip(
        0, 180
    )
    sessions_30d = rng.poisson(
        lam=5.5 + 1.8 * np.sin(2 * np.pi * time_idx / 30) + 0.8 * (~is_train),
        size=n,
    ).astype(float)
    avg_order_value = rng.lognormal(
        mean=3.4 + 0.10 * time_norm + 0.05 * (~is_train), sigma=0.52, size=n
    )
    discount_rate = rng.beta(a=2.0, b=8.0, size=n)
    support_wait_minutes = rng.gamma(shape=2.4, scale=6.5 + 0.7 * (~is_train), size=n)
    return_rate = rng.beta(a=1.8, b=11.0, size=n)
    complaints_90d = rng.poisson(
        lam=0.35 + 1.6 * return_rate + 0.01 * support_wait_minutes, size=n
    ).astype(float)
    marketing_touches_30d = rng.poisson(
        lam=2.0 + 0.35 * np.log1p(sessions_30d) + 0.25 * (~is_train),
        size=n,
    ).astype(float)

    # Categorical features (including a high-cardinality field).
    region = np.where(
        is_train,
        rng.choice(
            ["north", "south", "east", "west"],
            size=n,
            p=[0.28, 0.25, 0.25, 0.22],
        ),
        rng.choice(
            ["north", "south", "east", "west"],
            size=n,
            p=[0.24, 0.30, 0.23, 0.23],
        ),
    )
    segment = rng.choice(
        ["budget", "core", "premium", "enterprise"],
        size=n,
        p=[0.34, 0.36, 0.22, 0.08],
    )
    plan_tier = rng.choice(["basic", "plus", "pro"], size=n, p=[0.46, 0.34, 0.20])
    acquisition_channel = np.where(
        is_train,
        rng.choice(
            ["organic", "paid_search", "affiliate", "referral", "direct"],
            size=n,
            p=[0.26, 0.25, 0.16, 0.12, 0.21],
        ),
        rng.choice(
            ["organic", "paid_search", "affiliate", "referral", "direct"],
            size=n,
            p=[0.21, 0.30, 0.13, 0.09, 0.27],
        ),
    )
    city_pool = [f"city_{i:03d}" for i in range(120)]
    city_code = rng.choice(city_pool, size=n)
    campaign_pool = [f"cmp_{i:02d}" for i in range(30)]
    campaign_id = rng.choice(campaign_pool, size=n)
    customer_id = np.array([f"C{c:06d}" for c in rng.integers(10_000, 16_000, size=n)])

    # Distractor columns (easy for overfitting, weak/no stable signal).
    aux_metric_01 = rng.normal(0, 1, size=n)
    aux_bucket_02 = rng.zipf(a=2.0, size=n).astype(float)

    # Hidden effects with train/test mismatch to simulate leaderboard surprises.
    city_effect_map = dict(zip(city_pool, rng.normal(0, 0.25, size=len(city_pool))))
    city_effect = np.array([city_effect_map[c] for c in city_code], dtype=float)

    campaign_train_map = dict(
        zip(campaign_pool, rng.normal(0, 0.30, size=len(campaign_pool)))
    )
    campaign_test_map = dict(
        zip(campaign_pool, rng.normal(0, 0.30, size=len(campaign_pool)))
    )
    campaign_effect = np.array(
        [campaign_train_map[c] for c in campaign_id], dtype=float
    )
    campaign_effect[~is_train] = np.array(
        [campaign_test_map[c] for c in campaign_id[~is_train]]
    )
    campaign_quality_proxy = campaign_effect + rng.normal(0, 0.35, size=n)

    segment_effect = np.select(
        [
            segment == "enterprise",
            segment == "premium",
            segment == "core",
            segment == "budget",
        ],
        [0.65, 0.40, 0.18, -0.08],
        default=0.0,
    )
    plan_effect = np.select(
        [plan_tier == "pro", plan_tier == "plus", plan_tier == "basic"],
        [0.42, 0.20, -0.05],
        default=0.0,
    )
    channel_effect = np.select(
        [
            acquisition_channel == "organic",
            acquisition_channel == "direct",
            acquisition_channel == "referral",
            acquisition_channel == "paid_search",
            acquisition_channel == "affiliate",
        ],
        [0.22, 0.15, 0.08, -0.10, -0.14],
        default=0.0,
    )
    region_effect = np.select(
        [region == "west", region == "north", region == "east", region == "south"],
        [0.15, 0.08, 0.02, -0.04],
        default=0.0,
    )

    # Latent signal used for both regression and classification flavors.
    engagement = np.log1p(sessions_30d) * np.log1p(tenure_months + 1.0)
    risk = 0.06 * support_wait_minutes + 2.2 * return_rate + 0.015 * complaints_90d

    signal = (
        0.58 * np.log1p(income)
        + 0.24 * np.log1p(avg_order_value)
        + 0.30 * np.tanh(engagement - 1.8)
        - 0.50 * risk
        + 0.25 * segment_effect
        + 0.22 * plan_effect
        + 0.15 * channel_effect
        + 0.12 * region_effect
        + 0.20 * city_effect
        + 0.12 * campaign_quality_proxy
        + 0.09 * (plan_effect * np.log1p(marketing_touches_30d))
        + 0.07 * np.where(region == "west", np.log1p(sessions_30d), 0.0)
        + 0.10 * np.sin(2 * np.pi * time_idx / 30)
    )
    signal += rng.normal(0, task.noise_level, size=n)

    if task.task_type == "classification":
        threshold = np.quantile(signal[is_train], 0.65)
        target = (signal > threshold).astype(int)
        flip_prob = np.clip(0.015 + 0.12 * task.noise_level, 0.0, 0.15)
        flip_mask = rng.random(n) < flip_prob
        target[flip_mask] = 1 - target[flip_mask]
        if len(np.unique(target)) < 2:
            target = (signal > np.median(signal)).astype(int)
    else:
        noise_scale = 6 + task.noise_level * (
            8 + 0.015 * np.log1p(income) * 100 + 0.5 * support_wait_minutes
        )
        target = (
            70
            + 18 * signal
            + 3.5 * np.sqrt(np.clip(avg_order_value, a_min=0.0, a_max=None))
            + rng.normal(0, noise_scale, size=n)
        )
        outlier_mask = rng.random(n) < 0.02
        target[outlier_mask] += rng.normal(0, 55, size=outlier_mask.sum())

    df = pd.DataFrame(
        {
            "row_id": np.arange(1, n + 1, dtype=int),
            "event_date": event_date.strftime("%Y-%m-%d"),
            "customer_id": customer_id,
            "income": np.round(income, 2),
            "age": np.round(age, 1),
            "tenure_months": np.round(tenure_months, 1),
            "sessions_30d": sessions_30d.astype(int),
            "avg_order_value": np.round(avg_order_value, 2),
            "discount_rate": np.round(discount_rate, 4),
            "support_wait_minutes": np.round(support_wait_minutes, 2),
            "return_rate": np.round(return_rate, 4),
            "complaints_90d": complaints_90d.astype(int),
            "marketing_touches_30d": marketing_touches_30d.astype(int),
            "region": region,
            "segment": segment,
            "plan_tier": plan_tier,
            "acquisition_channel": acquisition_channel,
            "city_code": city_code,
            "campaign_id": campaign_id,
            "campaign_quality_proxy": np.round(campaign_quality_proxy, 4),
            "aux_metric_01": np.round(aux_metric_01, 4),
            "aux_bucket_02": aux_bucket_02.astype(int),
            "target": (
                target if task.task_type == "classification" else np.round(target, 4)
            ),
        }
    )

    _inject_missing_values(df, rng, task.missing_frac)

    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    test_features = test_df.drop(columns=["target"]).copy()
    solution = test_df[["row_id", "target"]].copy()

    if task.task_type == "classification":
        sample_target = np.zeros(len(test_features), dtype=int)
    else:
        sample_target = np.repeat(
            np.round(train_df["target"].mean(), 4), len(test_features)
        )
    sample_submission = pd.DataFrame(
        {"row_id": test_features["row_id"].astype(int), "target": sample_target}
    )

    os.makedirs(output_dir, exist_ok=True)
    base_name = f"synth_{task.name}"
    full_path = os.path.join(output_dir, f"{base_name}.csv")
    train_path = os.path.join(output_dir, f"{base_name}_train.csv")
    test_path = os.path.join(output_dir, f"{base_name}_test.csv")
    solution_path = os.path.join(output_dir, f"{base_name}_solution.csv")
    sample_path = os.path.join(output_dir, f"{base_name}_sample_submission.csv")
    meta_path = os.path.join(output_dir, f"{base_name}_meta.json")

    df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    solution.to_csv(solution_path, index=False)
    sample_submission.to_csv(sample_path, index=False)

    meta: dict[str, Any] = {
        "name": task.name,
        "task_type": task.task_type,
        "metric": task.metric,
        "n_samples": n,
        "train_samples": len(train_df),
        "test_samples": len(test_features),
        "train_frac": task.train_frac,
        "noise_level": task.noise_level,
        "missing_frac": task.missing_frac,
        "seed": task.seed,
        "kaggle_style": True,
        "files": {
            "full": os.path.abspath(full_path),
            "train": os.path.abspath(train_path),
            "test": os.path.abspath(test_path),
            "solution": os.path.abspath(solution_path),
            "sample_submission": os.path.abspath(sample_path),
        },
        "feature_columns": [c for c in train_df.columns if c != "target"],
        "notes": [
            "Kaggle-style: no ground-truth feature importance is exposed.",
            "Includes distribution shift, high-cardinality categories, and missingness patterns.",
        ],
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return os.path.abspath(train_path)


def _validate_task(task: SyntheticTask) -> None:
    if task.task_type not in {"classification", "regression"}:
        raise ValueError(f"unsupported task_type: {task.task_type}")
    if task.n_samples < 50:
        raise ValueError("n_samples must be >= 50")
    if not 0.5 <= task.train_frac < 1.0:
        raise ValueError("train_frac must be in [0.5, 1.0)")
    if task.missing_frac < 0:
        raise ValueError("missing_frac must be >= 0")


def _inject_missing_values(
    df: pd.DataFrame, rng: np.random.Generator, missing_frac: float
) -> None:
    if missing_frac <= 0:
        return

    n = len(df)
    mnar_cols = ["income", "avg_order_value", "support_wait_minutes"]
    for col in mnar_cols:
        vals = df[col].to_numpy(dtype=float)
        scale = np.nanmax(vals) - np.nanmin(vals)
        if scale <= 0:
            probs = np.full(n, missing_frac)
        else:
            norm = (vals - np.nanmin(vals)) / scale
            probs = np.clip(missing_frac * (0.35 + 1.25 * norm), 0.0, 0.90)
        mask = rng.random(n) < probs
        df.loc[mask, col] = np.nan

    mcar_cols = ["age", "sessions_30d", "discount_rate", "complaints_90d"]
    for col in mcar_cols:
        mask = rng.random(n) < missing_frac
        df.loc[mask, col] = np.nan

    cat_cols = ["segment", "acquisition_channel", "city_code"]
    cat_missing = np.clip(missing_frac * 0.5, 0.0, 0.20)
    for col in cat_cols:
        mask = rng.random(n) < cat_missing
        df.loc[mask, col] = np.nan
