"""Synthetic dataset generator for leak-free benchmarking.

Public datasets (Titanic, California Housing) are in LLM training data.
Any "improvement" could be memorization, not genuine feature engineering.

This module generates fresh tabular datasets with:
- Known ground truth (verifiable)
- Non-obvious feature interactions (rewards real feature engineering)
- Configurable noise, missing values, and categoricals
- Unique per seed (LLM has never seen them)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

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
    seed: int = 42


def generate(task: SyntheticTask, output_dir: str = "data") -> str:
    """Generate a synthetic dataset and return the CSV path.

    The ground truth function is a non-linear combination of features
    with interactions, so naive linear models score low but feature
    engineering + nonlinear models can reach high scores.
    """
    rng = np.random.default_rng(task.seed)
    n = task.n_samples

    # ── Raw features (some informative, some noisy) ──────────────
    income = rng.lognormal(mean=3.5, sigma=0.8, size=n)
    age = rng.normal(loc=40, scale=12, size=n).clip(18, 80)
    hours_worked = rng.uniform(10, 60, size=n)
    distance_km = rng.exponential(scale=15, size=n)

    regions = rng.choice(["north", "south", "east", "west", "central"], size=n)
    education = rng.choice(
        ["high_school", "bachelors", "masters", "phd"],
        size=n,
        p=[0.35, 0.35, 0.20, 0.10],
    )

    satisfaction = rng.integers(1, 6, size=n)  # 1-5 scale

    # Pure noise features (should be ignored by a good model)
    noise_a = rng.standard_normal(size=n)
    noise_b = rng.integers(0, 100, size=n)

    # ── Ground truth function ────────────────────────────────────
    edu_multiplier = np.where(
        education == "phd",
        1.5,
        np.where(
            education == "masters", 1.2, np.where(education == "bachelors", 1.0, 0.7)
        ),
    )
    region_offset = np.where(
        regions == "central",
        0.3,
        np.where(
            regions == "north",
            0.1,
            np.where(regions == "south", -0.1, np.where(regions == "east", 0.0, -0.2)),
        ),
    )
    age_effect = -0.002 * (age - 45) ** 2 + 1.0
    commute_ratio = np.log1p(hours_worked) / np.log1p(distance_km + 1)
    satisfaction_boost = np.where(satisfaction > 3, 0.5, 0.0)

    signal = (
        0.4 * np.log1p(income) * edu_multiplier
        + 0.2 * age_effect
        + 0.15 * commute_ratio
        + 0.1 * satisfaction_boost
        + region_offset
    )

    noise = rng.normal(0, task.noise_level, size=n)
    y_continuous = signal + noise

    if task.task_type == "classification":
        threshold = np.median(y_continuous)
        y = (y_continuous > threshold).astype(int)
    else:
        y = y_continuous

    target_col = "target"

    # ── Assemble DataFrame ───────────────────────────────────────
    df = pd.DataFrame(
        {
            "income": np.round(income, 2),
            "age": np.round(age, 1),
            "hours_worked": np.round(hours_worked, 1),
            "distance_km": np.round(distance_km, 2),
            "region": regions,
            "education": education,
            "satisfaction": satisfaction,
            "noise_feature_a": np.round(noise_a, 3),
            "noise_feature_b": noise_b,
            target_col: y if task.task_type == "classification" else np.round(y, 4),
        }
    )

    # ── Inject missing values ────────────────────────────────────
    if task.missing_frac > 0:
        cols_to_corrupt = [
            "income",
            "age",
            "hours_worked",
            "distance_km",
            "satisfaction",
        ]
        for col in cols_to_corrupt:
            mask = rng.random(n) < task.missing_frac
            df.loc[mask, col] = np.nan

    # ── Save ─────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"synth_{task.name}.csv")
    df.to_csv(path, index=False)

    meta = {
        "name": task.name,
        "task_type": task.task_type,
        "metric": task.metric,
        "n_samples": n,
        "noise_level": task.noise_level,
        "missing_frac": task.missing_frac,
        "seed": task.seed,
        "informative_features": [
            "income",
            "age",
            "hours_worked",
            "distance_km",
            "region",
            "education",
            "satisfaction",
        ],
        "noise_features": ["noise_feature_a", "noise_feature_b"],
        "interactions": [
            "income × education (multiplicative)",
            "age quadratic (peaks at 45)",
            "log(hours_worked) / log(distance_km) ratio",
            "satisfaction threshold at > 3",
            "region shifts baseline",
        ],
    }
    meta_path = os.path.join(output_dir, f"synth_{task.name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return os.path.abspath(path)
