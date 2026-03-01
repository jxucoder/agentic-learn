"""Harder synthetic benchmarks that test advanced ML engineering skills.

These generators create datasets with challenges beyond the basic synth module:
- Multi-class with class imbalance
- Temporal structure with high-cardinality categoricals
- High-dimensional feature selection (many noise features)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HardSyntheticTask:
    """Describes a hard synthetic data generation task."""

    name: str
    task_type: str  # "multiclass", "temporal_regression", or "high_dim"
    metric: str
    n_samples: int = 4000
    noise_level: float = 0.2
    missing_frac: float = 0.10
    seed: int = 42


def generate_multiclass(task: HardSyntheticTask, output_dir: str = "data") -> str:
    """Generate a multi-class classification dataset with class imbalance.

    Challenges:
    - 5 imbalanced classes (40%, 25%, 18%, 12%, 5%)
    - 3-way feature interactions
    - Conditional effects (feature only matters within a subgroup)
    - Correlated noise features (correlated with inputs, not target)
    - 15% missing values with non-random missingness
    """
    rng = np.random.default_rng(task.seed)
    n = task.n_samples

    # ── Informative features ──────────────────────────────────────
    salary = rng.lognormal(mean=4.0, sigma=1.0, size=n)
    experience_years = rng.exponential(scale=8, size=n).clip(0, 40)
    weekly_hours = rng.normal(loc=40, scale=10, size=n).clip(5, 80)
    project_count = rng.poisson(lam=5, size=n)
    team_size = rng.integers(1, 50, size=n).astype(float)

    department = rng.choice(
        ["engineering", "sales", "marketing", "support", "research",
         "hr", "finance", "operations"],
        size=n, p=[0.25, 0.15, 0.12, 0.15, 0.10, 0.08, 0.08, 0.07],
    )
    seniority = rng.choice(
        ["junior", "mid", "senior", "lead", "director"],
        size=n, p=[0.30, 0.30, 0.20, 0.12, 0.08],
    )
    location = rng.choice(
        ["urban", "suburban", "rural", "remote"],
        size=n, p=[0.40, 0.30, 0.15, 0.15],
    )

    performance_score = rng.beta(a=5, b=2, size=n) * 10  # 0-10, right-skewed
    commute_minutes = rng.exponential(scale=25, size=n)

    # ── Noise features (correlated with inputs but not target) ────
    noise_salary_echo = salary * rng.normal(1.0, 0.3, size=n)  # correlated with salary
    noise_hours_echo = weekly_hours + rng.normal(0, 5, size=n)  # correlated with hours
    noise_random_a = rng.standard_normal(size=n)
    noise_random_b = rng.uniform(0, 100, size=n)
    noise_random_c = rng.integers(0, 20, size=n).astype(float)

    # ── Ground truth: complex multi-class signal ──────────────────
    dept_code = np.select(
        [department == "engineering", department == "research",
         department == "sales", department == "marketing",
         department == "finance", department == "hr",
         department == "support", department == "operations"],
        [1.5, 1.3, 0.8, 0.6, 1.0, 0.4, 0.3, 0.5],
        default=0.5,
    )
    seniority_code = np.select(
        [seniority == "director", seniority == "lead",
         seniority == "senior", seniority == "mid", seniority == "junior"],
        [2.0, 1.5, 1.0, 0.5, 0.0],
        default=0.5,
    )
    location_code = np.select(
        [location == "urban", location == "suburban",
         location == "rural", location == "remote"],
        [0.3, 0.0, -0.2, 0.1],
        default=0.0,
    )

    # 3-way interaction: salary * seniority * department
    interaction_3way = np.log1p(salary) * seniority_code * dept_code * 0.05

    # Conditional: project_count only matters for engineering/research
    conditional_projects = np.where(
        np.isin(department, ["engineering", "research"]),
        np.log1p(project_count) * 0.4,
        0.0,
    )

    # Non-linear: performance × experience with diminishing returns
    perf_exp = np.sqrt(performance_score) * np.log1p(experience_years) * 0.15

    # Team dynamics: productivity peaks at team_size ~ 7
    team_effect = -0.02 * (team_size - 7) ** 2 + 0.5

    # Commute penalty: only for non-remote workers
    commute_penalty = np.where(
        location != "remote",
        -0.005 * commute_minutes,
        0.0,
    )

    signal = (
        interaction_3way
        + conditional_projects
        + perf_exp
        + 0.3 * team_effect
        + commute_penalty
        + 0.2 * location_code
    )
    signal += rng.normal(0, task.noise_level, size=n)

    # Map to 5 imbalanced classes using quantiles
    class_boundaries = np.percentile(signal, [5, 17, 35, 60])
    y = np.digitize(signal, class_boundaries)  # classes 0-4

    # ── Assemble DataFrame ────────────────────────────────────────
    df = pd.DataFrame({
        "salary": np.round(salary, 2),
        "experience_years": np.round(experience_years, 1),
        "weekly_hours": np.round(weekly_hours, 1),
        "project_count": project_count,
        "team_size": team_size.astype(int),
        "department": department,
        "seniority": seniority,
        "location": location,
        "performance_score": np.round(performance_score, 2),
        "commute_minutes": np.round(commute_minutes, 1),
        "noise_salary_corr": np.round(noise_salary_echo, 2),
        "noise_hours_corr": np.round(noise_hours_echo, 1),
        "noise_random_a": np.round(noise_random_a, 3),
        "noise_random_b": np.round(noise_random_b, 2),
        "noise_random_c": noise_random_c.astype(int),
        "target": y,
    })

    # ── Non-random missing values (MNAR: high values more likely missing) ──
    if task.missing_frac > 0:
        mnar_cols = ["salary", "performance_score", "commute_minutes"]
        for col in mnar_cols:
            vals = df[col].values.astype(float)
            probs = task.missing_frac * (vals / np.nanmax(vals))
            mask = rng.random(n) < probs
            df.loc[mask, col] = np.nan

        mcar_cols = ["experience_years", "weekly_hours", "team_size"]
        for col in mcar_cols:
            mask = rng.random(n) < task.missing_frac
            df.loc[mask, col] = np.nan

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"synth_{task.name}.csv")
    df.to_csv(path, index=False)

    meta = {
        "name": task.name,
        "task_type": "multiclass",
        "metric": task.metric,
        "n_samples": n,
        "n_classes": 5,
        "class_distribution": "imbalanced (5%, 12%, 18%, 25%, 40%)",
        "noise_level": task.noise_level,
        "missing_frac": task.missing_frac,
        "missing_pattern": "MNAR for salary/performance/commute; MCAR for others",
        "seed": task.seed,
        "informative_features": [
            "salary", "experience_years", "weekly_hours", "project_count",
            "team_size", "department", "seniority", "location",
            "performance_score", "commute_minutes",
        ],
        "noise_features": [
            "noise_salary_corr", "noise_hours_corr",
            "noise_random_a", "noise_random_b", "noise_random_c",
        ],
        "interactions": [
            "salary × seniority × department (3-way multiplicative)",
            "project_count only matters for engineering/research (conditional)",
            "sqrt(performance) × log(experience) (diminishing returns)",
            "team_size quadratic (peaks at ~7)",
            "commute penalty only for non-remote (conditional)",
        ],
    }
    meta_path = os.path.join(output_dir, f"synth_{task.name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return os.path.abspath(path)


def generate_temporal_regression(task: HardSyntheticTask, output_dir: str = "data") -> str:
    """Generate a regression dataset with temporal structure and outliers.

    Challenges:
    - Time-ordered data with trend + seasonal components
    - High-cardinality categorical (50 store IDs)
    - Heteroscedastic noise (variance depends on features)
    - 5% outlier contamination in the target
    - Lagged feature effects
    """
    rng = np.random.default_rng(task.seed)
    n = task.n_samples

    # ── Temporal backbone ─────────────────────────────────────────
    day_index = np.arange(n)
    day_of_week = day_index % 7  # 0=Mon, 6=Sun
    month = ((day_index % 365) // 30).clip(0, 11)

    # ── Store-level features (high-cardinality categorical) ───────
    n_stores = 50
    store_id = rng.choice([f"store_{i:03d}" for i in range(n_stores)], size=n)
    # Each store has a hidden "quality" that affects the target
    store_quality = {f"store_{i:03d}": rng.normal(0, 0.5) for i in range(n_stores)}
    store_q = np.array([store_quality[s] for s in store_id])

    # ── Informative features ──────────────────────────────────────
    temperature = 15 + 15 * np.sin(2 * np.pi * month / 12) + rng.normal(0, 3, size=n)
    price = rng.lognormal(mean=2.5, sigma=0.6, size=n)
    promotion = rng.binomial(1, 0.2, size=n).astype(float)
    competitor_price = price * rng.uniform(0.8, 1.2, size=n)
    foot_traffic = rng.poisson(lam=200, size=n).astype(float)
    online_reviews = rng.beta(a=4, b=2, size=n) * 5  # 0-5 rating
    ad_spend = rng.exponential(scale=500, size=n)
    inventory_level = rng.integers(10, 500, size=n).astype(float)

    # ── Noise features ────────────────────────────────────────────
    noise_ts_a = np.cumsum(rng.normal(0, 0.1, size=n))  # random walk (tricky)
    noise_ts_b = rng.standard_normal(size=n)
    noise_cat = rng.choice(["A", "B", "C", "D", "E"], size=n)
    noise_uniform = rng.uniform(0, 1000, size=n)
    noise_seasonal = np.sin(2 * np.pi * day_index / 7) * rng.normal(1, 0.5, size=n)

    # ── Ground truth ──────────────────────────────────────────────
    # Trend component
    trend = 0.001 * day_index

    # Seasonal: weekday vs weekend effect
    weekend_boost = np.where(day_of_week >= 5, 1.5, 0.0)
    monthly_season = 0.8 * np.sin(2 * np.pi * month / 12)

    # Price elasticity (non-linear)
    price_effect = -0.3 * np.log1p(price) + 0.2 * np.log1p(competitor_price)
    price_ratio = np.log(competitor_price / price.clip(min=0.01))

    # Promotion effect depends on foot traffic (interaction)
    promo_effect = promotion * np.log1p(foot_traffic) * 0.02

    # Reviews × store quality interaction
    review_effect = online_reviews * store_q * 0.1

    # Ad spend with diminishing returns, modulated by temperature
    ad_effect = np.log1p(ad_spend) * (1 + 0.02 * temperature) * 0.03

    # Inventory: too little or too much hurts (U-shape)
    optimal_inv = 200
    inv_effect = -0.0001 * (inventory_level - optimal_inv) ** 2

    signal = (
        5.0  # base level
        + trend
        + weekend_boost
        + 0.5 * monthly_season
        + price_effect
        + 0.3 * price_ratio
        + promo_effect
        + review_effect
        + ad_effect
        + inv_effect
        + 0.5 * store_q
    )

    # Heteroscedastic noise: variance depends on price and foot_traffic
    noise_scale = task.noise_level * (1 + 0.3 * np.log1p(price) + 0.001 * foot_traffic)
    noise = rng.normal(0, noise_scale)
    y = signal + noise

    # Outlier contamination (5%)
    outlier_mask = rng.random(n) < 0.05
    y[outlier_mask] *= rng.choice([-1, 3, 5], size=outlier_mask.sum())

    # ── Assemble DataFrame ────────────────────────────────────────
    df = pd.DataFrame({
        "day_index": day_index,
        "day_of_week": day_of_week,
        "month": month,
        "store_id": store_id,
        "temperature": np.round(temperature, 1),
        "price": np.round(price, 2),
        "promotion": promotion.astype(int),
        "competitor_price": np.round(competitor_price, 2),
        "foot_traffic": foot_traffic.astype(int),
        "online_reviews": np.round(online_reviews, 2),
        "ad_spend": np.round(ad_spend, 2),
        "inventory_level": inventory_level.astype(int),
        "noise_trend": np.round(noise_ts_a, 3),
        "noise_random": np.round(noise_ts_b, 3),
        "noise_category": noise_cat,
        "noise_uniform": np.round(noise_uniform, 2),
        "noise_seasonal": np.round(noise_seasonal, 3),
        "target": np.round(y, 4),
    })

    # ── Missing values ────────────────────────────────────────────
    if task.missing_frac > 0:
        cols_to_corrupt = [
            "temperature", "price", "competitor_price",
            "foot_traffic", "online_reviews", "ad_spend", "inventory_level",
        ]
        for col in cols_to_corrupt:
            mask = rng.random(n) < task.missing_frac
            df.loc[mask, col] = np.nan

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"synth_{task.name}.csv")
    df.to_csv(path, index=False)

    meta = {
        "name": task.name,
        "task_type": "temporal_regression",
        "metric": task.metric,
        "n_samples": n,
        "noise_level": task.noise_level,
        "missing_frac": task.missing_frac,
        "seed": task.seed,
        "n_stores": n_stores,
        "outlier_frac": 0.05,
        "informative_features": [
            "day_index", "day_of_week", "month", "store_id",
            "temperature", "price", "promotion", "competitor_price",
            "foot_traffic", "online_reviews", "ad_spend", "inventory_level",
        ],
        "noise_features": [
            "noise_trend", "noise_random", "noise_category",
            "noise_uniform", "noise_seasonal",
        ],
        "interactions": [
            "trend + weekend + monthly seasonality (temporal decomposition)",
            "price elasticity: -log(price) + log(competitor_price)",
            "price_ratio = log(competitor_price / price)",
            "promotion × log(foot_traffic) (conditional boost)",
            "online_reviews × store_quality (hidden entity embedding)",
            "log(ad_spend) × temperature (seasonal ad effectiveness)",
            "inventory U-shape penalty (optimal ~200)",
            "heteroscedastic noise (harder targets have more variance)",
            "5% outlier contamination in target",
        ],
    }
    meta_path = os.path.join(output_dir, f"synth_{task.name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return os.path.abspath(path)


def generate_high_dim(task: HardSyntheticTask, output_dir: str = "data") -> str:
    """Generate a high-dimensional binary classification dataset.

    Challenges:
    - 50 features total, only 8 are informative
    - XOR-like interaction patterns (defeat linear models)
    - Multicollinear feature groups
    - Redundant features (near-duplicates of informative ones)
    - Signal buried in noise — requires aggressive feature selection
    """
    rng = np.random.default_rng(task.seed)
    n = task.n_samples

    # ── 8 Core informative features ───────────────────────────────
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.exponential(scale=2, size=n)
    x4 = rng.uniform(-3, 3, size=n)
    x5 = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]).astype(float)
    x6 = rng.beta(a=2, b=5, size=n) * 10
    x7 = rng.normal(3, 2, size=n)
    x8 = rng.lognormal(mean=1, sigma=0.5, size=n)

    # ── Ground truth: XOR-like + non-linear interactions ──────────
    # XOR pattern: x1 and x2 interact non-linearly
    xor_signal = np.sign(x1) * np.sign(x2)  # XOR-like

    # Polynomial interaction: x3 and x4
    poly_signal = 0.1 * x3 * x4 - 0.05 * x4 ** 2

    # Threshold interaction: x5 modulates x6
    threshold_signal = np.where(x5 > 0, np.log1p(x6), -np.sqrt(x6.clip(0)))

    # Ratio: x7 / x8
    ratio_signal = x7 / x8.clip(min=0.1)

    signal = (
        0.5 * xor_signal
        + 0.3 * poly_signal
        + 0.2 * threshold_signal
        + 0.15 * np.tanh(ratio_signal - 2)
    )
    signal += rng.normal(0, task.noise_level, size=n)

    threshold = np.median(signal)
    y = (signal > threshold).astype(int)

    # ── Build feature matrix: 8 informative + 42 noise ────────────
    features = {}

    # Informative features (with obfuscated names)
    informative = {
        "feat_03": x1, "feat_17": x2, "feat_22": x3, "feat_08": x4,
        "feat_31": x5, "feat_45": x6, "feat_12": x7, "feat_39": x8,
    }
    features.update({k: np.round(v, 4) for k, v in informative.items()})

    # Redundant features (near-copies of informative ones with noise)
    features["feat_04"] = np.round(x1 + rng.normal(0, 0.3, size=n), 4)
    features["feat_18"] = np.round(x2 * 1.1 + rng.normal(0, 0.2, size=n), 4)
    features["feat_40"] = np.round(np.log1p(x8) + rng.normal(0, 0.1, size=n), 4)

    # Multicollinear group: 5 features that are linear combos of each other
    base_a = rng.normal(0, 1, size=n)
    base_b = rng.normal(0, 1, size=n)
    features["feat_25"] = np.round(base_a, 4)
    features["feat_26"] = np.round(0.8 * base_a + 0.2 * base_b, 4)
    features["feat_27"] = np.round(0.5 * base_a + 0.5 * base_b, 4)
    features["feat_28"] = np.round(0.2 * base_a + 0.8 * base_b, 4)
    features["feat_29"] = np.round(base_b, 4)

    # Pure random noise features to fill up to 50 total
    n_remaining = 50 - len(features)
    for i in range(n_remaining):
        idx = i
        while f"feat_{idx:02d}" in features:
            idx += 1
        dist = rng.choice(["normal", "uniform", "exponential", "integers"])
        if dist == "normal":
            features[f"feat_{idx:02d}"] = np.round(rng.normal(0, rng.uniform(0.5, 5), size=n), 4)
        elif dist == "uniform":
            lo, hi = sorted(rng.uniform(-10, 10, size=2))
            features[f"feat_{idx:02d}"] = np.round(rng.uniform(lo, hi, size=n), 4)
        elif dist == "exponential":
            features[f"feat_{idx:02d}"] = np.round(rng.exponential(rng.uniform(0.5, 5), size=n), 4)
        else:
            features[f"feat_{idx:02d}"] = rng.integers(0, rng.integers(2, 50), size=n)

    # Sort columns by name for consistent ordering
    df = pd.DataFrame(dict(sorted(features.items())))
    df["target"] = y

    # ── Missing values ────────────────────────────────────────────
    if task.missing_frac > 0:
        all_feat_cols = [c for c in df.columns if c != "target"]
        # Missing values in a random subset of columns
        cols_to_corrupt = rng.choice(all_feat_cols, size=15, replace=False)
        for col in cols_to_corrupt:
            mask = rng.random(n) < task.missing_frac
            df.loc[mask, col] = np.nan

    # ── Save ──────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"synth_{task.name}.csv")
    df.to_csv(path, index=False)

    informative_names = sorted(informative.keys())
    redundant_names = ["feat_04", "feat_18", "feat_40"]
    multicollinear_names = ["feat_25", "feat_26", "feat_27", "feat_28", "feat_29"]
    noise_names = sorted(
        set(features.keys()) - set(informative_names) - set(redundant_names) - set(multicollinear_names)
    )

    meta = {
        "name": task.name,
        "task_type": "high_dim_classification",
        "metric": task.metric,
        "n_samples": n,
        "n_features": 50,
        "noise_level": task.noise_level,
        "missing_frac": task.missing_frac,
        "seed": task.seed,
        "informative_features": informative_names,
        "redundant_features": redundant_names,
        "multicollinear_group": multicollinear_names,
        "noise_features": noise_names,
        "interactions": [
            "sign(feat_03) × sign(feat_17) (XOR pattern — defeats linear models)",
            "feat_22 × feat_08 - feat_08² (polynomial interaction)",
            "feat_31 > 0 ? log(feat_45) : -sqrt(feat_45) (conditional threshold)",
            "tanh(feat_12 / feat_39 - 2) (ratio with saturation)",
            "feat_04 ≈ feat_03 + noise (redundant copy)",
            "feat_18 ≈ feat_17 × 1.1 + noise (redundant copy)",
            "feat_40 ≈ log(feat_39) + noise (redundant copy)",
            "feat_25..29 form multicollinear group (should be reduced)",
        ],
    }
    meta_path = os.path.join(output_dir, f"synth_{task.name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return os.path.abspath(path)
