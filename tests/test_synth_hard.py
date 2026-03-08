"""Tests for the harder synthetic dataset generators."""

import json
import os
import tempfile

import numpy as np
import pandas as pd

from aglearn.data.synth_hard import (
    HardSyntheticTask,
    generate_high_dim,
    generate_multiclass,
    generate_temporal_regression,
)


def test_generate_multiclass():
    with tempfile.TemporaryDirectory() as tmp:
        task = HardSyntheticTask(
            name="test_mc",
            task_type="multiclass",
            metric="f1_macro",
            n_samples=500,
            seed=99,
        )
        path = generate_multiclass(task, output_dir=tmp)
        df = pd.read_csv(path)
        assert df.shape[0] == 500
        assert set(df["target"].dropna().unique()) == {0, 1, 2, 3, 4}
        assert not any(c.startswith("noise_") for c in df.columns)

        meta_path = os.path.join(tmp, "synth_test_mc_meta.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["n_classes"] == 5
        assert "informative_features" not in meta
        assert "noise_features" not in meta
        assert "feature_columns" in meta


def test_multiclass_imbalance():
    with tempfile.TemporaryDirectory() as tmp:
        task = HardSyntheticTask(
            name="test_mc_imb",
            task_type="multiclass",
            metric="f1_macro",
            n_samples=2000,
            seed=42,
        )
        df = pd.read_csv(generate_multiclass(task, output_dir=tmp))
        counts = df["target"].value_counts(normalize=True).sort_index()
        # Minority class (0) should be roughly 5%, majority (4) roughly 40%
        assert counts.iloc[counts.values.argmin()] < 0.10
        assert counts.iloc[counts.values.argmax()] > 0.30


def test_multiclass_mnar_missing():
    with tempfile.TemporaryDirectory() as tmp:
        task = HardSyntheticTask(
            name="test_mc_miss",
            task_type="multiclass",
            metric="f1_macro",
            n_samples=2000,
            missing_frac=0.15,
            seed=42,
        )
        df = pd.read_csv(generate_multiclass(task, output_dir=tmp))
        # MNAR columns should have missing values
        assert df["salary"].isna().sum() > 0
        assert df["performance_score"].isna().sum() > 0


def test_generate_temporal_regression():
    with tempfile.TemporaryDirectory() as tmp:
        task = HardSyntheticTask(
            name="test_temp",
            task_type="temporal_regression",
            metric="r2",
            n_samples=500,
            seed=77,
        )
        path = generate_temporal_regression(task, output_dir=tmp)
        df = pd.read_csv(path)
        assert df.shape[0] == 500
        assert df["target"].nunique() > 10
        assert not any(c.startswith("noise_") for c in df.columns)

        meta_path = os.path.join(tmp, "synth_test_temp_meta.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["n_stores"] == 50
        assert "informative_features" not in meta
        assert "noise_features" not in meta
        assert "feature_columns" in meta


def test_temporal_has_outliers():
    with tempfile.TemporaryDirectory() as tmp:
        task = HardSyntheticTask(
            name="test_temp_out",
            task_type="temporal_regression",
            metric="r2",
            n_samples=2000,
            seed=42,
        )
        df = pd.read_csv(generate_temporal_regression(task, output_dir=tmp))
        target = df["target"]
        q1, q3 = target.quantile(0.25), target.quantile(0.75)
        iqr = q3 - q1
        outliers = ((target < q1 - 3 * iqr) | (target > q3 + 3 * iqr)).sum()
        # Should have noticeable outliers due to 5% contamination
        assert outliers > 0


def test_temporal_high_cardinality():
    with tempfile.TemporaryDirectory() as tmp:
        task = HardSyntheticTask(
            name="test_temp_hc",
            task_type="temporal_regression",
            metric="r2",
            n_samples=2000,
            seed=42,
        )
        df = pd.read_csv(generate_temporal_regression(task, output_dir=tmp))
        assert df["store_id"].nunique() == 50


def test_generate_high_dim():
    with tempfile.TemporaryDirectory() as tmp:
        task = HardSyntheticTask(
            name="test_hd", task_type="high_dim", metric="f1", n_samples=500, seed=55
        )
        path = generate_high_dim(task, output_dir=tmp)
        df = pd.read_csv(path)
        assert df.shape[0] == 500
        # 50 features + 1 target column
        assert df.shape[1] == 51
        assert set(df["target"].dropna().unique()) == {0, 1}

        meta_path = os.path.join(tmp, "synth_test_hd_meta.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["n_features"] == 50
        assert len(meta["feature_columns"]) == 50
        assert "informative_features" not in meta
        assert "noise_features" not in meta


def test_high_dim_feature_count():
    with tempfile.TemporaryDirectory() as tmp:
        task = HardSyntheticTask(
            name="test_hd_fc", task_type="high_dim", metric="f1", n_samples=200, seed=42
        )
        df = pd.read_csv(generate_high_dim(task, output_dir=tmp))
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        assert len(feat_cols) == 50


def test_different_seeds_produce_different_hard_data():
    with tempfile.TemporaryDirectory() as tmp:
        task1 = HardSyntheticTask(
            name="s1", task_type="multiclass", metric="f1_macro", seed=1
        )
        task2 = HardSyntheticTask(
            name="s2", task_type="multiclass", metric="f1_macro", seed=2
        )
        df1 = pd.read_csv(generate_multiclass(task1, output_dir=tmp))
        df2 = pd.read_csv(generate_multiclass(task2, output_dir=tmp))
        assert not np.allclose(df1["salary"].values, df2["salary"].values)
