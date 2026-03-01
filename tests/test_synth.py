"""Tests for the synthetic dataset generator."""

import os
import json
import tempfile

import numpy as np
import pandas as pd

from aglearn.synth import SyntheticTask, generate


def test_generate_classification():
    """Classification dataset has correct shape and binary target."""
    with tempfile.TemporaryDirectory() as tmp:
        task = SyntheticTask(
            name="test_clf", task_type="classification", metric="f1",
            n_samples=200, seed=99,
        )
        path = generate(task, output_dir=tmp)
        df = pd.read_csv(path)

        assert df.shape[0] == 200
        assert "target" in df.columns
        assert set(df["target"].dropna().unique()) == {0, 1}
        assert "income" in df.columns
        assert "region" in df.columns
        assert "noise_feature_a" in df.columns

        # Metadata file should exist
        meta_path = os.path.join(tmp, "synth_test_clf_meta.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["task_type"] == "classification"
        assert "noise_feature_a" in meta["noise_features"]


def test_generate_regression():
    """Regression dataset has continuous target."""
    with tempfile.TemporaryDirectory() as tmp:
        task = SyntheticTask(
            name="test_reg", task_type="regression", metric="r2",
            n_samples=300, noise_level=0.2, seed=77,
        )
        path = generate(task, output_dir=tmp)
        df = pd.read_csv(path)

        assert df.shape[0] == 300
        assert "target" in df.columns
        # Regression target should have more than 2 unique values
        assert df["target"].nunique() > 10


def test_different_seeds_produce_different_data():
    """Different seeds should produce different datasets."""
    with tempfile.TemporaryDirectory() as tmp:
        task1 = SyntheticTask(name="s1", task_type="classification", metric="f1", seed=1)
        task2 = SyntheticTask(name="s2", task_type="classification", metric="f1", seed=2)
        path1 = generate(task1, output_dir=tmp)
        path2 = generate(task2, output_dir=tmp)
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        # Income values should differ between seeds
        assert not np.allclose(df1["income"].values, df2["income"].values)


def test_missing_values_injected():
    """Missing values should be present when missing_frac > 0."""
    with tempfile.TemporaryDirectory() as tmp:
        task = SyntheticTask(
            name="miss", task_type="regression", metric="r2",
            n_samples=1000, missing_frac=0.1, seed=42,
        )
        path = generate(task, output_dir=tmp)
        df = pd.read_csv(path)

        # At least some missing values in numeric columns
        assert df["income"].isna().sum() > 0
        assert df["age"].isna().sum() > 0
