"""Tests for the synthetic dataset generator."""

import os
import json
import tempfile

import numpy as np
import pandas as pd

from aglearn.data.synth import SyntheticTask, generate


def test_generate_classification():
    with tempfile.TemporaryDirectory() as tmp:
        task = SyntheticTask(
            name="test_clf",
            task_type="classification",
            metric="f1",
            n_samples=200,
            seed=99,
        )
        path = generate(task, output_dir=tmp)
        train_df = pd.read_csv(path)
        full_df = pd.read_csv(os.path.join(tmp, "synth_test_clf.csv"))
        test_df = pd.read_csv(os.path.join(tmp, "synth_test_clf_test.csv"))
        solution_df = pd.read_csv(os.path.join(tmp, "synth_test_clf_solution.csv"))
        sample_df = pd.read_csv(
            os.path.join(tmp, "synth_test_clf_sample_submission.csv")
        )

        assert train_df.shape[0] == int(200 * task.train_frac)
        assert full_df.shape[0] == 200
        assert "target" in train_df.columns
        assert "target" not in test_df.columns
        assert set(train_df["target"].dropna().unique()) == {0, 1}
        assert set(solution_df.columns) == {"row_id", "target"}
        assert set(sample_df.columns) == {"row_id", "target"}
        assert sample_df.shape[0] == solution_df.shape[0]

        meta_path = os.path.join(tmp, "synth_test_clf_meta.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["kaggle_style"] is True
        assert meta["files"]["train"] == os.path.abspath(path)


def test_generate_regression():
    with tempfile.TemporaryDirectory() as tmp:
        task = SyntheticTask(
            name="test_reg", task_type="regression", metric="r2", n_samples=300, seed=77
        )
        path = generate(task, output_dir=tmp)
        df = pd.read_csv(path)
        assert df.shape[0] == int(300 * task.train_frac)
        assert df["target"].nunique() > 10


def test_different_seeds_produce_different_data():
    with tempfile.TemporaryDirectory() as tmp:
        task1 = SyntheticTask(
            name="s1", task_type="classification", metric="f1", seed=1
        )
        task2 = SyntheticTask(
            name="s2", task_type="classification", metric="f1", seed=2
        )
        df1 = pd.read_csv(generate(task1, output_dir=tmp))
        df2 = pd.read_csv(generate(task2, output_dir=tmp))
        assert not np.allclose(df1["income"].values, df2["income"].values)


def test_missing_values_injected():
    with tempfile.TemporaryDirectory() as tmp:
        task = SyntheticTask(
            name="miss",
            task_type="regression",
            metric="r2",
            n_samples=1000,
            missing_frac=0.1,
            seed=42,
        )
        df = pd.read_csv(generate(task, output_dir=tmp))
        assert df["income"].isna().sum() > 0
