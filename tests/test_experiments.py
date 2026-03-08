"""Tests for experiment-side benchmark and arena helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aglearn_experiments.arena import ContestantSpec, build_submission_evaluator
from aglearn_experiments.benchmarks import generate_benchmark


def test_generate_benchmark_writes_manifest_and_challenge(tmp_path: Path):
    manifest = generate_benchmark(
        task_type="multiclass",
        seed=7,
        samples=200,
        noise=0.2,
        output_root=str(tmp_path),
        brief_generator=lambda prompt, model, cwd: (
            {
                "title": "Synthetic Employee Signals",
                "short_description": "Hard multiclass benchmark.",
                "scenario": "Predict a hidden workforce outcome.",
                "objective": "Maximize macro F1.",
                "data_highlights": ["Mixed numeric and categorical inputs"],
                "modeling_challenges": ["Class imbalance"],
                "submission_requirements": ["Write a valid submission file"],
                "evaluation_summary": "Hidden leaderboard uses macro F1.",
            },
            "stub",
        ),
    )

    manifest_path = Path(tmp_path) / manifest.slug / "manifest.json"
    challenge_path = Path(manifest.challenge_markdown_path)

    assert manifest_path.exists()
    assert challenge_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["brief_source"] == "stub"
    assert Path(payload["test_path"]).exists()
    assert Path(payload["sample_submission_path"]).exists()


def test_submission_evaluator_scores_hidden_solution(tmp_path: Path):
    solution_path = tmp_path / "solution.csv"
    pd.DataFrame({"row_id": [1, 2], "target": [0, 1]}).to_csv(
        solution_path, index=False
    )
    work_dir = tmp_path / "step_000"
    work_dir.mkdir()
    pd.DataFrame({"row_id": [1, 2], "target": [0, 1]}).to_csv(
        work_dir / "submission.csv", index=False
    )

    evaluator = build_submission_evaluator(
        {
            "solution_path": str(solution_path),
            "target_column": "target",
            "metric": "f1",
        }
    )
    result = evaluator(str(work_dir), {})

    assert result.is_buggy is False
    assert result.metric_value == 1.0
    result_payload = json.loads((work_dir / "result.json").read_text(encoding="utf-8"))
    assert result_payload["source"] == "hidden_submission_eval"


def test_contestant_spec_parses_custom_fields():
    spec = ContestantSpec.from_dict(
        {
            "name": "custom-agent",
            "provider": "custom",
            "program": "runner",
            "args_before_model": ["--json"],
            "model_flag": ["--model"],
            "prompt_mode": "arg",
            "prompt_flag": ["--prompt"],
        }
    )

    assert spec.program == "runner"
    assert spec.args_before_model == ("--json",)
    assert spec.prompt_mode == "arg"
