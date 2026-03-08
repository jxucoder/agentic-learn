"""Tests for experiment-side benchmark and arena helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aglearn.runtime.loop import EvaluationResult
from aglearn_experiments.arena import (
    ContestantSpec,
    _resolve_cli,
    build_submission_evaluator,
    run_arena,
)
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
    assert payload["experiment_name"] == manifest.slug
    assert payload["brief_source"] == "stub"
    assert Path(payload["test_path"]).exists()
    assert Path(payload["sample_submission_path"]).exists()


def test_generate_benchmark_uses_random_two_word_name(tmp_path: Path):
    manifest = generate_benchmark(
        task_type="classification",
        seed=11,
        samples=100,
        noise=0.1,
        output_root=str(tmp_path),
        brief_generator=lambda prompt, model, cwd: (
            {
                "title": "Synthetic Signals",
                "short_description": "Test brief.",
                "scenario": "Predict a hidden outcome.",
                "objective": "Maximize F1.",
                "data_highlights": ["A"],
                "modeling_challenges": ["B"],
                "submission_requirements": ["C"],
                "evaluation_summary": "Hidden F1.",
            },
            "stub",
        ),
    )

    parts = manifest.experiment_name.split("-")
    assert len(parts) == 2
    assert all(parts)


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


def test_run_arena_uses_private_contestant_workspaces(tmp_path: Path, monkeypatch):
    manifest_dir = tmp_path / "exp"
    manifest_dir.mkdir()
    pd.DataFrame({"row_id": [1, 2], "target": [0, 1]}).to_csv(
        manifest_dir / "train.csv", index=False
    )
    pd.DataFrame({"row_id": [3, 4]}).to_csv(manifest_dir / "test.csv", index=False)
    pd.DataFrame({"row_id": [3, 4], "target": [0, 1]}).to_csv(
        manifest_dir / "solution.csv", index=False
    )
    pd.DataFrame({"row_id": [3, 4], "target": [0, 0]}).to_csv(
        manifest_dir / "sample.csv", index=False
    )
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark_id": "silent-orbit",
                "slug": "silent-orbit",
                "public_description": "demo",
                "agent_instructions": "write submission.csv",
                "train_path": str((manifest_dir / "train.csv").resolve()),
                "test_path": str((manifest_dir / "test.csv").resolve()),
                "sample_submission_path": str((manifest_dir / "sample.csv").resolve()),
                "solution_path": str((manifest_dir / "solution.csv").resolve()),
                "target_column": "target",
                "metric": "f1",
            }
        ),
        encoding="utf-8",
    )
    contestants_path = manifest_dir / "contestants.json"
    contestants_path.write_text(
        json.dumps(
            {
                "agents": [
                    {"name": "alpha", "provider": "custom", "program": "runner"},
                    {"name": "beta", "provider": "custom", "program": "runner"},
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_evolve(task, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        assert ".private_runs" in str(output_dir)
        assert task.data_path.startswith(str(output_dir))
        for resource_path in task.resource_paths.values():
            assert resource_path.startswith(str(output_dir))
        (output_dir / "submission.csv").write_text(
            "row_id,target\n3,0\n4,1\n", encoding="utf-8"
        )
        return type("Best", (), {"metric_value": 1.0})()

    monkeypatch.setattr("aglearn_experiments.arena.evolve", fake_evolve)
    monkeypatch.setattr(
        "aglearn_experiments.arena.build_submission_evaluator",
        lambda manifest: (
            lambda work_dir, result: EvaluationResult(metric_value=1.0, is_buggy=False)
        ),
    )

    leaderboard = run_arena(
        manifest_path=str(manifest_path),
        contestants_path=str(contestants_path),
        steps=1,
        timeout=1,
        output_root=str(tmp_path / "arena"),
    )

    assert len(leaderboard["contestants"]) == 2
    assert (tmp_path / "arena" / "silent-orbit" / "leaderboard.json").exists()


def test_codex_contestants_use_workspace_sandbox():
    cli = _resolve_cli(ContestantSpec(name="codex", provider="codex"))
    assert "--sandbox" in cli.args_before_model
    assert "workspace-write" in cli.args_before_model
