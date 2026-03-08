"""Tests for experiment-side benchmark and arena helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from aglearn_experiments.arena import (
    ContestantSpec,
    _prepare_contestant_workspace,
    _resolve_cli,
    build_public_validation_evaluator,
    build_submission_evaluator,
    run_arena,
)
from aglearn_experiments.benchmarks import _parse_brief_payload, generate_benchmark
from aglearn_experiments.modal_backend import _modal_worker_command, _sandbox_secrets


def test_generate_benchmark_writes_manifest_and_challenge(tmp_path: Path):
    manifest = generate_benchmark(
        task_type="multiclass",
        seed=7,
        samples=200,
        noise=0.2,
        output_root=str(tmp_path),
        theme="Mars housing price",
        brief_generator=lambda prompt, model, cwd: (
            {
                "title": "Mars Habitat Price Forecasting",
                "short_description": "Hard multiclass benchmark for Mars housing markets.",
                "scenario": "Predict a hidden outcome in a Martian real-estate pricing dataset.",
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
    validator_path = Path(payload["validator_script_path"])
    evaluation_path = Path(payload["evaluation_script_path"])
    assert payload["experiment_name"] == manifest.slug
    assert payload["theme"] == "Mars housing price"
    assert payload["brief_source"] == "stub"
    assert payload["submission_filename"] == "submission.csv"
    assert Path(payload["validation_path"]).exists()
    assert Path(payload["validation_sample_submission_path"]).exists()
    assert Path(payload["test_path"]).exists()
    assert Path(payload["sample_submission_path"]).exists()
    assert validator_path.exists()
    assert evaluation_path.exists()
    challenge_text = challenge_path.read_text(encoding="utf-8")
    assert "## Submission Format" in challenge_text
    assert "## Public Validation" in challenge_text
    assert "## Local Validation" in challenge_text
    assert (
        "uv run python validate_submission.py --submission submission.csv"
        in challenge_text
    )
    assert (
        "uv run python evaluate_validation.py --submission validation_submission.csv"
        in challenge_text
    )
    validation = subprocess.run(
        [
            sys.executable,
            str(validator_path),
            "--submission",
            payload["sample_submission_path"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr
    validation_frame = pd.read_csv(payload["validation_path"])
    validation_submission_path = Path(tmp_path) / "validation_submission.csv"
    validation_frame[["row_id", "target"]].to_csv(
        validation_submission_path, index=False
    )
    evaluation = subprocess.run(
        [
            sys.executable,
            str(evaluation_path),
            "--submission",
            str(validation_submission_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert evaluation.returncode == 0, evaluation.stdout + evaluation.stderr
    assert json.loads(evaluation.stdout)["metric"] == 1.0


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


def test_generate_benchmark_theme_flows_into_fallback_brief(tmp_path: Path):
    manifest = generate_benchmark(
        task_type="regression",
        seed=5,
        samples=120,
        noise=0.1,
        output_root=str(tmp_path),
        experiment_name="mars-housing-price",
        theme="Mars housing price",
        allow_fallback=True,
        brief_generator=lambda prompt, model, cwd: (_ for _ in ()).throw(
            RuntimeError("force fallback")
        ),
    )

    challenge = Path(manifest.challenge_markdown_path).read_text(encoding="utf-8")
    assert "Mars Housing Price" in challenge


def test_generate_benchmark_requires_gemini_success_by_default(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Gemini brief generation failed"):
        generate_benchmark(
            task_type="regression",
            seed=5,
            samples=120,
            noise=0.1,
            output_root=str(tmp_path),
            experiment_name="mars-housing-price",
            theme="Mars housing price",
            brief_generator=lambda prompt, model, cwd: (_ for _ in ()).throw(
                RuntimeError("force failure")
            ),
        )


def test_generate_benchmark_rejects_generic_brief_for_themed_setup(tmp_path: Path):
    with pytest.raises(RuntimeError, match="did not reflect the requested theme"):
        generate_benchmark(
            task_type="regression",
            seed=5,
            samples=120,
            noise=0.1,
            output_root=str(tmp_path),
            experiment_name="mars-housing-price",
            theme="Mars housing price",
            brief_generator=lambda prompt, model, cwd: (
                {
                    "title": "Regression Benchmark",
                    "short_description": "A hard Kaggle-style tabular benchmark.",
                    "scenario": "You are competing on a tabular prediction task.",
                    "objective": "Maximize r2.",
                    "data_highlights": ["A"],
                    "modeling_challenges": ["B"],
                    "submission_requirements": ["C"],
                    "evaluation_summary": "Hidden leaderboard uses r2.",
                },
                "stub",
            ),
        )


def test_parse_brief_payload_unwraps_fenced_json_from_response_envelope():
    payload = _parse_brief_payload(
        json.dumps(
            {
                "session_id": "abc",
                "response": (
                    "```json\n"
                    '{"title":"Mars Housing Price Challenge","scenario":"Predict Mars home values."}\n'
                    "```"
                ),
            }
        )
    )

    assert payload["title"] == "Mars Housing Price Challenge"
    assert payload["scenario"] == "Predict Mars home values."


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


def test_public_validation_evaluator_uses_validation_script(tmp_path: Path):
    manifest = generate_benchmark(
        task_type="classification",
        seed=3,
        samples=120,
        noise=0.1,
        output_root=str(tmp_path),
        experiment_name="clear-signal",
        brief_generator=lambda prompt, model, cwd: (
            {
                "title": "Clear Signal",
                "short_description": "A benchmark.",
                "scenario": "Predict the target.",
                "objective": "Maximize F1.",
                "data_highlights": ["A"],
                "modeling_challenges": ["B"],
                "submission_requirements": ["C"],
                "evaluation_summary": "Public validation uses F1.",
            },
            "stub",
        ),
    )
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    validation_df = pd.read_csv(manifest.validation_path)
    validation_df[["row_id", "target"]].to_csv(
        work_dir / "validation_submission.csv",
        index=False,
    )
    pd.read_csv(manifest.sample_submission_path).to_csv(
        work_dir / "submission.csv",
        index=False,
    )

    evaluator = build_public_validation_evaluator(manifest.to_dict())
    result = evaluator(str(work_dir), {})

    assert result.is_buggy is False
    assert result.metric_value == 1.0
    result_payload = json.loads((work_dir / "result.json").read_text(encoding="utf-8"))
    assert result_payload["source"] == "public_validation_eval"
    validator_payload = json.loads(
        (work_dir / "submission_validation.json").read_text(encoding="utf-8")
    )
    assert validator_payload["ok"] is True


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
            "secret_env": ["OPENAI_API_KEY"],
        }
    )

    assert spec.program == "runner"
    assert spec.args_before_model == ("--json",)
    assert spec.prompt_mode == "arg"
    assert spec.secret_env == ("OPENAI_API_KEY",)


def test_prepare_contestant_workspace_writes_sanitized_manifest(tmp_path: Path):
    manifest_dir = tmp_path / "exp"
    manifest_dir.mkdir()
    for name in (
        "train.csv",
        "validation.csv",
        "validation_sample.csv",
        "test.csv",
        "sample.csv",
        "solution.csv",
    ):
        pd.DataFrame({"row_id": [1], "target": [0]}).to_csv(
            manifest_dir / name, index=False
        )
    (manifest_dir / "validate_submission.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )
    (manifest_dir / "evaluate_validation.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )

    copied = _prepare_contestant_workspace(
        tmp_path / "run",
        manifest={
            "benchmark_id": "mars-demo",
            "slug": "mars-demo",
            "task_type": "classification",
            "metric": "f1",
            "target_column": "target",
            "submission_filename": "submission.csv",
            "public_description": "demo",
            "agent_instructions": "demo",
            "train_path": str((manifest_dir / "train.csv").resolve()),
            "validation_path": str((manifest_dir / "validation.csv").resolve()),
            "validation_sample_submission_path": str(
                (manifest_dir / "validation_sample.csv").resolve()
            ),
            "test_path": str((manifest_dir / "test.csv").resolve()),
            "sample_submission_path": str((manifest_dir / "sample.csv").resolve()),
            "solution_path": str((manifest_dir / "solution.csv").resolve()),
            "evaluation_script_path": str(
                (manifest_dir / "evaluate_validation.py").resolve()
            ),
            "validator_script_path": str(
                (manifest_dir / "validate_submission.py").resolve()
            ),
        },
    )

    public_manifest = json.loads(
        Path(copied["public_manifest_path"]).read_text(encoding="utf-8")
    )
    assert "solution_path" not in public_manifest
    assert public_manifest["train_path"] == "data/train.csv"
    assert public_manifest["validation_path"] == "data/validation.csv"
    assert public_manifest["evaluation_script_path"] == "evaluate_validation.py"


def test_run_arena_uses_private_contestant_workspaces(tmp_path: Path, monkeypatch):
    manifest_dir = tmp_path / "exp"
    manifest_dir.mkdir()
    pd.DataFrame({"row_id": [1, 2], "target": [0, 1]}).to_csv(
        manifest_dir / "train.csv", index=False
    )
    pd.DataFrame({"row_id": [11, 12], "target": [1, 0]}).to_csv(
        manifest_dir / "validation.csv", index=False
    )
    pd.DataFrame({"row_id": [11, 12], "target": [0, 0]}).to_csv(
        manifest_dir / "validation_sample.csv", index=False
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
                "validation_path": str((manifest_dir / "validation.csv").resolve()),
                "validation_sample_submission_path": str(
                    (manifest_dir / "validation_sample.csv").resolve()
                ),
                "test_path": str((manifest_dir / "test.csv").resolve()),
                "sample_submission_path": str((manifest_dir / "sample.csv").resolve()),
                "solution_path": str((manifest_dir / "solution.csv").resolve()),
                "evaluation_script_path": str(
                    (manifest_dir / "evaluate_validation.py").resolve()
                ),
                "validator_script_path": str(
                    (manifest_dir / "validate_submission.py").resolve()
                ),
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
    (manifest_dir / "validate_submission.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )
    (manifest_dir / "evaluate_validation.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )

    def fake_evolve(task, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        assert ".private_runs" in str(output_dir)
        assert task.data_path.startswith(str(output_dir))
        for resource_path in task.resource_paths.values():
            assert resource_path.startswith(str(output_dir))
        assert "submission_validator" in task.resource_paths
        assert "validation_data" in task.resource_paths
        assert "validation_sample_submission" in task.resource_paths
        assert "validation_evaluator" in task.resource_paths
        (output_dir / "best_submission.csv").write_text(
            "row_id,target\n3,0\n4,1\n", encoding="utf-8"
        )
        (output_dir / "run_summary.json").write_text(
            json.dumps({"public_score": 1.0, "status": "ok"}),
            encoding="utf-8",
        )
        return type("Best", (), {"metric_value": 1.0})()

    monkeypatch.setattr("aglearn_experiments.arena.evolve", fake_evolve)

    leaderboard = run_arena(
        manifest_path=str(manifest_path),
        contestants_path=str(contestants_path),
        steps=1,
        timeout=1,
        output_root=str(tmp_path / "arena"),
    )

    assert len(leaderboard["contestants"]) == 2
    assert (tmp_path / "arena" / "silent-orbit" / "leaderboard.json").exists()
    assert leaderboard["contestants"][0]["public_score"] == 1.0


def test_codex_contestants_use_workspace_sandbox():
    cli = _resolve_cli(ContestantSpec(name="codex", provider="codex"))
    assert "--sandbox" in cli.args_before_model
    assert "workspace-write" in cli.args_before_model


def test_modal_worker_command_uses_private_input_mount():
    assert _modal_worker_command(steps=8, timeout=600) == [
        "uv",
        "run",
        "python",
        "-m",
        "aglearn_experiments.modal_worker",
        "--manifest",
        "/workspace/inputs/contestant_manifest.json",
        "--contestant",
        "/workspace/inputs/contestant_spec.json",
        "--steps",
        "8",
        "--timeout",
        "600",
        "--output-dir",
        "/workspace/output",
    ]


def test_modal_backend_requires_provider_secrets(monkeypatch):
    class DummySecret:
        @staticmethod
        def from_dict(payload):
            return payload

    class DummyModal:
        Secret = DummySecret

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    secrets = _sandbox_secrets(
        DummyModal,
        ContestantSpec(name="codex", provider="codex", model="gpt-5.3-codex"),
    )

    assert secrets == [{"OPENAI_API_KEY": "test-openai-key"}]
