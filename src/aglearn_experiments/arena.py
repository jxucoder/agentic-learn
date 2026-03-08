"""Arena utilities for running multiple model contestants on one benchmark."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score

from aglearn.runtime import (
    AgentCLIConfig,
    EvaluationResult,
    TaskConfig,
    claude_cli_config,
    codex_cli_config,
    evolve,
)


@dataclass(frozen=True)
class ContestantSpec:
    name: str
    provider: str
    model: str | None = None
    local_provider: str | None = None
    program: str | None = None
    args_before_model: tuple[str, ...] = ()
    args_after_model: tuple[str, ...] = ()
    model_flag: tuple[str, ...] = ()
    prompt_mode: str = "stdin"
    prompt_flag: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    secret_env: tuple[str, ...] = ()
    python_packages: tuple[str, ...] = ()
    npm_packages: tuple[str, ...] = ()
    apt_packages: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContestantSpec":
        return cls(
            name=str(payload["name"]),
            provider=str(payload["provider"]),
            model=payload.get("model"),
            local_provider=payload.get("local_provider"),
            program=payload.get("program"),
            args_before_model=tuple(payload.get("args_before_model", [])),
            args_after_model=tuple(payload.get("args_after_model", [])),
            model_flag=tuple(payload.get("model_flag", [])),
            prompt_mode=str(payload.get("prompt_mode", "stdin")),
            prompt_flag=tuple(payload.get("prompt_flag", [])),
            env={str(k): str(v) for k, v in payload.get("env", {}).items()},
            secret_env=tuple(str(item) for item in payload.get("secret_env", [])),
            python_packages=tuple(
                str(item) for item in payload.get("python_packages", [])
            ),
            npm_packages=tuple(str(item) for item in payload.get("npm_packages", [])),
            apt_packages=tuple(str(item) for item in payload.get("apt_packages", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "local_provider": self.local_provider,
            "program": self.program,
            "args_before_model": list(self.args_before_model),
            "args_after_model": list(self.args_after_model),
            "model_flag": list(self.model_flag),
            "prompt_mode": self.prompt_mode,
            "prompt_flag": list(self.prompt_flag),
            "env": dict(self.env),
            "secret_env": list(self.secret_env),
            "python_packages": list(self.python_packages),
            "npm_packages": list(self.npm_packages),
            "apt_packages": list(self.apt_packages),
        }


def load_contestants(path: str) -> list[ContestantSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload["agents"] if isinstance(payload, dict) else payload
    return [ContestantSpec.from_dict(item) for item in items]


def load_public_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in _PUBLIC_MANIFEST_PATH_KEYS:
        if key in manifest:
            manifest[key] = str(
                _resolve_manifest_path(manifest_path.parent, manifest[key])
            )
    return manifest


def build_public_validation_evaluator(
    manifest: dict[str, Any],
) -> Callable[[str, dict[str, Any]], EvaluationResult]:
    evaluation_script = Path(manifest["evaluation_script_path"])
    validator_script = Path(manifest["validator_script_path"])
    metric = str(manifest["metric"])
    submission_filename = str(manifest.get("submission_filename", "submission.csv"))

    def evaluate(work_dir: str, _: dict[str, Any]) -> EvaluationResult:
        work_path = Path(work_dir)
        validation_submission_path = work_path / "validation_submission.csv"
        submission_path = work_path / submission_filename

        validation_payload, validation_ok = _run_submission_script(
            script_path=evaluation_script,
            submission_path=validation_submission_path,
            work_dir=work_path,
            output_path=work_path / "validation_result.json",
        )
        validator_payload, validator_ok = _run_submission_script(
            script_path=validator_script,
            submission_path=submission_path,
            work_dir=work_path,
            output_path=work_path / "submission_validation.json",
        )
        if not validation_ok or not validator_ok:
            return EvaluationResult(metric_value=None, is_buggy=True)

        score = (
            _as_float(validation_payload.get("metric")) if validation_payload else None
        )
        if score is None:
            return EvaluationResult(metric_value=None, is_buggy=True)

        result_payload = {
            "metric": score,
            "metric_name": metric,
            "source": "public_validation_eval",
            "validation_submission_path": str(validation_submission_path.resolve()),
            "submission_path": str(submission_path.resolve()),
        }
        (work_path / "result.json").write_text(
            json.dumps(result_payload, indent=2), encoding="utf-8"
        )
        return EvaluationResult(metric_value=score, is_buggy=False)

    return evaluate


def build_submission_evaluator(
    manifest: dict[str, Any],
) -> Callable[[str, dict[str, Any]], EvaluationResult]:
    def evaluate(work_dir: str, _: dict[str, Any]) -> EvaluationResult:
        return evaluate_hidden_submission(
            manifest=manifest,
            submission_path=Path(work_dir)
            / str(manifest.get("submission_filename", "submission.csv")),
            result_path=Path(work_dir) / "result.json",
        )

    return evaluate


def evaluate_hidden_submission(
    *,
    manifest: dict[str, Any],
    submission_path: str | Path,
    result_path: str | Path | None = None,
) -> EvaluationResult:
    resolved_submission_path = Path(submission_path)
    solution_df = pd.read_csv(Path(manifest["solution_path"]))
    target_column = str(manifest["target_column"])
    metric = str(manifest["metric"])

    if not resolved_submission_path.exists():
        return EvaluationResult(metric_value=None, is_buggy=True)

    try:
        submission_df = pd.read_csv(resolved_submission_path)
    except Exception:
        return EvaluationResult(metric_value=None, is_buggy=True)

    if (
        "row_id" not in submission_df.columns
        or target_column not in submission_df.columns
    ):
        return EvaluationResult(metric_value=None, is_buggy=True)
    if submission_df["row_id"].duplicated().any():
        return EvaluationResult(metric_value=None, is_buggy=True)

    merged = solution_df.merge(
        submission_df[["row_id", target_column]],
        on="row_id",
        how="left",
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )
    pred_column = f"{target_column}_pred"
    true_column = f"{target_column}_true"
    if merged[pred_column].isna().any():
        return EvaluationResult(metric_value=None, is_buggy=True)
    if len(submission_df) != len(solution_df):
        return EvaluationResult(metric_value=None, is_buggy=True)

    score = _score_submission(
        metric=metric,
        y_true=merged[true_column],
        y_pred=merged[pred_column],
    )
    if score is None:
        return EvaluationResult(metric_value=None, is_buggy=True)

    if result_path is not None:
        Path(result_path).write_text(
            json.dumps(
                {
                    "metric": score,
                    "metric_name": metric,
                    "source": "hidden_submission_eval",
                    "submission_path": str(resolved_submission_path.resolve()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return EvaluationResult(metric_value=score, is_buggy=False)


def run_public_contestant_loop(
    *,
    manifest: dict[str, Any],
    contestant: ContestantSpec,
    steps: int,
    timeout: int,
    output_dir: str,
) -> dict[str, Any]:
    task = TaskConfig(
        description=manifest["public_description"],
        data_path=str(manifest["train_path"]),
        target_column=manifest["target_column"],
        metric=manifest["metric"],
        instructions=manifest["agent_instructions"],
        resource_paths={
            "validation_data": str(manifest["validation_path"]),
            "validation_sample_submission": str(
                manifest["validation_sample_submission_path"]
            ),
            "validation_evaluator": str(manifest["evaluation_script_path"]),
            "test_data": str(manifest["test_path"]),
            "sample_submission": str(manifest["sample_submission_path"]),
            "submission_validator": str(manifest["validator_script_path"]),
            "challenge": str(manifest["challenge_markdown_path"]),
        },
    )
    best = evolve(
        task,
        model=contestant.model,
        max_steps=steps,
        timeout=timeout,
        output_dir=output_dir,
        cli=_resolve_cli(contestant),
        evaluator=build_public_validation_evaluator(manifest),
    )
    run_path = Path(output_dir)
    summary = {
        "name": contestant.name,
        "provider": contestant.provider,
        "model": contestant.model,
        "public_score": None if best is None else best.metric_value,
        "status": "ok" if best is not None else "failed",
        "output_dir": str(run_path.resolve()),
        "best_submission_path": _existing_path(run_path / "best_submission.csv"),
        "best_validation_submission_path": _existing_path(
            run_path / "best_validation_submission.csv"
        ),
    }
    (run_path / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_arena(
    *,
    manifest_path: str,
    contestants_path: str,
    steps: int,
    timeout: int,
    output_root: str = "output/arena",
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    contestants = load_contestants(contestants_path)

    benchmark_output = Path(output_root).resolve() / manifest["slug"]
    private_root = benchmark_output / ".private_runs"
    benchmark_output.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for contestant in contestants:
        run_dir = private_root / contestant.name
        workspace_paths = _prepare_contestant_workspace(run_dir, manifest=manifest)
        public_manifest = load_public_manifest(workspace_paths["public_manifest_path"])
        started = time.time()
        public_summary = run_public_contestant_loop(
            manifest=public_manifest,
            contestant=contestant,
            steps=steps,
            timeout=timeout,
            output_dir=str(run_dir),
        )
        hidden_result = evaluate_hidden_submission(
            manifest=manifest,
            submission_path=run_dir / "best_submission.csv",
            result_path=run_dir / "final_hidden_result.json",
        )
        duration_seconds = round(time.time() - started, 2)
        results.append(
            {
                "name": contestant.name,
                "provider": contestant.provider,
                "model": contestant.model,
                "public_score": public_summary["public_score"],
                "score": None if hidden_result.is_buggy else hidden_result.metric_value,
                "final_hidden_score": None
                if hidden_result.is_buggy
                else hidden_result.metric_value,
                "status": "ok"
                if public_summary["status"] == "ok" and not hidden_result.is_buggy
                else "failed",
                "duration_seconds": duration_seconds,
                "output_dir": str(run_dir.resolve()),
            }
        )

    leaderboard = {
        "benchmark_id": manifest["benchmark_id"],
        "manifest_path": str(Path(manifest_path).resolve()),
        "contestants": sorted(
            results,
            key=lambda item: (
                item["score"] is None,
                -_score_key(item["score"]),
                item["name"],
            ),
        ),
    }
    _write_leaderboard(benchmark_output, leaderboard)
    return leaderboard


def _resolve_cli(contestant: ContestantSpec) -> AgentCLIConfig:
    if contestant.provider == "codex":
        return codex_cli_config(access_mode="sandbox", sandbox_mode="workspace-write")
    if contestant.provider == "codex-oss":
        return codex_cli_config(
            oss=True,
            local_provider=contestant.local_provider,
            access_mode="sandbox",
            sandbox_mode="workspace-write",
        )
    if contestant.provider == "claude":
        return claude_cli_config()
    if contestant.provider == "custom":
        if not contestant.program:
            raise ValueError("custom contestants require `program`")
        return AgentCLIConfig(
            name=contestant.name,
            program=contestant.program,
            args_before_model=contestant.args_before_model,
            args_after_model=contestant.args_after_model,
            model_flag=contestant.model_flag,
            prompt_mode=contestant.prompt_mode,
            prompt_flag=contestant.prompt_flag,
            env=contestant.env,
        )
    raise ValueError(f"Unsupported provider: {contestant.provider}")


def _score_submission(
    *,
    metric: str,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float | None:
    try:
        if metric == "f1":
            return float(f1_score(y_true.astype(int), y_pred.astype(int)))
        if metric == "f1_macro":
            return float(
                f1_score(y_true.astype(int), y_pred.astype(int), average="macro")
            )
        if metric == "accuracy":
            return float(accuracy_score(y_true.astype(int), y_pred.astype(int)))
        if metric == "r2":
            return float(r2_score(y_true.astype(float), y_pred.astype(float)))
    except (TypeError, ValueError):
        return None
    raise ValueError(f"Unsupported metric: {metric}")


def _score_key(value: float | None) -> float:
    return float("-inf") if value is None else value


def _write_leaderboard(output_dir: Path, leaderboard: dict[str, Any]) -> None:
    (output_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2), encoding="utf-8"
    )

    lines = [
        f"# Arena Results: {leaderboard['benchmark_id']}",
        "",
        "| Rank | Contestant | Provider | Model | Public Score | Final Score | Status | Duration (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(leaderboard["contestants"], start=1):
        public_score = (
            "" if item["public_score"] is None else f"{item['public_score']:.6f}"
        )
        score = "" if item["score"] is None else f"{item['score']:.6f}"
        model = item["model"] or ""
        lines.append(
            f"| {index} | {item['name']} | {item['provider']} | {model} | {public_score} | {score} | {item['status']} | {item['duration_seconds']:.2f} |"
        )
    (output_dir / "leaderboard.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _prepare_contestant_workspace(
    run_dir: Path,
    *,
    manifest: dict[str, Any],
) -> dict[str, str]:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    inputs_dir = run_dir / "inputs"
    inputs_data_dir = inputs_dir / "data"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    inputs_data_dir.mkdir(parents=True, exist_ok=True)

    copied_paths = {
        "train_path": _copy_input(Path(manifest["train_path"]), inputs_data_dir),
        "validation_path": _copy_input(
            Path(manifest["validation_path"]), inputs_data_dir
        ),
        "validation_sample_submission_path": _copy_input(
            Path(manifest["validation_sample_submission_path"]), inputs_data_dir
        ),
        "test_path": _copy_input(Path(manifest["test_path"]), inputs_data_dir),
        "sample_submission_path": _copy_input(
            Path(manifest["sample_submission_path"]), inputs_data_dir
        ),
        "evaluation_script_path": _copy_input(
            Path(manifest["evaluation_script_path"]), inputs_dir
        ),
        "validator_script_path": _copy_input(
            Path(manifest["validator_script_path"]), inputs_dir
        ),
        "challenge_markdown_path": _write_text_file(
            inputs_dir / "challenge.md",
            manifest["public_description"],
        ),
    }
    copied_paths["public_manifest_path"] = _write_text_file(
        inputs_dir / "contestant_manifest.json",
        json.dumps(_build_public_manifest(manifest, copied_paths), indent=2),
    )
    return copied_paths


_PUBLIC_MANIFEST_PATH_KEYS = (
    "train_path",
    "validation_path",
    "validation_sample_submission_path",
    "test_path",
    "sample_submission_path",
    "evaluation_script_path",
    "validator_script_path",
    "challenge_markdown_path",
)


def _build_public_manifest(
    manifest: dict[str, Any],
    copied_paths: dict[str, str],
) -> dict[str, Any]:
    public_manifest = {
        "benchmark_id": manifest["benchmark_id"],
        "slug": manifest["slug"],
        "title": manifest.get("title"),
        "task_type": manifest.get("task_type"),
        "metric": manifest["metric"],
        "target_column": manifest["target_column"],
        "submission_filename": manifest.get("submission_filename", "submission.csv"),
        "public_description": manifest["public_description"],
        "agent_instructions": manifest["agent_instructions"],
    }
    for key in _PUBLIC_MANIFEST_PATH_KEYS:
        public_manifest[key] = _public_manifest_path(copied_paths[key])
    return public_manifest


def _public_manifest_path(path: str) -> str:
    resolved = Path(path)
    if resolved.parent.name == "data":
        return str(Path("data") / resolved.name)
    return resolved.name


def _resolve_manifest_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _run_submission_script(
    *,
    script_path: Path,
    submission_path: Path,
    work_dir: Path,
    output_path: Path,
) -> tuple[dict[str, Any] | None, bool]:
    if not submission_path.exists():
        output_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "errors": [f"Missing submission file: {submission_path.name}"],
                    "submission_path": str(submission_path.resolve()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return None, False

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--submission",
            str(submission_path.resolve()),
        ],
        capture_output=True,
        text=True,
        cwd=work_dir,
        check=False,
    )
    payload = _parse_json_payload(proc.stdout)
    if payload is None:
        payload = {
            "ok": False,
            "errors": ["Submission script did not return valid JSON."],
            "submission_path": str(submission_path.resolve()),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, proc.returncode == 0 and bool(payload.get("ok"))


def _parse_json_payload(stdout: str) -> dict[str, Any] | None:
    if not stdout.strip():
        return None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _existing_path(path: Path) -> str | None:
    return str(path.resolve()) if path.exists() else None


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _copy_input(source: Path, target_dir: Path) -> str:
    destination = target_dir / source.name
    shutil.copy2(source, destination)
    return str(destination.resolve())


def _write_text_file(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path.resolve())
