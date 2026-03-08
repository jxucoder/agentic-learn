"""Arena utilities for running multiple model contestants on one benchmark."""

from __future__ import annotations

import json
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
        )


def load_contestants(path: str) -> list[ContestantSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload["agents"] if isinstance(payload, dict) else payload
    return [ContestantSpec.from_dict(item) for item in items]


def build_submission_evaluator(
    manifest: dict[str, Any],
) -> Callable[[str, dict[str, Any]], EvaluationResult]:
    solution_path = Path(manifest["solution_path"])
    target_column = str(manifest["target_column"])
    metric = str(manifest["metric"])
    solution_df = pd.read_csv(solution_path)

    def evaluate(work_dir: str, _: dict[str, Any]) -> EvaluationResult:
        submission_path = Path(work_dir) / "submission.csv"
        if not submission_path.exists():
            return EvaluationResult(metric_value=None, is_buggy=True)

        try:
            submission_df = pd.read_csv(submission_path)
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

        result_payload = {
            "metric": score,
            "metric_name": metric,
            "source": "hidden_submission_eval",
            "submission_path": str(submission_path.resolve()),
        }
        (Path(work_dir) / "result.json").write_text(
            json.dumps(result_payload, indent=2), encoding="utf-8"
        )
        return EvaluationResult(metric_value=score, is_buggy=False)

    return evaluate


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
    evaluator = build_submission_evaluator(manifest)

    benchmark_output = Path(output_root).resolve() / manifest["slug"]
    benchmark_output.mkdir(parents=True, exist_ok=True)

    task = TaskConfig(
        description=manifest["public_description"],
        data_path=manifest["train_path"],
        target_column=manifest["target_column"],
        metric=manifest["metric"],
        instructions=manifest["agent_instructions"],
        resource_paths={
            "test_data": manifest["test_path"],
            "sample_submission": manifest["sample_submission_path"],
        },
    )

    results: list[dict[str, Any]] = []
    for contestant in contestants:
        run_dir = benchmark_output / contestant.name
        started = time.time()
        best = evolve(
            task,
            model=contestant.model,
            max_steps=steps,
            timeout=timeout,
            output_dir=str(run_dir),
            cli=_resolve_cli(contestant),
            evaluator=evaluator,
        )
        duration_seconds = round(time.time() - started, 2)
        results.append(
            {
                "name": contestant.name,
                "provider": contestant.provider,
                "model": contestant.model,
                "score": None if best is None else best.metric_value,
                "status": "ok" if best is not None else "failed",
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
                -(item["score"] or float("-inf")),
                item["name"],
            ),
        ),
    }
    _write_leaderboard(benchmark_output, leaderboard)
    return leaderboard


def _resolve_cli(contestant: ContestantSpec) -> AgentCLIConfig:
    if contestant.provider == "codex":
        return codex_cli_config()
    if contestant.provider == "codex-oss":
        return codex_cli_config(oss=True, local_provider=contestant.local_provider)
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


def _write_leaderboard(output_dir: Path, leaderboard: dict[str, Any]) -> None:
    (output_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2), encoding="utf-8"
    )

    lines = [
        f"# Arena Results: {leaderboard['benchmark_id']}",
        "",
        "| Rank | Contestant | Provider | Model | Score | Status | Duration (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(leaderboard["contestants"], start=1):
        score = "" if item["score"] is None else f"{item['score']:.6f}"
        model = item["model"] or ""
        lines.append(
            f"| {index} | {item['name']} | {item['provider']} | {model} | {score} | {item['status']} | {item['duration_seconds']:.2f} |"
        )
    (output_dir / "leaderboard.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
