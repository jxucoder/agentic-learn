"""Modal-backed arena execution with one sandbox per contestant."""

from __future__ import annotations

import json
import os
import shlex
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .arena import (
    ContestantSpec,
    _prepare_contestant_workspace,
    _write_leaderboard,
    evaluate_hidden_submission,
    load_contestants,
)

_ROOT_ARTIFACTS = (
    "journal.jsonl",
    "best_solution.py",
    "best_submission.csv",
    "best_validation_submission.csv",
    "best_result.json",
    "best_exploration.md",
    "report.md",
    "report.pdf",
    "run_summary.json",
)
_STEP_ARTIFACTS = (
    "solution.py",
    "submission.csv",
    "validation_submission.csv",
    "result.json",
    "validation_result.json",
    "submission_validation.json",
    "exploration.md",
    "trace.jsonl",
    "trace.stderr.log",
)


@dataclass
class _ModalRun:
    contestant: ContestantSpec
    run_dir: Path
    volume: Any
    sandbox: Any
    started_at: float


def run_modal_arena(
    *,
    manifest_path: str,
    contestants_path: str,
    steps: int,
    timeout: int,
    output_root: str = "output/modal_arena",
    app_name: str = "aglearn-arena",
    sandbox_timeout: int | None = None,
    apt_packages: tuple[str, ...] = (),
    npm_packages: tuple[str, ...] = (),
    python_packages: tuple[str, ...] = (),
    image_commands: tuple[str, ...] = (),
) -> dict[str, Any]:
    modal = _load_modal()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    contestants = load_contestants(contestants_path)

    benchmark_output = Path(output_root).resolve() / manifest["slug"]
    private_root = benchmark_output / ".private_runs"
    benchmark_output.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)

    image = _build_modal_image(
        modal,
        repo_root=_repo_root(),
        contestants=contestants,
        apt_packages=apt_packages,
        npm_packages=npm_packages,
        python_packages=python_packages,
        image_commands=image_commands,
    )
    app = modal.App.lookup(app_name, create_if_missing=True)
    resolved_sandbox_timeout = sandbox_timeout or _default_sandbox_timeout(
        steps=steps,
        timeout=timeout,
    )

    results: list[dict[str, Any]] = []
    with ExitStack() as stack:
        runs: list[_ModalRun] = []
        for contestant in contestants:
            run_dir = private_root / contestant.name
            workspace_paths = _prepare_contestant_workspace(run_dir, manifest=manifest)
            _write_json(
                Path(workspace_paths["public_manifest_path"]).parent
                / "contestant_spec.json",
                contestant.to_dict(),
            )

            volume = stack.enter_context(modal.Volume.ephemeral())
            _upload_inputs(volume, run_dir / "inputs")
            sandbox = modal.Sandbox.create(
                *_modal_worker_command(steps=steps, timeout=timeout),
                app=app,
                image=image,
                workdir="/repo",
                timeout=resolved_sandbox_timeout,
                volumes={"/workspace": volume},
                env=_sandbox_env(contestant),
                secrets=_sandbox_secrets(modal, contestant),
            )
            runs.append(
                _ModalRun(
                    contestant=contestant,
                    run_dir=run_dir,
                    volume=volume,
                    sandbox=sandbox,
                    started_at=time.time(),
                )
            )

        for run in runs:
            stdout = run.sandbox.stdout.read() or ""
            stderr = run.sandbox.stderr.read() or ""
            run.sandbox.wait()
            if stdout:
                (run.run_dir / "modal.stdout.log").write_text(stdout, encoding="utf-8")
            if stderr:
                (run.run_dir / "modal.stderr.log").write_text(stderr, encoding="utf-8")

            _download_run_artifacts(run.volume, run.run_dir, steps=steps)
            run_summary = _load_json_if_exists(run.run_dir / "run_summary.json") or {}
            hidden_result = evaluate_hidden_submission(
                manifest=manifest,
                submission_path=run.run_dir / "best_submission.csv",
                result_path=run.run_dir / "final_hidden_result.json",
            )
            duration_seconds = round(time.time() - run.started_at, 2)
            sandbox_returncode = getattr(run.sandbox, "returncode", None)
            results.append(
                {
                    "name": run.contestant.name,
                    "provider": run.contestant.provider,
                    "model": run.contestant.model,
                    "public_score": run_summary.get("public_score"),
                    "score": (
                        None if hidden_result.is_buggy else hidden_result.metric_value
                    ),
                    "final_hidden_score": (
                        None if hidden_result.is_buggy else hidden_result.metric_value
                    ),
                    "status": (
                        "ok"
                        if run_summary.get("status") == "ok"
                        and not hidden_result.is_buggy
                        and sandbox_returncode in {None, 0}
                        else "failed"
                    ),
                    "duration_seconds": duration_seconds,
                    "output_dir": str(run.run_dir.resolve()),
                    "sandbox_returncode": sandbox_returncode,
                }
            )

    leaderboard = {
        "benchmark_id": manifest["benchmark_id"],
        "manifest_path": str(Path(manifest_path).resolve()),
        "executor": "modal",
        "contestants": sorted(
            results,
            key=lambda item: (
                item["score"] is None,
                -(item["score"] if item["score"] is not None else float("-inf")),
                item["name"],
            ),
        ),
    }
    _write_leaderboard(benchmark_output, leaderboard)
    return leaderboard


def _build_modal_image(
    modal: Any,
    *,
    repo_root: Path,
    contestants: list[ContestantSpec],
    apt_packages: tuple[str, ...],
    npm_packages: tuple[str, ...],
    python_packages: tuple[str, ...],
    image_commands: tuple[str, ...],
) -> Any:
    image = modal.Image.debian_slim(python_version="3.11")
    resolved_apt = set(apt_packages)
    resolved_python = {"uv", *python_packages}
    resolved_npm = set(npm_packages)

    if any(item.provider in {"codex", "codex-oss"} for item in contestants):
        resolved_npm.add("@openai/codex")
    if any(item.provider == "claude" for item in contestants):
        resolved_npm.add("@anthropic-ai/claude-code")
    if resolved_npm:
        resolved_apt.update({"git", "nodejs", "npm"})

    for contestant in contestants:
        resolved_apt.update(contestant.apt_packages)
        resolved_python.update(contestant.python_packages)
        resolved_npm.update(contestant.npm_packages)

    if resolved_apt:
        image = image.apt_install(*sorted(resolved_apt))
    if resolved_python:
        image = image.pip_install(*sorted(resolved_python))
    image = image.add_local_dir(str(repo_root), remote_path="/repo")

    commands = ["cd /repo && uv sync --dev"]
    if resolved_npm:
        commands.append(
            "npm install -g "
            + " ".join(shlex.quote(pkg) for pkg in sorted(resolved_npm))
        )
    commands.extend(image_commands)
    if commands:
        image = image.run_commands(*commands)
    return image


def _default_sandbox_timeout(*, steps: int, timeout: int) -> int:
    return max(steps * timeout + 900, 1800)


def _sandbox_env(contestant: ContestantSpec) -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        **contestant.env,
    }


def _sandbox_secrets(modal: Any, contestant: ContestantSpec) -> list[Any]:
    secret_names = set(contestant.secret_env)
    if contestant.provider == "claude":
        secret_names.add("ANTHROPIC_API_KEY")
    if contestant.provider == "codex":
        secret_names.add("OPENAI_API_KEY")

    if not secret_names:
        return []

    payload: dict[str, str] = {}
    missing: list[str] = []
    for name in sorted(secret_names):
        value = os.getenv(name)
        if value:
            payload[name] = value
        else:
            missing.append(name)
    if missing:
        missing_joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variable(s) for {contestant.name}: "
            f"{missing_joined}"
        )
    return [modal.Secret.from_dict(payload)]


def _modal_worker_command(*, steps: int, timeout: int) -> list[str]:
    return [
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
        str(steps),
        "--timeout",
        str(timeout),
        "--output-dir",
        "/workspace/output",
    ]


def _upload_inputs(volume: Any, inputs_dir: Path) -> None:
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(inputs_dir), "/inputs")


def _download_run_artifacts(volume: Any, run_dir: Path, *, steps: int) -> None:
    for relative_path in _ROOT_ARTIFACTS:
        _copy_volume_file(
            volume=volume,
            remote_path=f"output/{relative_path}",
            local_path=run_dir / relative_path,
        )
    for step_index in range(steps):
        step_name = f"step_{step_index:03d}"
        for artifact_name in _STEP_ARTIFACTS:
            _copy_volume_file(
                volume=volume,
                remote_path=f"output/{step_name}/{artifact_name}",
                local_path=run_dir / step_name / artifact_name,
            )


def _copy_volume_file(*, volume: Any, remote_path: str, local_path: Path) -> None:
    try:
        chunks = volume.read_file(remote_path)
    except FileNotFoundError:
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as handle:
        for chunk in chunks:
            handle.write(chunk)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_modal() -> Any:
    try:
        import modal
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Modal support requires the `modal` package. Install it with "
            "`uv add modal` or `uv sync --extra modal`."
        ) from exc
    return modal
