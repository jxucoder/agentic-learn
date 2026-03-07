"""Run a coding agent (codex CLI) to produce a solution."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from typing import TypedDict


class AgentRunResult(TypedDict):
    code: str
    hypothesis: str
    exploration: str
    metric_value: float | None
    is_buggy: bool
    stdout: str
    stderr: str


def run(
    prompt: str,
    work_dir: str,
    *,
    model: str | None = None,
    timeout: int = 600,
) -> AgentRunResult:
    """Invoke ``codex exec`` and return the result.

    The agent works inside *work_dir*: it explores data, writes
    ``solution.py``, runs it, and writes ``result.json``.

    The full session trace (thinking, commands, outputs) is saved
    to ``trace.jsonl`` for research reproducibility.

    Returns dict with keys: code, hypothesis, metric_value, is_buggy.
    """
    os.makedirs(work_dir, exist_ok=True)
    response_path = os.path.join(work_dir, ".agent_response.md")
    trace_path = os.path.join(work_dir, "trace.jsonl")
    stderr_path = os.path.join(work_dir, "trace.stderr.log")
    _clear_run_artifacts(
        response_path,
        trace_path,
        stderr_path,
        os.path.join(work_dir, "solution.py"),
        os.path.join(work_dir, "result.json"),
        os.path.join(work_dir, "exploration.md"),
    )
    cmd = _build_command(work_dir, response_path, model=model)
    run_env = _build_run_env()

    raw_stdout, raw_stderr, return_code, timed_out = _invoke_codex(
        cmd=cmd,
        prompt=prompt,
        timeout=timeout,
        run_env=run_env,
    )

    if raw_stdout:
        _write(trace_path, raw_stdout)
    if raw_stderr:
        _write(stderr_path, raw_stderr)

    code, result_data, exploration, hypothesis = _load_run_artifacts(
        work_dir=work_dir,
        response_path=response_path,
    )
    metric = _metric_from_result(result_data)

    fallback_note = ""
    if metric is None and code:
        metric, fallback_note = _run_solution_fallback(
            work_dir, timeout=min(timeout, 180)
        )

    if not hypothesis:
        hypothesis = _fallback_hypothesis(
            timed_out=timed_out,
            timeout=timeout,
            return_code=return_code,
            stderr=raw_stderr,
            fallback_note=fallback_note,
            metric=metric,
        )

    return {
        "code": code,
        "hypothesis": hypothesis or "",
        "exploration": exploration,
        "metric_value": metric,
        "is_buggy": metric is None,
        "stdout": raw_stdout,
        "stderr": raw_stderr,
    }


def _invoke_codex(
    *,
    cmd: list[str],
    prompt: str,
    timeout: int,
    run_env: dict[str, str],
) -> tuple[str, str, int | None, bool]:
    """Run codex and return stdout, stderr, return_code, timed_out."""
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        return proc.stdout or "", proc.stderr or "", proc.returncode, False
    except subprocess.TimeoutExpired as e:
        return _to_text(e.stdout), _to_text(e.stderr), None, True


def _load_run_artifacts(
    *, work_dir: str, response_path: str
) -> tuple[str, dict | None, str, str]:
    """Load all artifacts expected from a codex run."""
    code = _read(os.path.join(work_dir, "solution.py"))
    result_data = _read_json(os.path.join(work_dir, "result.json"))
    exploration = _read(os.path.join(work_dir, "exploration.md")).strip()
    hypothesis = _read(response_path).strip()
    return code, result_data, exploration, hypothesis


def _metric_from_result(result_data: dict | None) -> float | None:
    if not result_data:
        return None
    return _as_float(result_data.get("metric"))


def _build_command(
    work_dir: str, response_path: str, *, model: str | None
) -> list[str]:
    """Build codex CLI invocation with robust local-access defaults."""
    access_mode = os.getenv("AGLEARN_CODEX_ACCESS_MODE", "bypass").strip().lower()

    cmd = [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        "-C",
        work_dir,
        "-o",
        response_path,
    ]
    if access_mode in {"full-auto", "sandbox"}:
        cmd[2:2] = ["--full-auto", "--sandbox", "danger-full-access"]
    else:
        # Default to bypass mode to prevent sandbox_apply failures in restricted runtimes.
        cmd.insert(2, "--dangerously-bypass-approvals-and-sandbox")

    if model:
        cmd.extend(["-m", model])
    cmd.append("-")  # read prompt from stdin
    return cmd


def _build_run_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("LOKY_MAX_CPU_COUNT", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def _run_solution_fallback(work_dir: str, *, timeout: int) -> tuple[float | None, str]:
    """Recover a metric by running solution.py if codex omitted result.json."""
    solution_path = os.path.join(work_dir, "solution.py")
    result_path = os.path.join(work_dir, "result.json")
    if not os.path.exists(solution_path):
        return None, ""

    try:
        proc = subprocess.run(
            [sys.executable, solution_path],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_build_run_env(),
        )
    except subprocess.TimeoutExpired:
        return None, f"Fallback execution timed out after {timeout}s."
    except OSError as e:
        return None, f"Fallback execution failed: {e}"

    result_data = _read_json(result_path)
    metric = _as_float(result_data.get("metric")) if result_data else None
    if metric is None:
        metric = _metric_from_stdout(proc.stdout or "")
        if metric is not None:
            _write_json(result_path, {"metric": metric})

    if metric is not None:
        return (
            metric,
            "Recovered metric by running solution.py locally after missing result.json.",
        )

    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        reason = (detail[-1] if detail else "non-zero exit")[:240]
        return None, f"Fallback execution failed: {reason}"

    return None, "Fallback execution finished but did not emit a metric."


def _fallback_hypothesis(
    *,
    timed_out: bool,
    timeout: int,
    return_code: int | None,
    stderr: str,
    fallback_note: str,
    metric: float | None,
) -> str:
    if metric is not None and fallback_note:
        return fallback_note
    if timed_out:
        return f"codex exec timed out after {timeout}s."
    if return_code not in (None, 0):
        err_line = _last_nonempty_line(stderr)
        if err_line:
            return f"codex exec failed (exit {return_code}): {err_line[:240]}"
        return f"codex exec failed with exit code {return_code}."
    if fallback_note:
        return fallback_note
    return "codex exec did not produce a hypothesis."


def _metric_from_stdout(stdout: str) -> float | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            metric = _as_float(payload.get("metric"))
            if metric is not None:
                return metric
    return None


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _clear_run_artifacts(*paths: str) -> None:
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            continue


def _write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
