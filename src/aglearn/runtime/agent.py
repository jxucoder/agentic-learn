"""Run a coding agent through a CLI tool to produce a solution."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Literal, Mapping, TypedDict

PromptMode = Literal["stdin", "arg"]


@dataclass(frozen=True)
class AgentCLIConfig:
    """Describe how to invoke an agentic CLI."""

    name: str
    program: str
    args_before_model: tuple[str, ...] = ()
    args_after_model: tuple[str, ...] = ()
    model_flag: tuple[str, ...] = ()
    prompt_mode: PromptMode = "stdin"
    prompt_flag: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


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
    cli: AgentCLIConfig | None = None,
) -> AgentRunResult:
    """Invoke an agent CLI and return the result.

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
    cli_config = cli or codex_cli_config()
    _clear_run_artifacts(
        response_path,
        trace_path,
        stderr_path,
        os.path.join(work_dir, "solution.py"),
        os.path.join(work_dir, "result.json"),
        os.path.join(work_dir, "exploration.md"),
        os.path.join(work_dir, "submission.csv"),
    )
    cmd = _build_command(
        work_dir,
        response_path,
        model=model,
        cli=cli_config,
        prompt=prompt,
    )
    run_env = _build_run_env(cli=cli_config)

    raw_stdout, raw_stderr, return_code, timed_out = _invoke_agent(
        cmd=cmd,
        prompt=prompt if cli_config.prompt_mode == "stdin" else None,
        work_dir=work_dir,
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
        raw_stdout=raw_stdout,
    )
    metric = _metric_from_result(result_data)

    fallback_note = ""
    if metric is None and code:
        metric, fallback_note = _run_solution_fallback(
            work_dir, timeout=min(timeout, 180)
        )

    if not hypothesis:
        hypothesis = _fallback_hypothesis(
            runner_name=cli_config.name,
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


def codex_cli_config(
    *,
    oss: bool = False,
    local_provider: str | None = None,
    access_mode: str | None = None,
    sandbox_mode: str = "danger-full-access",
) -> AgentCLIConfig:
    """Return the default Codex CLI configuration."""
    resolved_access_mode = (
        access_mode or os.getenv("AGLEARN_CODEX_ACCESS_MODE", "bypass").strip().lower()
    )
    access_args: list[str]
    if resolved_access_mode in {"full-auto", "sandbox"}:
        access_args = ["--full-auto", "--sandbox", sandbox_mode]
    else:
        access_args = ["--dangerously-bypass-approvals-and-sandbox"]

    args_before_model = [
        "exec",
        *access_args,
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        "-o",
        "{response_file}",
    ]
    if oss:
        args_before_model.append("--oss")
        if local_provider:
            args_before_model.extend(["--local-provider", local_provider])

    return AgentCLIConfig(
        name="codex-oss" if oss else "codex",
        program="codex",
        args_before_model=tuple(args_before_model),
        args_after_model=("-",),
        model_flag=("-m",),
    )


def claude_cli_config() -> AgentCLIConfig:
    """Return a non-interactive Claude Code configuration."""
    return AgentCLIConfig(
        name="claude",
        program="claude",
        args_before_model=(
            "-p",
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
        ),
        model_flag=("--model",),
    )


def _invoke_agent(
    *,
    cmd: list[str],
    prompt: str | None,
    work_dir: str,
    timeout: int,
    run_env: dict[str, str],
) -> tuple[str, str, int | None, bool]:
    """Run an agent CLI and return stdout, stderr, return_code, timed_out."""
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
            cwd=work_dir,
        )
        return proc.stdout or "", proc.stderr or "", proc.returncode, False
    except subprocess.TimeoutExpired as e:
        return _to_text(e.stdout), _to_text(e.stderr), None, True


def _load_run_artifacts(
    *, work_dir: str, response_path: str, raw_stdout: str
) -> tuple[str, dict | None, str, str]:
    """Load all artifacts expected from an agent run."""
    code = _read(os.path.join(work_dir, "solution.py"))
    result_data = _read_json(os.path.join(work_dir, "result.json"))
    exploration = _read(os.path.join(work_dir, "exploration.md")).strip()
    hypothesis = _read(response_path).strip()
    if not hypothesis:
        hypothesis = _extract_response_text(raw_stdout).strip()
        if hypothesis:
            _write(response_path, hypothesis)
    return code, result_data, exploration, hypothesis


def _metric_from_result(result_data: dict | None) -> float | None:
    if not result_data:
        return None
    return _as_float(result_data.get("metric"))


def _build_command(
    work_dir: str,
    response_path: str,
    *,
    model: str | None,
    cli: AgentCLIConfig | None = None,
    prompt: str | None = None,
) -> list[str]:
    """Build an agent CLI invocation."""
    cli_config = cli or codex_cli_config()
    cmd = [cli_config.program]
    cmd.extend(
        _format_args(
            cli_config.args_before_model,
            work_dir=work_dir,
            response_path=response_path,
        )
    )
    if model:
        cmd.extend(
            _format_args(
                cli_config.model_flag,
                work_dir=work_dir,
                response_path=response_path,
            )
        )
        cmd.append(model)
    cmd.extend(
        _format_args(
            cli_config.args_after_model,
            work_dir=work_dir,
            response_path=response_path,
        )
    )
    if cli_config.prompt_mode == "arg":
        if prompt is None:
            raise ValueError("prompt is required when prompt_mode='arg'")
        cmd.extend(
            _format_args(
                cli_config.prompt_flag,
                work_dir=work_dir,
                response_path=response_path,
            )
        )
        cmd.append(prompt)
    return cmd


def _build_run_env(*, cli: AgentCLIConfig | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("LOKY_MAX_CPU_COUNT", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    if cli:
        for key, value in cli.env.items():
            env[key] = value
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
    runner_name: str,
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
        return f"{runner_name} timed out after {timeout}s."
    if return_code not in (None, 0):
        err_line = _last_nonempty_line(stderr)
        if err_line:
            return f"{runner_name} failed (exit {return_code}): {err_line[:240]}"
        return f"{runner_name} failed with exit code {return_code}."
    if fallback_note:
        return fallback_note
    return f"{runner_name} did not produce a hypothesis."


def _format_args(
    args: tuple[str, ...],
    *,
    work_dir: str,
    response_path: str,
) -> list[str]:
    response_file = os.path.basename(response_path)
    return [
        arg.format(
            work_dir=work_dir,
            response_path=response_path,
            response_file=response_file,
        )
        for arg in args
    ]


def _extract_response_text(stdout: str) -> str:
    stripped = stdout.strip()
    if not stripped:
        return ""

    payloads: list[object] = []
    try:
        payloads.append(json.loads(stripped))
    except json.JSONDecodeError:
        pass

    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    for payload in payloads:
        text = _extract_text_from_payload(payload)
        if text:
            return text.strip()

    return stripped


def _extract_text_from_payload(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts = [_extract_text_from_payload(item) for item in payload]
        return "\n".join(part for part in parts if part)
    if not isinstance(payload, dict):
        return ""

    if payload.get("type") == "text":
        text = payload.get("text")
        return text if isinstance(text, str) else ""

    for key in ("result", "output", "message", "content", "text", "response"):
        if key in payload:
            text = _extract_text_from_payload(payload[key])
            if text:
                return text

    return ""


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
