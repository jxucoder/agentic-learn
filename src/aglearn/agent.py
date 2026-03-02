"""Run a coding agent (codex CLI) to produce a solution."""

from __future__ import annotations

import json
import os
import subprocess


def run(
    prompt: str,
    work_dir: str,
    *,
    model: str | None = None,
    timeout: int = 300,
) -> dict:
    """Invoke ``codex exec`` and return the result.

    The agent works inside *work_dir*: it explores data, writes
    ``solution.py``, runs it, and writes ``result.json``.

    The full session trace (thinking, commands, outputs) is saved
    to ``trace.jsonl`` for research reproducibility.

    Returns dict with keys: code, hypothesis, metric_value, is_buggy.
    """
    response_path = os.path.join(work_dir, ".agent_response.md")
    trace_path = os.path.join(work_dir, "trace.jsonl")

    cmd = [
        "codex",
        "exec",
        "--full-auto",
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        "-C",
        work_dir,
        "-o",
        response_path,
    ]
    if model:
        cmd.extend(["-m", model])
    cmd.append("-")  # read prompt from stdin

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Save full session trace
        if proc.stdout:
            with open(trace_path, "w") as f:
                f.write(proc.stdout)
    except subprocess.TimeoutExpired as e:
        if e.stdout:
            with open(trace_path, "w") as f:
                f.write(e.stdout if isinstance(e.stdout, str) else e.stdout.decode())

    code = _read(os.path.join(work_dir, "solution.py"))
    result_data = _read_json(os.path.join(work_dir, "result.json"))
    hypothesis = _read(response_path).strip()
    exploration = _read(os.path.join(work_dir, "exploration.md")).strip()

    metric = None
    if result_data:
        metric = result_data.get("metric")

    return {
        "code": code,
        "hypothesis": hypothesis or "",
        "exploration": exploration,
        "metric_value": float(metric) if metric is not None else None,
        "is_buggy": metric is None,
    }


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _read_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
