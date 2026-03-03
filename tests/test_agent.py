"""Unit tests for aglearn.agent helper behavior."""

from __future__ import annotations

import json
from pathlib import Path

from aglearn import agent


def test_build_command_defaults_to_bypass(monkeypatch):
    monkeypatch.delenv("AGLEARN_CODEX_ACCESS_MODE", raising=False)
    cmd = agent._build_command("/tmp/work", "/tmp/work/.agent_response.md", model=None)
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert "--full-auto" not in cmd


def test_build_command_full_auto_mode(monkeypatch):
    monkeypatch.setenv("AGLEARN_CODEX_ACCESS_MODE", "full-auto")
    cmd = agent._build_command("/tmp/work", "/tmp/work/.agent_response.md", model="foo")
    assert "--full-auto" in cmd
    assert "--sandbox" in cmd
    assert "danger-full-access" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
    assert cmd[-3:] == ["-m", "foo", "-"]


def test_metric_from_stdout_uses_last_json_metric():
    stdout = "\n".join(
        [
            "some logging",
            json.dumps({"metric": 0.12}),
            "more logging",
            json.dumps({"metric": 0.89}),
        ]
    )
    assert agent._metric_from_stdout(stdout) == 0.89


def test_run_solution_fallback_recovers_metric(tmp_path: Path):
    solution = tmp_path / "solution.py"
    solution.write_text(
        "import json\nprint(json.dumps({'metric': 0.77}))\n",
        encoding="utf-8",
    )

    metric, note = agent._run_solution_fallback(str(tmp_path), timeout=10)

    assert metric == 0.77
    assert "Recovered metric" in note
    result_payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result_payload["metric"] == 0.77
