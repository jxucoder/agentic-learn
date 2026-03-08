"""Tests for evolve loop state handling."""

from __future__ import annotations

import json
from pathlib import Path

import aglearn.runtime.loop as loop


def test_evolve_starts_from_clean_output_dir(tmp_path: Path, monkeypatch):
    (tmp_path / "journal.jsonl").write_text(
        json.dumps(
            {
                "id": "old",
                "code": "print('old')\n",
                "hypothesis": "old run",
                "exploration": "",
                "metric_value": 0.99,
                "is_buggy": False,
                "stdout": "",
                "stderr": "",
                "created_at": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "best_solution.py").write_text(
        "print('stale best')\n", encoding="utf-8"
    )
    (tmp_path / "report.md").write_text("stale report\n", encoding="utf-8")
    stale_step = tmp_path / "step_999"
    stale_step.mkdir()
    (stale_step / "result.json").write_text('{"metric": 0.99}\n', encoding="utf-8")

    prompts: list[str] = []

    def fake_run(
        prompt: str,
        work_dir: str,
        *,
        model: str | None = None,
        timeout: int = loop.DEFAULT_TIMEOUT_SECONDS,
        cli=None,
    ) -> dict[str, object]:
        del model, timeout, cli
        prompts.append(prompt)
        work_path = Path(work_dir)
        if work_path.name == "_report":
            (work_path / "report.md").write_text("# Fresh report\n", encoding="utf-8")
            return {
                "code": "",
                "hypothesis": "",
                "exploration": "",
                "metric_value": None,
                "is_buggy": True,
                "stdout": "",
                "stderr": "",
            }

        code = "print('fresh best')\n"
        (work_path / "solution.py").write_text(code, encoding="utf-8")
        (work_path / "result.json").write_text('{"metric": 0.4}\n', encoding="utf-8")
        (work_path / "exploration.md").write_text(
            "fresh exploration\n", encoding="utf-8"
        )
        return {
            "code": code,
            "hypothesis": "fresh run",
            "exploration": "fresh exploration",
            "metric_value": 0.4,
            "is_buggy": False,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(loop.agent, "run", fake_run)

    best = loop.evolve(
        loop.TaskConfig(
            description="demo",
            data_path="/tmp/data.csv",
            target_column="target",
            metric="accuracy",
        ),
        max_steps=1,
        output_dir=str(tmp_path),
    )

    assert best is not None
    assert best.metric_value == 0.4
    assert best.hypothesis == "fresh run"
    assert "No successful experiments yet." in prompts[0]
    assert not stale_step.exists()
    assert (tmp_path / "best_solution.py").read_text(
        encoding="utf-8"
    ) == "print('fresh best')\n"
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == "# Fresh report\n"

    journal_lines = (
        (tmp_path / "journal.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(journal_lines) == 1


def test_evolve_can_use_external_evaluator(tmp_path: Path, monkeypatch):
    def fake_run(
        prompt: str,
        work_dir: str,
        *,
        model: str | None = None,
        timeout: int = loop.DEFAULT_TIMEOUT_SECONDS,
        cli=None,
    ) -> dict[str, object]:
        del prompt, model, timeout, cli
        work_path = Path(work_dir)
        if work_path.name == "_report":
            return {
                "code": "",
                "hypothesis": "",
                "exploration": "",
                "metric_value": None,
                "is_buggy": True,
                "stdout": "",
                "stderr": "",
            }
        (work_path / "solution.py").write_text("print('x')\n", encoding="utf-8")
        (work_path / "submission.csv").write_text(
            "row_id,target\n1,1\n", encoding="utf-8"
        )
        return {
            "code": "print('x')\n",
            "hypothesis": "self reported",
            "exploration": "",
            "metric_value": 0.1,
            "is_buggy": False,
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(loop.agent, "run", fake_run)

    best = loop.evolve(
        loop.TaskConfig(
            description="demo",
            data_path="/tmp/data.csv",
            target_column="target",
            metric="f1",
            resource_paths={"test_data": "/tmp/test.csv"},
        ),
        max_steps=1,
        output_dir=str(tmp_path),
        evaluator=lambda work_dir, result: loop.EvaluationResult(
            metric_value=0.9,
            is_buggy=False,
        ),
    )

    assert best is not None
    assert best.metric_value == 0.9
