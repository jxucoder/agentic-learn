"""Tests for Journal edge cases."""

from __future__ import annotations

from aglearn.journal import Experiment, Journal


def test_journal_accepts_relative_filename_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    journal = Journal("journal.jsonl")
    journal.add(Experiment(hypothesis="ok", metric_value=1.0))

    assert (tmp_path / "journal.jsonl").exists()


def test_journal_ignores_non_finite_metrics():
    journal = Journal()
    journal.add(Experiment(hypothesis="nan", metric_value=float("nan")))
    journal.add(Experiment(hypothesis="inf", metric_value=float("inf")))
    journal.add(Experiment(hypothesis="best", metric_value=0.9))

    best = journal.best()

    assert best is not None
    assert best.hypothesis == "best"
    assert "nan" not in journal.summary()
    assert "inf" not in journal.summary()
