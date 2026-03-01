"""Experiment Journal — append-only shared memory.

Stores every experiment the agent has tried. Persisted as JSON lines.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class Experiment:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    code: str = ""
    hypothesis: str = ""
    metric_value: float | None = None
    is_buggy: bool = False
    stdout: str = ""
    stderr: str = ""
    created_at: float = field(default_factory=time.time)


class Journal:
    """Append-only experiment journal backed by a JSON-lines file."""

    def __init__(self, path: str | None = None):
        self._experiments: list[Experiment] = []
        self._path = path
        if path and os.path.exists(path):
            self._load()

    def add(self, exp: Experiment) -> None:
        self._experiments.append(exp)
        if self._path:
            self._append(exp)

    def best(self) -> Experiment | None:
        good = self._good()
        return good[0] if good else None

    def count(self) -> int:
        return len(self._experiments)

    def summary(self) -> str:
        """Full metric-sorted summary for LLM context."""
        good = self._good()
        if not good:
            return "No successful experiments yet."
        lines = []
        for i, e in enumerate(good, 1):
            lines.append(f"#{i}  score={e.metric_value:.4f}  —  {e.hypothesis}")
        return "\n".join(lines)

    # ---- persistence (JSON lines) ----

    def _append(self, exp: Experiment) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(exp)) + "\n")

    def _load(self) -> None:
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._experiments.append(Experiment(**json.loads(line)))

    def _good(self) -> list[Experiment]:
        return sorted(
            [e for e in self._experiments if not e.is_buggy and e.metric_value is not None],
            key=lambda e: e.metric_value,
            reverse=True,
        )
