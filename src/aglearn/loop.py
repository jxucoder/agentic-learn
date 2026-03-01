"""The evolve loop.

Each step: brief the agent → agent does everything → record result.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from . import agent
from .journal import Experiment, Journal

log = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Human-provided problem specification."""

    description: str
    data_path: str
    target_column: str
    metric: str = "accuracy"
    instructions: str = ""  # optional human steering


def evolve(
    task: TaskConfig,
    *,
    model: str | None = None,
    max_steps: int = 10,
    timeout: int = 300,
    output_dir: str = "./output",
) -> Experiment | None:
    """Run the evolve loop."""
    os.makedirs(output_dir, exist_ok=True)
    journal = Journal(os.path.join(output_dir, "journal.jsonl"))

    log.info("evolve | task=%s model=%s steps=%d", task.description[:60], model, max_steps)

    for step in range(max_steps):
        log.info("step %d/%d", step + 1, max_steps)

        work_dir = os.path.join(output_dir, f"step_{step:03d}")
        os.makedirs(work_dir, exist_ok=True)

        prompt = _briefing(task, journal)
        result = agent.run(prompt, work_dir, model=model, timeout=timeout)

        exp = Experiment(
            code=result["code"],
            hypothesis=result["hypothesis"],
            metric_value=result["metric_value"],
            is_buggy=result["is_buggy"],
        )
        journal.add(exp)

        if exp.is_buggy:
            log.warning("  BUGGY | %s", (exp.hypothesis or "unknown")[:200])
        else:
            log.info("  score=%.4f | %s", exp.metric_value, exp.hypothesis[:120])
            _save_best(journal, output_dir)

    best = journal.best()
    if best:
        log.info("done | best=%.4f  experiments=%d", best.metric_value, journal.count())
    else:
        log.warning("done | no successful solutions after %d steps", max_steps)
    return best


def _briefing(task: TaskConfig, journal: Journal) -> str:
    parts = [
        f"You are an ML engineer. Your task:\n{task.description}",
        f"Data: {task.data_path}\n"
        f"Target column: {task.target_column}\n"
        f"Metric: {task.metric} (higher is better)",
    ]

    if task.instructions:
        parts.append(f"Additional instructions: {task.instructions}")

    parts.append(
        f"What has been tried so far (best first):\n{journal.summary()}"
    )

    parts.append(
        "Do the following:\n"
        "1. Explore the data, then write a scikit-learn solution as solution.py\n"
        '2. Run solution.py — it must print {"metric": <float>} as its only stdout line\n'
        '3. Save {"metric": <float>} to result.json\n'
        "4. Try something meaningfully different from what's in the journal"
    )

    return "\n\n".join(parts)


def _save_best(journal: Journal, output_dir: str) -> None:
    best = journal.best()
    if best is None:
        return
    with open(os.path.join(output_dir, "best_solution.py"), "w") as f:
        f.write(best.code)
