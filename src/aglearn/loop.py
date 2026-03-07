"""The evolve loop.

Each step: brief the agent → agent does everything → record result.
After all steps: agent analyzes all artifacts and writes a research report.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

from . import agent
from .journal import Experiment, Journal

log = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS = 600


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
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    output_dir: str = "./output",
) -> Experiment | None:
    """Run the evolve loop."""
    _prepare_output_dir(output_dir)
    journal = Journal(os.path.join(output_dir, "journal.jsonl"))

    log.info(
        "evolve | task=%s model=%s steps=%d", task.description[:60], model, max_steps
    )

    for step in range(max_steps):
        log.info("step %d/%d", step + 1, max_steps)

        work_dir = os.path.join(output_dir, f"step_{step:03d}")
        os.makedirs(work_dir, exist_ok=True)

        prompt = _briefing(task, journal)
        result = agent.run(prompt, work_dir, model=model, timeout=timeout)

        exp = Experiment(
            code=result["code"],
            hypothesis=result["hypothesis"],
            exploration=result.get("exploration", ""),
            metric_value=result["metric_value"],
            is_buggy=result["is_buggy"],
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
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

    _generate_report(task, journal, output_dir, model=model, timeout=timeout)

    return best


# ---------------------------------------------------------------------------
# Experiment briefing
# ---------------------------------------------------------------------------


def _briefing(task: TaskConfig, journal: Journal) -> str:
    parts = [
        f"You are an ML engineer. Your task:\n{task.description}",
        f"Data: {task.data_path}\n"
        f"Target column: {task.target_column}\n"
        f"Metric: {task.metric} (higher is better)",
    ]

    if task.instructions:
        parts.append(f"Additional instructions: {task.instructions}")

    parts.append(f"What has been tried so far (best first):\n{journal.summary()}")

    parts.append(
        "Do the following:\n"
        "1. Explore the data — look at distributions, correlations, missing values\n"
        "2. Write a scikit-learn solution as solution.py\n"
        '3. Run solution.py — it must print {"metric": <float>} as its only stdout line\n'
        '4. Save {"metric": <float>} to result.json\n'
        "5. Try something meaningfully different from what's in the journal\n"
        "6. Write exploration.md documenting:\n"
        "   - Key patterns and distributions you found in the data\n"
        "   - Feature engineering decisions and why (what signal did you see?)\n"
        "   - Model choice rationale\n"
        "   - What you tried that didn't work and why\n"
        "7. Environment constraints:\n"
        "   - Use single-process execution only (`n_jobs=1`; avoid multiprocessing)\n"
        "   - Avoid process-management shell commands (`ps`, `pkill`, etc.)\n"
        "   - Prefer pandas-compatible APIs (avoid fragile version-specific args)\n"
        "   - If any step fails, still persist as many artifacts as possible"
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(
    task: TaskConfig,
    journal: Journal,
    output_dir: str,
    *,
    model: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Run the agent to analyze all artifacts and write a research report."""
    if journal.count() == 0:
        log.info("report | skipped (no experiments)")
        return

    log.info("report | generating…")

    report_work_dir, report_path = _prepare_report_workspace(output_dir)

    prompt = _report_briefing(task, journal, output_dir)
    agent.run(prompt, report_work_dir, model=model, timeout=timeout)

    # Move report.md to output root if the agent wrote it
    src = os.path.join(report_work_dir, "report.md")
    if os.path.exists(src):
        shutil.move(src, report_path)
        log.info("report | wrote %s", report_path)
        _convert_to_pdf(report_path)
    else:
        _write_fallback_report(task, journal, report_path)
        log.warning("report | agent did not produce report.md; wrote fallback report")


def _report_briefing(task: TaskConfig, journal: Journal, output_dir: str) -> str:
    step_listing = _step_artifact_listing(output_dir)

    parts = [
        "You are a research analyst writing a report on an ML experiment run.",
        f"Task that was optimized:\n"
        f"  {task.description}\n"
        f"  Data: {task.data_path}\n"
        f"  Target: {task.target_column}\n"
        f"  Metric: {task.metric} (higher is better)",
        f"Experiment journal (all experiments, best first):\n{journal.summary()}",
        "Artifact locations — read these files to write your analysis:\n"
        f"  Journal: {os.path.join(output_dir, 'journal.jsonl')}\n"
        f"  Best solution: {os.path.join(output_dir, 'best_solution.py')}\n"
        "  Step directories (each may contain solution.py, result.json, exploration.md, trace.jsonl):\n"
        + "\n".join(step_listing),
        "Some steps may be incomplete or missing files. Call out missing artifacts explicitly.",
        "Write report.md — a detailed research report covering:\n"
        "\n"
        "1. **Summary** — task description, number of steps, metric trajectory,\n"
        "   best result achieved\n"
        "\n"
        "2. **Evolution trajectory** — table of ALL steps: step number, score\n"
        "   (or BUGGY), model used, key change from previous step.\n"
        "   Read each step's solution.py and result.json.\n"
        "\n"
        "3. **Step-by-step analysis** — for EACH step, read the solution.py,\n"
        "   exploration.md, and trace.jsonl. Describe:\n"
        "   - What approach the agent took and why\n"
        "   - What feature engineering was applied\n"
        "   - What model and hyperparameters were chosen\n"
        "   - What worked and what didn't\n"
        "   - Key reasoning from the trace\n"
        "\n"
        "4. **Best solution analysis** — detailed walkthrough of the winning\n"
        "   code from best_solution.py. Explain the pipeline, features,\n"
        "   model, and why it outperformed the others.\n"
        "\n"
        "5. **Failure analysis** — for buggy steps, read the trace.jsonl to\n"
        "   understand what went wrong.\n"
        "\n"
        "6. **Conclusions** — patterns observed across the evolution,\n"
        "   what the agent learned (or failed to learn), insights about\n"
        "   the dataset and task.\n"
        "\n"
        "Be specific — include actual metric values, code snippets, feature\n"
        "names, and model parameters. This is a research artifact, not a\n"
        "summary. Write it as report.md.",
    ]

    return "\n\n".join(parts)


def _write_fallback_report(
    task: TaskConfig, journal: Journal, report_path: str
) -> None:
    """Write a minimal fresh report when report generation fails."""
    best = journal.best()
    best_metric = f"{best.metric_value:.4f}" if best else "N/A"
    contents = (
        "# ML Experiment Research Report\n\n"
        "## Report Status\n"
        "The agent did not generate `report.md` for this run.\n"
        "This fallback report is auto-generated to avoid stale report artifacts.\n\n"
        "## Run Summary\n"
        f"- Metric optimized: `{task.metric}`\n"
        f"- Total experiments: `{journal.count()}`\n"
        f"- Best metric: `{best_metric}`\n\n"
        "## Successful Experiments (Best First)\n"
        f"{journal.summary()}\n"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(contents)


def _prepare_report_workspace(output_dir: str) -> tuple[str, str]:
    report_work_dir = os.path.join(output_dir, "_report")
    report_path = os.path.join(output_dir, "report.md")
    shutil.rmtree(report_work_dir, ignore_errors=True)
    os.makedirs(report_work_dir, exist_ok=True)
    if os.path.exists(report_path):
        os.remove(report_path)
    return report_work_dir, report_path


def _prepare_output_dir(output_dir: str) -> None:
    """Remove generated artifacts so each evolve() call starts fresh."""
    os.makedirs(output_dir, exist_ok=True)

    generated_files = {"journal.jsonl", "best_solution.py", "report.md", "report.pdf"}
    generated_dirs = {"_report"}

    for entry in os.listdir(output_dir):
        path = os.path.join(output_dir, entry)
        if entry in generated_files and os.path.isfile(path):
            os.remove(path)
        elif entry in generated_dirs and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif entry.startswith("step_"):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)


def _step_artifact_listing(output_dir: str) -> list[str]:
    step_dirs = sorted(
        d
        for d in os.listdir(output_dir)
        if d.startswith("step_") and os.path.isdir(os.path.join(output_dir, d))
    )
    listing = []
    for d in step_dirs:
        step_path = os.path.join(output_dir, d)
        files = os.listdir(step_path)
        listing.append(f"  {step_path}/  ({', '.join(sorted(files))})")
    return listing


def _convert_to_pdf(md_path: str) -> None:
    """Best-effort conversion of report.md to report.pdf via pandoc."""
    if not shutil.which("pandoc"):
        log.info("report | pandoc not found, skipping PDF conversion")
        return

    pdf_path = md_path.replace(".md", ".pdf")
    try:
        subprocess.run(
            [
                "pandoc",
                md_path,
                "-o",
                pdf_path,
                "--pdf-engine=xelatex",
                "-V",
                "geometry:margin=1in",
                "-V",
                "fontsize=11pt",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        log.info("report | wrote %s", pdf_path)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ) as e:
        log.warning("report | PDF conversion failed: %s", e)


def _save_best(journal: Journal, output_dir: str) -> None:
    best = journal.best()
    if best is None:
        return
    with open(os.path.join(output_dir, "best_solution.py"), "w", encoding="utf-8") as f:
        f.write(best.code)
