#!/usr/bin/env python3
"""MLE-bench agent: main entry point.

Implements the greedy iteration loop:
  parse task -> generate baseline -> execute -> validate -> improve -> repeat

Usage (inside MLE-bench container):
    python src/agent.py --data-dir /home/data --submission-dir /home/submission

Usage (local development):
    python src/agent.py --data-dir ./test_data --submission-dir ./test_submission
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task_parser import parse_task, TaskInfo
from src.prompts import (
    SYSTEM_PROMPT,
    build_initial_prompt,
    build_improvement_prompt,
    build_error_fix_prompt,
)
from src.runner import execute_script, ExecutionResult
from src.validator import validate_submission, check_submission_basic
from src.budget import BudgetManager, Phase
from src.llm import LLMClient

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mlebench-agent")


# =============================================================================
# Agent State
# =============================================================================

class AgentState:
    """Tracks the agent's state across iterations."""

    def __init__(self) -> None:
        self.iteration: int = 0
        self.best_score: Optional[float] = None
        self.best_code: str = ""
        self.best_submission_path: Optional[str] = None
        self.higher_is_better: bool = True
        self.history: list[dict] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.consecutive_errors: int = 0

    def is_improvement(self, score: float) -> bool:
        """Check if a score is an improvement over the current best."""
        if self.best_score is None:
            return True
        if self.higher_is_better:
            return score > self.best_score
        else:
            return score < self.best_score

    def record(
        self,
        code: str,
        result: ExecutionResult,
        is_improvement: bool,
    ) -> None:
        """Record an iteration result."""
        self.iteration += 1
        entry = {
            "iteration": self.iteration,
            "success": result.success,
            "score": result.score,
            "score_label": result.score_label,
            "submission_created": result.submission_created,
            "execution_time": result.execution_time_seconds,
            "error_type": result.error_type,
            "is_improvement": is_improvement,
        }
        self.history.append(entry)

        if result.success and not result.error_type:
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1

    def summary(self) -> str:
        """Get a summary of the agent's progress."""
        lines = [
            f"Iterations: {self.iteration}",
            f"Best score: {self.best_score}",
            f"Consecutive errors: {self.consecutive_errors}",
            f"Total tokens: {self.total_input_tokens}in / {self.total_output_tokens}out",
        ]
        return " | ".join(lines)


# =============================================================================
# Main Agent Loop
# =============================================================================

def run_agent(
    data_dir: str = "/home/data",
    submission_dir: str = "/home/submission",
    code_dir: str = "/home/code",
    logs_dir: str = "/home/logs",
    time_limit: float = 86400,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_steps: int = 500,
    step_timeout: float = 32400,
) -> None:
    """Run the MLE-bench agent.

    Args:
        data_dir: Path to competition data.
        submission_dir: Path to write submission.csv.
        code_dir: Path to write solution scripts.
        logs_dir: Path to write logs.
        time_limit: Total time budget in seconds.
        provider: LLM provider (anthropic, openai).
        model: LLM model name.
        api_key: API key (or use env var).
        base_url: Base URL for OpenAI-compatible APIs.
        max_steps: Maximum number of iterations.
        step_timeout: Maximum time per step in seconds.
    """
    # Initialize
    budget = BudgetManager(total_seconds=time_limit)
    budget.start()

    state = AgentState()

    logger.info("=" * 60)
    logger.info("agentic-learn MLE-bench Agent")
    logger.info(f"Data: {data_dir}")
    logger.info(f"Submission: {submission_dir}")
    logger.info(f"Time limit: {time_limit}s")
    logger.info(f"Provider: {provider}, Model: {model}")
    logger.info("=" * 60)

    # =========================================================================
    # Phase 1: Parse task
    # =========================================================================
    budget.advance_phase(Phase.PARSING)
    logger.info("Phase: PARSING")

    task = parse_task(data_dir=data_dir, submission_dir=submission_dir)
    state.higher_is_better = task.higher_is_better

    logger.info(f"Task: {task.competition_id}")
    logger.info(f"Type: {task.task_type.value}")
    logger.info(f"Metric: {task.metric_name} ({'↑' if task.higher_is_better else '↓'})")
    logger.info(f"Train: {task.num_train_rows} rows")
    logger.info(f"Test: {task.num_test_rows} rows")
    logger.info(f"Files: {len(task.data_files)}")

    # Initialize LLM client
    llm = LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    # =========================================================================
    # Phase 2: Generate baseline
    # =========================================================================
    budget.advance_phase(Phase.BASELINE)
    logger.info("Phase: BASELINE")

    # Try to use sample submission as immediate fallback
    _submit_sample_as_fallback(task, submission_dir)

    # Generate baseline solution
    prompt = build_initial_prompt(task)
    code = llm.generate_code(prompt, system=SYSTEM_PROMPT)

    if not code:
        logger.error("LLM returned empty code, retrying...")
        code = llm.generate_code(prompt, system=SYSTEM_PROMPT)

    if code:
        timeout = budget.step_timeout(max_timeout=step_timeout)
        result = execute_script(
            code=code,
            code_dir=code_dir,
            submission_dir=submission_dir,
            timeout=timeout,
            iteration=0,
        )

        _process_result(state, task, code, result, submission_dir, budget)

        # If baseline failed, try a simpler approach
        if not result.success or not result.submission_created:
            logger.info("Baseline failed, trying simpler approach...")
            code = _generate_simple_baseline(task)
            result = execute_script(
                code=code,
                code_dir=code_dir,
                submission_dir=submission_dir,
                timeout=min(600, timeout),  # 10 min max for simple baseline
                iteration=0,
            )
            _process_result(state, task, code, result, submission_dir, budget)

    # =========================================================================
    # Phase 3: Iterative improvement (greedy search)
    # =========================================================================
    budget.advance_phase(Phase.SEARCH)
    logger.info("Phase: SEARCH")

    previous_code = state.best_code or code
    previous_result: Optional[ExecutionResult] = None

    for step in range(1, max_steps):
        # Check budget
        if budget.should_stop():
            logger.info("Time budget exhausted, stopping.")
            break

        if not budget.in_phase(Phase.SEARCH) and not budget.in_phase(Phase.REFINEMENT):
            logger.info("Search phase complete, moving to refinement.")
            break

        # Stop if too many consecutive errors
        if state.consecutive_errors >= 5:
            logger.warning("Too many consecutive errors, resetting...")
            state.consecutive_errors = 0
            previous_code = state.best_code  # Reset to best known working code

        logger.info(f"--- Iteration {step} | {budget.status()} ---")

        # Decide which prompt to use
        if previous_result and not previous_result.success:
            # Fix error
            error_output = previous_result.stderr or previous_result.error_message or "Unknown error"
            prompt = build_error_fix_prompt(
                task=task,
                previous_code=previous_code,
                error_output=error_output,
                iteration=step,
            )
        else:
            # Improve
            score_str = f"{state.best_score:.6f}" if state.best_score is not None else "N/A"
            prev_score = "N/A"
            if previous_result and previous_result.score is not None:
                prev_score = f"{previous_result.score:.6f}"
            prompt = build_improvement_prompt(
                task=task,
                previous_code=previous_code,
                previous_score=prev_score,
                previous_stdout=(previous_result.stdout if previous_result else ""),
                iteration=step,
                best_score=score_str,
            )

        # Generate improved code
        try:
            code = llm.generate_code(prompt, system=SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            time.sleep(5)  # Brief pause before retry
            continue

        if not code:
            logger.warning("Empty code generated, skipping iteration")
            continue

        # Execute
        timeout = budget.step_timeout(max_timeout=step_timeout)
        result = execute_script(
            code=code,
            code_dir=code_dir,
            submission_dir=submission_dir,
            timeout=timeout,
            iteration=step,
        )

        _process_result(state, task, code, result, submission_dir, budget)

        # Update state for next iteration
        previous_code = code
        previous_result = result

        budget.record_iteration()

    # =========================================================================
    # Finalize
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Agent finished")
    logger.info(state.summary())
    logger.info(budget.status())

    # Ensure best submission is in place
    if state.best_submission_path:
        final_path = Path(submission_dir) / "submission.csv"
        if state.best_submission_path != str(final_path):
            shutil.copy2(state.best_submission_path, str(final_path))
            logger.info(f"Restored best submission (score={state.best_score})")

    # Save run summary
    _save_summary(state, budget, task, logs_dir)

    logger.info("=" * 60)


# =============================================================================
# Helpers
# =============================================================================

def _process_result(
    state: AgentState,
    task: TaskInfo,
    code: str,
    result: ExecutionResult,
    submission_dir: str,
    budget: BudgetManager,
) -> None:
    """Process an execution result and update state."""
    is_improvement = False

    if result.success and result.submission_created:
        # Validate submission
        validation = validate_submission(
            submission_path=result.submission_path or f"{submission_dir}/submission.csv"
        )

        if validation.valid:
            budget.record_valid_submission()

            if result.score is not None and state.is_improvement(result.score):
                is_improvement = True
                state.best_score = result.score
                state.best_code = code
                state.best_submission_path = result.submission_path

                # Save best submission as backup
                backup = Path(submission_dir) / "best_submission.csv"
                if result.submission_path:
                    shutil.copy2(result.submission_path, str(backup))

                logger.info(f"NEW BEST: score={result.score:.6f}")

            elif result.score is None and not budget.has_valid_submission:
                # First valid submission (no score available)
                state.best_code = code
                state.best_submission_path = result.submission_path
                logger.info("First valid submission (no score parsed)")

        else:
            logger.warning(f"Submission invalid: {validation.error}")

    state.record(code, result, is_improvement)


def _submit_sample_as_fallback(task: TaskInfo, submission_dir: str) -> None:
    """Submit the sample submission as an immediate fallback.

    This ensures we always have at least something to submit.
    """
    if task.sample_submission_file:
        src = Path(task.sample_submission_file.path)
        dst = Path(submission_dir) / "submission.csv"
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            logger.info(f"Sample submission copied as fallback: {src.name}")


def _generate_simple_baseline(task: TaskInfo) -> str:
    """Generate a simple baseline script that should almost always work.

    Falls back to submitting the sample submission or a trivial prediction.
    """
    code_lines = [
        '"""Simple baseline -- produces a valid submission."""',
        "import pandas as pd",
        "import numpy as np",
        "import os",
        "",
        f'SUBMISSION_DIR = "{task.submission_dir}"',
        f'DATA_DIR = "{task.data_dir}"',
        "",
        "os.makedirs(SUBMISSION_DIR, exist_ok=True)",
        "",
    ]

    if task.sample_submission_file:
        # Use sample submission as starting point
        code_lines.extend([
            "# Load sample submission as baseline",
            f'sample = pd.read_csv("{task.sample_submission_file.path}")',
            "print(f'Sample submission shape: {sample.shape}')",
            "print(sample.head())",
            f'sample.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)',
            'print("Saved sample submission as baseline")',
        ])
    elif task.test_file and task.submission_columns:
        # Generate trivial predictions
        id_col = task.id_column or task.submission_columns[0]
        target_cols = task.target_columns or task.submission_columns[1:]

        code_lines.extend([
            f'test = pd.read_csv("{task.test_file.path}")',
            "print(f'Test shape: {test.shape}')",
            f'submission = pd.DataFrame()',
            f'submission["{id_col}"] = test["{id_col}"]',
        ])
        for col in target_cols:
            code_lines.append(f'submission["{col}"] = 0')

        code_lines.extend([
            f'submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)',
            'print("Saved trivial baseline submission")',
            "print(submission.head())",
        ])
    else:
        code_lines.extend([
            '# Cannot determine submission format -- list data files',
            "for f in os.listdir(DATA_DIR):",
            '    fpath = os.path.join(DATA_DIR, f)',
            '    if os.path.isfile(fpath):',
            '        size = os.path.getsize(fpath)',
            '        print(f"  {f}: {size} bytes")',
            'print("ERROR: Could not determine submission format")',
        ])

    return "\n".join(code_lines)


def _save_summary(
    state: AgentState,
    budget: BudgetManager,
    task: TaskInfo,
    logs_dir: str,
) -> None:
    """Save a JSON summary of the run."""
    summary = {
        "competition_id": task.competition_id,
        "task_type": task.task_type.value,
        "metric": task.metric_name,
        "higher_is_better": task.higher_is_better,
        "iterations": state.iteration,
        "best_score": state.best_score,
        "total_time": budget.elapsed,
        "total_input_tokens": state.total_input_tokens,
        "total_output_tokens": state.total_output_tokens,
        "has_valid_submission": budget.has_valid_submission,
        "history": state.history,
    }

    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    summary_file = logs_path / "run_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    logger.info(f"Run summary saved to {summary_file}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="agentic-learn MLE-bench Agent")

    parser.add_argument("--data-dir", default="/home/data", help="Competition data directory")
    parser.add_argument("--submission-dir", default="/home/submission", help="Submission output directory")
    parser.add_argument("--code-dir", default="/home/code", help="Code output directory")
    parser.add_argument("--logs-dir", default="/home/logs", help="Logs output directory")
    parser.add_argument("--time-limit", type=float, default=86400, help="Time budget in seconds")

    # LLM config
    parser.add_argument("--provider", default="anthropic", help="LLM provider")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible APIs")

    # Agent config
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum iterations")
    parser.add_argument("--step-timeout", type=float, default=32400, help="Timeout per step (seconds)")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    run_agent(
        data_dir=args.data_dir,
        submission_dir=args.submission_dir,
        code_dir=args.code_dir,
        logs_dir=args.logs_dir,
        time_limit=args.time_limit,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_steps=args.max_steps,
        step_timeout=args.step_timeout,
    )


if __name__ == "__main__":
    main()
