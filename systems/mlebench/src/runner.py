"""Script execution runner for MLE-bench.

Executes generated Python scripts, captures output, and parses results.
Runs inside the MLE-bench Docker container -- no additional sandboxing needed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a solution script."""

    # Status
    success: bool = False
    return_code: int = -1

    # Output
    stdout: str = ""
    stderr: str = ""

    # Error info
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timed_out: bool = False

    # Score (parsed from stdout)
    score: Optional[float] = None
    score_label: str = ""

    # Submission
    submission_created: bool = False
    submission_path: Optional[str] = None

    # Timing
    execution_time_seconds: float = 0.0

    # Script info
    script_path: Optional[str] = None

    def get_summary(self) -> str:
        """Get a brief summary of the result."""
        if self.timed_out:
            return f"TIMEOUT after {self.execution_time_seconds:.0f}s"
        if not self.success:
            return f"FAILED: {self.error_type or 'Unknown'}: {self.error_message or ''}"
        parts = [f"OK ({self.execution_time_seconds:.0f}s)"]
        if self.score is not None:
            parts.append(f"score={self.score:.6f}")
        if self.submission_created:
            parts.append("submission=OK")
        else:
            parts.append("submission=MISSING")
        return " | ".join(parts)


def _parse_score(stdout: str) -> tuple[Optional[float], str]:
    """Parse a score from stdout.

    Looks for common patterns like:
        CV Score: 0.8456
        Accuracy: 0.92
        RMSE: 1.234
        score: 0.95
        validation score: 0.87

    Returns (score_value, score_label).
    """
    # Patterns ordered by specificity
    patterns = [
        r"(?:CV|cross.?val(?:idation)?)\s*(?:score|accuracy|auc|f1)?\s*[:=]\s*([\d.]+)",
        r"(?:validation|val)\s*(?:score|accuracy|auc|f1|rmse|mae|loss)\s*[:=]\s*([\d.]+)",
        r"(?:best|final)\s*(?:score|accuracy|auc|f1|rmse|mae)\s*[:=]\s*([\d.]+)",
        r"(?:test|eval)\s*(?:score|accuracy|auc|f1|rmse|mae)\s*[:=]\s*([\d.]+)",
        r"(?:accuracy|auc|f1.?score|rmse|mae|log.?loss|r2|mse)\s*[:=]\s*([\d.]+)",
        r"score\s*[:=]\s*([\d.]+)",
    ]

    # Search from the end of stdout (most recent output)
    lines = stdout.strip().split("\n")
    for line in reversed(lines):
        line_lower = line.lower().strip()
        for pattern in patterns:
            match = re.search(pattern, line_lower)
            if match:
                try:
                    score = float(match.group(1))
                    # Sanity check -- scores should be reasonable
                    if -1000 < score < 1000:
                        label = line.strip()[:100]
                        return score, label
                except ValueError:
                    continue

    return None, ""


def _parse_error(stderr: str) -> tuple[Optional[str], Optional[str]]:
    """Parse error type and message from stderr."""
    if not stderr:
        return None, None

    lines = stderr.strip().split("\n")

    for line in reversed(lines):
        line = line.strip()
        if ": " in line:
            parts = line.split(": ", 1)
            if any(err in parts[0] for err in ("Error", "Exception")):
                error_type = parts[0].split(".")[-1].strip()
                error_msg = parts[1] if len(parts) > 1 else ""
                return error_type, error_msg

    # Fallback: last non-empty line
    for line in reversed(lines):
        if line.strip():
            return "RuntimeError", line.strip()

    return None, None


def execute_script(
    code: str,
    code_dir: str = "/home/code",
    submission_dir: str = "/home/submission",
    timeout: float = 32400,  # 9 hours default (like AIDE)
    python_path: Optional[str] = None,
    script_name: str = "solution.py",
    iteration: int = 0,
) -> ExecutionResult:
    """Execute a Python solution script.

    Args:
        code: The Python code to execute.
        code_dir: Directory to write the script to.
        submission_dir: Directory where submission.csv should appear.
        timeout: Maximum execution time in seconds.
        python_path: Path to Python interpreter.
        script_name: Name for the script file.
        iteration: Current iteration number (for naming).

    Returns:
        ExecutionResult with execution details.
    """
    result = ExecutionResult()
    python = python_path or sys.executable

    # Create directories
    code_path = Path(code_dir)
    sub_path = Path(submission_dir)
    code_path.mkdir(parents=True, exist_ok=True)
    sub_path.mkdir(parents=True, exist_ok=True)

    # Write script
    script_file = code_path / script_name
    script_file.write_text(code)
    result.script_path = str(script_file)

    # Also save a versioned copy for debugging
    versioned_file = code_path / f"solution_v{iteration}.py"
    versioned_file.write_text(code)

    logger.info(f"Executing {script_file} (timeout={timeout}s)")

    # Execute
    start_time = time.time()
    try:
        process = subprocess.run(
            [python, str(script_file)],
            cwd=str(code_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
            },
        )

        result.return_code = process.returncode
        result.stdout = process.stdout
        result.stderr = process.stderr
        result.success = process.returncode == 0

    except subprocess.TimeoutExpired as e:
        result.timed_out = True
        result.success = False
        result.error_type = "TimeoutError"
        result.error_message = f"Execution timed out after {timeout}s"
        result.stdout = (e.stdout or b"").decode("utf-8", errors="replace")
        result.stderr = (e.stderr or b"").decode("utf-8", errors="replace")

    except Exception as e:
        result.success = False
        result.error_type = type(e).__name__
        result.error_message = str(e)

    result.execution_time_seconds = time.time() - start_time

    # Parse error if failed
    if not result.success and not result.timed_out:
        result.error_type, result.error_message = _parse_error(result.stderr)

    # Parse score from stdout
    result.score, result.score_label = _parse_score(result.stdout)

    # Check for submission file
    submission_csv = sub_path / "submission.csv"
    if submission_csv.exists():
        result.submission_created = True
        result.submission_path = str(submission_csv)
    else:
        # Also check if the script saved to the code directory
        alt_submission = code_path / "submission.csv"
        if alt_submission.exists():
            # Copy to the correct location
            shutil.copy2(str(alt_submission), str(submission_csv))
            result.submission_created = True
            result.submission_path = str(submission_csv)
            logger.info("Copied submission.csv from code dir to submission dir")

    logger.info(f"Execution result: {result.get_summary()}")

    return result
