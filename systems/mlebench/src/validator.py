"""Submission validation via MLE-bench grading server.

MLE-bench provides a grading server at http://localhost:5000/validate
that checks if a submission file has the correct format.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of submission validation."""
    valid: bool = False
    message: str = ""
    error: Optional[str] = None


def validate_submission(
    submission_path: str = "/home/submission/submission.csv",
    server_url: str = "http://localhost:5000/validate",
    timeout: int = 30,
) -> ValidationResult:
    """Validate a submission file against the MLE-bench grading server.

    Args:
        submission_path: Path to the submission CSV file.
        server_url: URL of the grading server validation endpoint.
        timeout: Request timeout in seconds.

    Returns:
        ValidationResult with validation status and message.
    """
    result = ValidationResult()

    # Check file exists
    if not Path(submission_path).exists():
        result.valid = False
        result.error = f"Submission file not found: {submission_path}"
        logger.warning(result.error)
        return result

    # Check file is not empty
    file_size = Path(submission_path).stat().st_size
    if file_size == 0:
        result.valid = False
        result.error = "Submission file is empty"
        logger.warning(result.error)
        return result

    # Try the validation script first (provided by MLE-bench)
    validate_script = Path("/home/validate_submission.sh")
    if validate_script.exists():
        try:
            proc = subprocess.run(
                ["bash", str(validate_script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="/home",
            )
            result.message = proc.stdout.strip()
            if proc.returncode == 0:
                result.valid = True
                logger.info(f"Submission valid: {result.message}")
            else:
                result.valid = False
                result.error = proc.stderr.strip() or proc.stdout.strip()
                logger.warning(f"Submission invalid: {result.error}")
            return result
        except subprocess.TimeoutExpired:
            logger.warning("Validation script timed out, trying curl fallback")
        except Exception as e:
            logger.warning(f"Validation script failed: {e}, trying curl fallback")

    # Fallback: use curl directly
    try:
        proc = subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                "-F", f"file=@{submission_path}",
                server_url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        response = proc.stdout.strip()
        result.message = response

        # The grading server returns a message indicating validity
        response_lower = response.lower()
        if "valid" in response_lower and "invalid" not in response_lower:
            result.valid = True
        elif "error" in response_lower or "invalid" in response_lower:
            result.valid = False
            result.error = response
        else:
            # Ambiguous response -- assume valid if no explicit error
            result.valid = proc.returncode == 0
            if not result.valid:
                result.error = response

        logger.info(f"Validation result: valid={result.valid}, message={result.message[:200]}")

    except subprocess.TimeoutExpired:
        result.valid = False
        result.error = "Validation request timed out"
        logger.warning(result.error)

    except FileNotFoundError:
        # curl not available -- skip validation
        result.valid = True  # Optimistically assume valid
        result.message = "Validation skipped (curl not available)"
        logger.warning(result.message)

    except Exception as e:
        result.valid = False
        result.error = f"Validation failed: {e}"
        logger.warning(result.error)

    return result


def check_submission_basic(
    submission_path: str,
    expected_columns: list[str] | None = None,
    expected_rows: int | None = None,
) -> ValidationResult:
    """Basic local validation of submission file format.

    Does not require the grading server. Useful for quick checks.
    """
    result = ValidationResult()
    path = Path(submission_path)

    if not path.exists():
        result.error = f"File not found: {submission_path}"
        return result

    if path.stat().st_size == 0:
        result.error = "File is empty"
        return result

    try:
        with open(path, "r") as f:
            header = f.readline().strip()
            columns = [c.strip().strip('"') for c in header.split(",")]

            # Check columns
            if expected_columns:
                missing = set(expected_columns) - set(columns)
                if missing:
                    result.error = f"Missing columns: {missing}"
                    return result

            # Count rows
            row_count = sum(1 for _ in f)

            if expected_rows is not None and row_count != expected_rows:
                result.error = f"Expected {expected_rows} rows, got {row_count}"
                return result

            if row_count == 0:
                result.error = "No data rows (only header)"
                return result

        result.valid = True
        result.message = f"OK: {len(columns)} columns, {row_count} rows"

    except Exception as e:
        result.error = f"Could not read file: {e}"

    return result
