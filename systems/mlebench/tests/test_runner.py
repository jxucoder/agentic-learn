"""Tests for runner.py -- works without MLE-bench Docker setup."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner import execute_script, _parse_score, _parse_error


def test_parse_score():
    """Test score parsing from stdout."""
    cases = [
        ("CV Score: 0.8456\n", 0.8456),
        ("accuracy: 0.92\n", 0.92),
        ("RMSE: 1.234\n", 1.234),
        ("validation score: 0.87\n", 0.87),
        ("Best score: 0.95\n", 0.95),
        ("F1 Score: 0.78\n", 0.78),
        ("No score here\n", None),
    ]
    for stdout, expected in cases:
        score, _ = _parse_score(stdout)
        if expected is None:
            assert score is None, f"Expected None, got {score} for: {stdout.strip()}"
        else:
            assert score is not None, f"Expected {expected}, got None for: {stdout.strip()}"
            assert abs(score - expected) < 0.001, f"Expected {expected}, got {score} for: {stdout.strip()}"


def test_parse_error():
    """Test error parsing from stderr."""
    stderr = "Traceback (most recent call last):\n  File 'x.py', line 1\nFileNotFoundError: train.csv"
    error_type, error_msg = _parse_error(stderr)
    assert error_type == "FileNotFoundError"
    assert "train.csv" in error_msg


def test_execute_simple_script():
    """Test executing a simple Python script."""
    with tempfile.TemporaryDirectory() as code_dir, tempfile.TemporaryDirectory() as sub_dir:
        code = "print('hello world')\nprint('score: 0.85')"
        result = execute_script(
            code=code,
            code_dir=code_dir,
            submission_dir=sub_dir,
            timeout=30,
        )

        assert result.success is True
        assert "hello world" in result.stdout
        assert result.score is not None
        assert abs(result.score - 0.85) < 0.001


def test_execute_script_with_submission():
    """Test that the runner detects submission files."""
    with tempfile.TemporaryDirectory() as code_dir, tempfile.TemporaryDirectory() as sub_dir:
        code = f"""
import os
os.makedirs("{sub_dir}", exist_ok=True)
with open(os.path.join("{sub_dir}", "submission.csv"), "w") as f:
    f.write("id,target\\n1,0\\n2,1\\n")
print("CV Score: 0.92")
"""
        result = execute_script(
            code=code,
            code_dir=code_dir,
            submission_dir=sub_dir,
            timeout=30,
        )

        assert result.success is True
        assert result.submission_created is True
        assert result.score is not None
        assert abs(result.score - 0.92) < 0.001


def test_execute_script_error():
    """Test handling of script errors."""
    with tempfile.TemporaryDirectory() as code_dir, tempfile.TemporaryDirectory() as sub_dir:
        code = "raise ValueError('test error')"
        result = execute_script(
            code=code,
            code_dir=code_dir,
            submission_dir=sub_dir,
            timeout=30,
        )

        assert result.success is False
        assert result.error_type == "ValueError"


def test_execute_script_timeout():
    """Test script timeout handling."""
    with tempfile.TemporaryDirectory() as code_dir, tempfile.TemporaryDirectory() as sub_dir:
        code = "import time; time.sleep(100)"
        result = execute_script(
            code=code,
            code_dir=code_dir,
            submission_dir=sub_dir,
            timeout=2,
        )

        assert result.success is False
        assert result.timed_out is True


if __name__ == "__main__":
    test_parse_score()
    print("test_parse_score PASSED")

    test_parse_error()
    print("test_parse_error PASSED")

    test_execute_simple_script()
    print("test_execute_simple_script PASSED")

    test_execute_script_with_submission()
    print("test_execute_script_with_submission PASSED")

    test_execute_script_error()
    print("test_execute_script_error PASSED")

    test_execute_script_timeout()
    print("test_execute_script_timeout PASSED")

    print("\nAll tests passed!")
