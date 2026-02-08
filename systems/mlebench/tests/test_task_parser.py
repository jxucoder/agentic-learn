"""Tests for task_parser.py -- works without MLE-bench Docker setup."""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task_parser import (
    parse_task,
    _detect_metric,
    _detect_task_type,
    _read_csv_head,
    _human_size,
    DataFileInfo,
    TaskType,
)


def test_detect_metric():
    """Test metric detection from competition descriptions."""
    cases = [
        ("Submissions are evaluated on area under the ROC curve", "AUC-ROC", True),
        ("Evaluated using Root Mean Squared Error", "RMSE", False),
        ("Scored by categorization accuracy", "Accuracy", True),
        ("Submissions are scored using log loss", "Log Loss", False),
        ("Evaluated on the F1 score", "F1 Score", True),
        ("Metric: mean absolute error", "MAE", False),
        ("quadratic weighted kappa", "Cohen's Kappa", True),
    ]
    for desc, expected_name, expected_higher in cases:
        name, higher = _detect_metric(desc)
        assert name == expected_name, f"Expected {expected_name}, got {name} for: {desc}"
        assert higher == expected_higher, f"Expected higher={expected_higher} for: {desc}"


def test_detect_task_type():
    """Test task type detection."""
    img_files = [DataFileInfo(path="train/img1.jpg", name="img1.jpg", extension=".jpg")]
    csv_files = [DataFileInfo(path="train.csv", name="train.csv", extension=".csv")]

    assert _detect_task_type("image classification task", img_files) == TaskType.IMAGE_CLASSIFICATION
    assert _detect_task_type("predict the price of houses", csv_files) == TaskType.REGRESSION
    assert _detect_task_type("classify the sentiment of text", csv_files) == TaskType.TEXT_CLASSIFICATION
    assert _detect_task_type("segment the image", img_files) == TaskType.IMAGE_SEGMENTATION


def test_read_csv_head():
    """Test CSV head reading."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("id,feature1,target\n")
        for i in range(10):
            f.write(f"{i},{i*2},{i%2}\n")
        f.flush()

        columns, preview, row_count = _read_csv_head(f.name)

        assert columns == ["id", "feature1", "target"]
        assert row_count == 10
        assert "id,feature1,target" in preview

        os.unlink(f.name)


def test_human_size():
    """Test human-readable file size."""
    assert _human_size(100) == "100.0B"
    assert _human_size(1024) == "1.0KB"
    assert _human_size(1024 * 1024) == "1.0MB"
    assert _human_size(1024 * 1024 * 1024) == "1.0GB"


def test_parse_task_with_mock_data():
    """Test parse_task with a mock competition directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock competition data
        desc = Path(tmpdir) / "description.md"
        desc.write_text(
            "# Spaceship Titanic\n\n"
            "Predict which passengers were transported.\n\n"
            "Submissions are evaluated on classification accuracy.\n"
        )

        train = Path(tmpdir) / "train.csv"
        train.write_text(
            "PassengerId,HomePlanet,CryoSleep,Transported\n"
            "0001_01,Europa,False,False\n"
            "0002_01,Earth,True,True\n"
        )

        test = Path(tmpdir) / "test.csv"
        test.write_text(
            "PassengerId,HomePlanet,CryoSleep\n"
            "0003_01,Mars,False\n"
        )

        sample = Path(tmpdir) / "sample_submission.csv"
        sample.write_text(
            "PassengerId,Transported\n"
            "0003_01,False\n"
        )

        # Parse
        task = parse_task(data_dir=tmpdir, submission_dir="/tmp/test_sub")

        assert task.metric_name == "Accuracy"
        assert task.higher_is_better is True
        assert task.train_file is not None
        assert task.test_file is not None
        assert task.sample_submission_file is not None
        assert task.submission_columns == ["PassengerId", "Transported"]
        assert task.id_column == "PassengerId"
        assert task.target_columns == ["Transported"]
        assert task.num_train_rows == 2
        assert task.num_test_rows == 1

        # Check context generation
        context = task.to_context()
        assert "Accuracy" in context
        assert "PassengerId" in context


def test_parse_task_context_output():
    """Test that to_context produces valid LLM-consumable output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        desc = Path(tmpdir) / "description.md"
        desc.write_text("Predict house prices. Evaluated on RMSE.")

        train = Path(tmpdir) / "train.csv"
        train.write_text("Id,Feature1,SalePrice\n1,100,200000\n2,150,250000\n")

        test = Path(tmpdir) / "test.csv"
        test.write_text("Id,Feature1\n3,120\n")

        sample = Path(tmpdir) / "sample_submission.csv"
        sample.write_text("Id,SalePrice\n3,0\n")

        task = parse_task(data_dir=tmpdir, submission_dir="/tmp/sub")
        context = task.to_context()

        assert "RMSE" in context
        assert "lower is better" in context
        assert "Id" in context


if __name__ == "__main__":
    test_detect_metric()
    print("test_detect_metric PASSED")

    test_detect_task_type()
    print("test_detect_task_type PASSED")

    test_read_csv_head()
    print("test_read_csv_head PASSED")

    test_human_size()
    print("test_human_size PASSED")

    test_parse_task_with_mock_data()
    print("test_parse_task_with_mock_data PASSED")

    test_parse_task_context_output()
    print("test_parse_task_context_output PASSED")

    print("\nAll tests passed!")
