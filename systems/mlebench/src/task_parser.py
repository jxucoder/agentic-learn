"""Parse MLE-bench competition description and data files into structured TaskInfo."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Type of ML task."""
    CLASSIFICATION = "classification"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_REGRESSION = "image_regression"
    IMAGE_SEGMENTATION = "image_segmentation"
    TEXT_CLASSIFICATION = "text_classification"
    SEQUENCE_TO_SEQUENCE = "sequence_to_sequence"
    AUDIO_CLASSIFICATION = "audio_classification"
    OBJECT_DETECTION = "object_detection"
    RANKING = "ranking"
    OTHER = "other"


@dataclass
class DataFileInfo:
    """Information about a data file."""
    path: str
    name: str
    size_bytes: int = 0
    size_human: str = ""
    extension: str = ""
    # For CSV files
    num_rows: Optional[int] = None
    num_cols: Optional[int] = None
    columns: list[str] = field(default_factory=list)
    head_preview: str = ""  # First few lines


@dataclass
class TaskInfo:
    """Parsed information about an MLE-bench competition task."""

    # Basic info
    competition_id: str = ""
    name: str = ""
    description: str = ""  # Full description.md content

    # Task classification
    task_type: TaskType = TaskType.OTHER

    # Metric
    metric_name: str = ""
    higher_is_better: bool = True

    # Data
    data_dir: str = "/home/data"
    data_files: list[DataFileInfo] = field(default_factory=list)
    train_file: Optional[DataFileInfo] = None
    test_file: Optional[DataFileInfo] = None
    sample_submission_file: Optional[DataFileInfo] = None

    # Submission
    submission_dir: str = "/home/submission"
    submission_columns: list[str] = field(default_factory=list)
    id_column: str = ""
    target_columns: list[str] = field(default_factory=list)

    # Data characteristics (extracted from sample/train)
    num_train_rows: Optional[int] = None
    num_test_rows: Optional[int] = None
    feature_columns: list[str] = field(default_factory=list)

    def to_context(self) -> str:
        """Convert to string context for LLM consumption."""
        lines = [
            f"# Competition: {self.competition_id}",
            "",
            f"## Task Type: {self.task_type.value}",
            f"## Metric: {self.metric_name} ({'higher is better' if self.higher_is_better else 'lower is better'})",
            "",
            "## Data Files:",
        ]

        for f in self.data_files:
            size = f.size_human or f"{f.size_bytes} bytes"
            lines.append(f"  - {f.name} ({size})")
            if f.columns:
                lines.append(f"    Columns: {', '.join(f.columns[:20])}")
                if len(f.columns) > 20:
                    lines.append(f"    ... and {len(f.columns) - 20} more columns")
            if f.num_rows is not None:
                lines.append(f"    Rows: {f.num_rows}")

        if self.train_file:
            lines.append(f"\n## Train file: {self.train_file.name}")
            if self.train_file.head_preview:
                lines.append(f"Preview:\n{self.train_file.head_preview}")

        if self.test_file:
            lines.append(f"\n## Test file: {self.test_file.name}")
            if self.test_file.head_preview:
                lines.append(f"Preview:\n{self.test_file.head_preview}")

        if self.sample_submission_file:
            lines.append(f"\n## Sample submission: {self.sample_submission_file.name}")
            if self.sample_submission_file.head_preview:
                lines.append(f"Preview:\n{self.sample_submission_file.head_preview}")
            if self.submission_columns:
                lines.append(f"Submission columns: {', '.join(self.submission_columns)}")

        return "\n".join(lines)


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def _read_csv_head(path: str, max_lines: int = 6) -> tuple[list[str], str, Optional[int]]:
    """Read the first few lines of a CSV file.

    Returns (columns, preview_text, estimated_row_count).
    """
    columns: list[str] = []
    lines_collected: list[str] = []
    total_lines = 0

    try:
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i == 0:
                    columns = [c.strip().strip('"') for c in line.strip().split(",")]
                if i < max_lines:
                    lines_collected.append(line.rstrip())
                total_lines += 1
    except Exception as e:
        logger.warning(f"Could not read CSV head for {path}: {e}")
        return [], "", None

    preview = "\n".join(lines_collected)
    row_count = total_lines - 1 if total_lines > 0 else None  # Subtract header

    return columns, preview, row_count


def _scan_data_file(filepath: str) -> DataFileInfo:
    """Scan a single data file and extract metadata."""
    p = Path(filepath)
    size = p.stat().st_size if p.exists() else 0

    info = DataFileInfo(
        path=str(p),
        name=p.name,
        size_bytes=size,
        size_human=_human_size(size),
        extension=p.suffix.lower(),
    )

    # For CSV files, read header and preview
    if info.extension == ".csv":
        cols, preview, row_count = _read_csv_head(filepath)
        info.columns = cols
        info.head_preview = preview
        info.num_rows = row_count
        info.num_cols = len(cols)

    return info


def _identify_key_files(
    data_files: list[DataFileInfo],
) -> tuple[Optional[DataFileInfo], Optional[DataFileInfo], Optional[DataFileInfo]]:
    """Identify train, test, and sample submission files from the data directory."""
    train_file = None
    test_file = None
    sample_sub = None

    for f in data_files:
        name_lower = f.name.lower()

        # Sample submission detection
        if "sample" in name_lower and "submission" in name_lower:
            sample_sub = f
        elif name_lower == "samplesubmission.csv":
            sample_sub = f
        elif name_lower == "sample_submission.csv":
            sample_sub = f
        # Train detection
        elif "train" in name_lower and f.extension == ".csv":
            if train_file is None or f.size_bytes > train_file.size_bytes:
                train_file = f
        # Test detection
        elif "test" in name_lower and f.extension == ".csv":
            if test_file is None or f.size_bytes > test_file.size_bytes:
                test_file = f

    # Fallback: if no explicit train/test, look for common patterns
    if train_file is None:
        csv_files = [f for f in data_files if f.extension == ".csv" and f != sample_sub]
        if csv_files:
            # Largest CSV that isn't sample submission is likely train
            csv_files.sort(key=lambda x: x.size_bytes, reverse=True)
            train_file = csv_files[0]
            if len(csv_files) > 1:
                test_file = csv_files[1]

    return train_file, test_file, sample_sub


def _detect_metric(description: str) -> tuple[str, bool]:
    """Detect the evaluation metric from the competition description.

    Returns (metric_name, higher_is_better).
    """
    desc_lower = description.lower()

    # Ordered by specificity -- check specific metrics first
    metric_patterns: list[tuple[str, str, bool]] = [
        # Regression metrics (lower is better)
        (r"root mean squared error|rmse", "RMSE", False),
        (r"mean squared error|mse", "MSE", False),
        (r"mean absolute error|mae", "MAE", False),
        (r"mean absolute percentage error|mape", "MAPE", False),
        (r"rmsle|root mean squared logarithmic error", "RMSLE", False),
        # Classification metrics (higher is better)
        (r"area under.*roc|auc[\s\-]?roc|roc[\s\-]?auc|auroc", "AUC-ROC", True),
        (r"log[\s\-]?loss|logarithmic loss|binary crossentropy", "Log Loss", False),
        (r"categorization accuracy|classification accuracy|accuracy", "Accuracy", True),
        (r"f1[\s\-]?score|f1-score|f-measure", "F1 Score", True),
        (r"macro[\s\-]?f1", "Macro F1", True),
        (r"micro[\s\-]?f1", "Micro F1", True),
        (r"weighted[\s\-]?f1", "Weighted F1", True),
        (r"precision", "Precision", True),
        (r"recall", "Recall", True),
        (r"matthews correlation|mcc", "MCC", True),
        (r"cohen'?s?\s*kappa|quadratic weighted kappa", "Cohen's Kappa", True),
        # Ranking
        (r"mean average precision|map@|map ", "MAP", True),
        (r"ndcg", "NDCG", True),
        # Segmentation / detection
        (r"dice[\s\-]?coefficient|dice score", "Dice", True),
        (r"intersection over union|iou|jaccard", "IoU", True),
        # Other
        (r"spearman", "Spearman Correlation", True),
        (r"pearson", "Pearson Correlation", True),
        (r"r[\s\-]?squared|r2", "R2", True),
    ]

    for pattern, name, higher in metric_patterns:
        if re.search(pattern, desc_lower):
            return name, higher

    return "Unknown", True


def _detect_task_type(description: str, data_files: list[DataFileInfo]) -> TaskType:
    """Detect the task type from description and data files."""
    desc_lower = description.lower()

    # Check for image data
    has_images = any(
        f.extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm")
        for f in data_files
    )
    has_image_dirs = any(
        "image" in f.name.lower() or "img" in f.name.lower() or "photo" in f.name.lower()
        for f in data_files
    )

    # Check for audio data
    has_audio = any(f.extension in (".wav", ".mp3", ".flac", ".ogg") for f in data_files)

    # Check for text-heavy tasks
    text_keywords = ["text classification", "sentiment", "nlp", "natural language",
                     "toxic", "spam", "author", "comment", "tweet", "normalize"]
    is_text = any(kw in desc_lower for kw in text_keywords)

    # Image tasks
    if has_images or has_image_dirs:
        if "segment" in desc_lower:
            return TaskType.IMAGE_SEGMENTATION
        if "detect" in desc_lower and "object" in desc_lower:
            return TaskType.OBJECT_DETECTION
        if "regression" in desc_lower:
            return TaskType.IMAGE_REGRESSION
        return TaskType.IMAGE_CLASSIFICATION

    # Audio tasks
    if has_audio:
        return TaskType.AUDIO_CLASSIFICATION

    # Text tasks
    if is_text:
        if "sequence" in desc_lower or "normalization" in desc_lower:
            return TaskType.SEQUENCE_TO_SEQUENCE
        return TaskType.TEXT_CLASSIFICATION

    # Tabular tasks
    if "regression" in desc_lower or "predict" in desc_lower and (
        "price" in desc_lower or "value" in desc_lower or "amount" in desc_lower
        or "fare" in desc_lower or "cost" in desc_lower
    ):
        return TaskType.REGRESSION

    if "rank" in desc_lower:
        return TaskType.RANKING

    if "classif" in desc_lower or "predict" in desc_lower:
        return TaskType.CLASSIFICATION

    return TaskType.OTHER


def parse_task(
    data_dir: str = "/home/data",
    submission_dir: str = "/home/submission",
) -> TaskInfo:
    """Parse a MLE-bench competition task from the data directory.

    Args:
        data_dir: Path to the competition data directory.
        submission_dir: Path where submission.csv should be written.

    Returns:
        TaskInfo with parsed competition metadata.
    """
    data_path = Path(data_dir)
    task = TaskInfo(data_dir=data_dir, submission_dir=submission_dir)

    # Read competition description
    desc_path = data_path / "description.md"
    if desc_path.exists():
        task.description = desc_path.read_text(errors="replace")
    else:
        # Try other description file names
        for name in ("description.txt", "README.md", "overview.md"):
            alt = data_path / name
            if alt.exists():
                task.description = alt.read_text(errors="replace")
                break

    # Get competition ID from env or directory name
    task.competition_id = os.environ.get("COMPETITION_ID", data_path.name)
    task.name = task.competition_id.replace("-", " ").title()

    # Scan all data files
    if data_path.exists():
        for item in sorted(data_path.iterdir()):
            if item.is_file() and item.name != "description.md":
                try:
                    task.data_files.append(_scan_data_file(str(item)))
                except Exception as e:
                    logger.warning(f"Could not scan {item}: {e}")
            elif item.is_dir():
                # Record directories (image folders, etc.)
                dir_info = DataFileInfo(
                    path=str(item),
                    name=item.name + "/",
                    extension="dir",
                )
                # Count files in directory
                try:
                    files_in_dir = list(item.rglob("*"))
                    dir_info.num_rows = len([f for f in files_in_dir if f.is_file()])
                    dir_info.size_human = f"{dir_info.num_rows} files"
                except Exception:
                    pass
                task.data_files.append(dir_info)

    # Identify key files
    task.train_file, task.test_file, task.sample_submission_file = _identify_key_files(
        task.data_files
    )

    # Extract submission format from sample submission
    if task.sample_submission_file and task.sample_submission_file.columns:
        task.submission_columns = task.sample_submission_file.columns
        if task.submission_columns:
            task.id_column = task.submission_columns[0]
            task.target_columns = task.submission_columns[1:]
        task.num_test_rows = task.sample_submission_file.num_rows

    # Extract train metadata
    if task.train_file:
        task.num_train_rows = task.train_file.num_rows
        if task.train_file.columns and task.submission_columns:
            # Feature columns = train columns minus target and ID columns
            submission_col_set = set(task.submission_columns)
            task.feature_columns = [
                c for c in task.train_file.columns if c not in submission_col_set
            ]

    # Detect metric
    task.metric_name, task.higher_is_better = _detect_metric(task.description)

    # Detect task type
    task.task_type = _detect_task_type(task.description, task.data_files)

    logger.info(
        f"Parsed task: {task.competition_id} | "
        f"type={task.task_type.value} | "
        f"metric={task.metric_name} | "
        f"train={task.num_train_rows} rows | "
        f"test={task.num_test_rows} rows | "
        f"files={len(task.data_files)}"
    )

    return task
