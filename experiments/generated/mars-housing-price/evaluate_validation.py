from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score

METRIC = "r2"
TARGET_COLUMN = "target"
VALIDATION_PATH = Path("data/synth_mars_housing_price_validation.csv")
VALIDATION_SAMPLE_SUBMISSION_PATH = Path(
    "data/synth_mars_housing_price_validation_sample_submission.csv"
)
REQUIRE_NUMERIC_TARGET = True


def validate_submission(submission_path: Path) -> list[str]:
    root = Path(__file__).resolve().parent
    validation_df = pd.read_csv(root / VALIDATION_PATH)
    sample_submission_df = pd.read_csv(root / VALIDATION_SAMPLE_SUBMISSION_PATH)

    try:
        submission_df = pd.read_csv(submission_path)
    except Exception as exc:
        return [f"Unable to read submission CSV: {exc}"]

    errors: list[str] = []
    expected_columns = list(sample_submission_df.columns)
    if list(submission_df.columns) != expected_columns:
        errors.append(
            "Columns must exactly match the validation sample submission: "
            + ", ".join(expected_columns)
        )
        return errors

    if submission_df["row_id"].isna().any():
        errors.append("`row_id` contains missing values.")
    if submission_df["row_id"].duplicated().any():
        errors.append("`row_id` contains duplicates.")
    if submission_df[TARGET_COLUMN].isna().any():
        errors.append(f"`{TARGET_COLUMN}` contains missing values.")

    expected_row_ids = set(validation_df["row_id"].tolist())
    actual_row_ids = set(submission_df["row_id"].tolist())
    missing_row_ids = sorted(expected_row_ids - actual_row_ids)[:10]
    extra_row_ids = sorted(actual_row_ids - expected_row_ids)[:10]
    if len(submission_df) != len(validation_df):
        errors.append(
            f"Expected {len(validation_df)} rows but found {len(submission_df)}."
        )
    if missing_row_ids:
        errors.append(f"Missing row_id values (first 10): {missing_row_ids}")
    if extra_row_ids:
        errors.append(f"Unexpected row_id values (first 10): {extra_row_ids}")

    if REQUIRE_NUMERIC_TARGET:
        numeric_target = pd.to_numeric(submission_df[TARGET_COLUMN], errors="coerce")
        if numeric_target.isna().any():
            errors.append(f"`{TARGET_COLUMN}` must be numeric for this task.")

    return errors


def score_submission(submission_path: Path) -> tuple[float | None, list[str]]:
    errors = validate_submission(submission_path)
    if errors:
        return None, errors

    root = Path(__file__).resolve().parent
    validation_df = pd.read_csv(root / VALIDATION_PATH)
    submission_df = pd.read_csv(submission_path)
    merged = validation_df[["row_id", TARGET_COLUMN]].merge(
        submission_df[["row_id", TARGET_COLUMN]],
        on="row_id",
        how="left",
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )

    pred_column = f"{TARGET_COLUMN}_pred"
    true_column = f"{TARGET_COLUMN}_true"
    try:
        if METRIC == "f1":
            score = float(
                f1_score(
                    merged[true_column].astype(int),
                    merged[pred_column].astype(int),
                )
            )
        elif METRIC == "f1_macro":
            score = float(
                f1_score(
                    merged[true_column].astype(int),
                    merged[pred_column].astype(int),
                    average="macro",
                )
            )
        elif METRIC == "accuracy":
            score = float(
                accuracy_score(
                    merged[true_column].astype(int),
                    merged[pred_column].astype(int),
                )
            )
        elif METRIC == "r2":
            score = float(
                r2_score(
                    merged[true_column].astype(float),
                    merged[pred_column].astype(float),
                )
            )
        else:
            return None, [f"Unsupported metric: {METRIC}"]
    except (TypeError, ValueError) as exc:
        return None, [f"Unable to score submission: {exc}"]

    return score, []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a submission against the public validation split."
    )
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()

    submission_path = Path(args.submission).resolve()
    score, errors = score_submission(submission_path)
    payload = {
        "ok": not errors,
        "submission_path": str(submission_path),
        "metric_name": METRIC,
        "metric": score,
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
