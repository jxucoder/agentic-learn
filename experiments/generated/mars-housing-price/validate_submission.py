from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET_COLUMN = "target"
EXPECTED_TEST_PATH = Path("data/synth_mars_housing_price_test.csv")
SAMPLE_SUBMISSION_PATH = Path("data/synth_mars_housing_price_sample_submission.csv")
REQUIRE_NUMERIC_TARGET = True


def validate_submission(submission_path: Path) -> list[str]:
    root = Path(__file__).resolve().parent
    expected_test_path = root / EXPECTED_TEST_PATH
    sample_submission_path = root / SAMPLE_SUBMISSION_PATH

    try:
        submission_df = pd.read_csv(submission_path)
    except Exception as exc:
        return [f"Unable to read submission CSV: {exc}"]

    try:
        expected_test_df = pd.read_csv(expected_test_path)
        sample_submission_df = pd.read_csv(sample_submission_path)
    except Exception as exc:
        return [f"Unable to read benchmark reference files: {exc}"]

    errors: list[str] = []
    expected_columns = list(sample_submission_df.columns)
    if list(submission_df.columns) != expected_columns:
        errors.append(
            "Columns must exactly match the sample submission: "
            + ", ".join(expected_columns)
        )
        return errors

    if submission_df["row_id"].isna().any():
        errors.append("`row_id` contains missing values.")
    if submission_df["row_id"].duplicated().any():
        errors.append("`row_id` contains duplicates.")
    if submission_df[TARGET_COLUMN].isna().any():
        errors.append(f"`{TARGET_COLUMN}` contains missing values.")

    expected_row_ids = set(expected_test_df["row_id"].tolist())
    actual_row_ids = set(submission_df["row_id"].tolist())
    missing_row_ids = sorted(expected_row_ids - actual_row_ids)[:10]
    extra_row_ids = sorted(actual_row_ids - expected_row_ids)[:10]
    if len(submission_df) != len(expected_test_df):
        errors.append(
            f"Expected {len(expected_test_df)} rows but found {len(submission_df)}."
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a competition submission against the public schema."
    )
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()

    submission_path = Path(args.submission).resolve()
    errors = validate_submission(submission_path)
    payload = {
        "ok": not errors,
        "submission_path": str(submission_path),
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
