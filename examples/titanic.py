"""Titanic survival prediction — binary classification.

Why this benchmark:
  - Mixed types: numeric (age, fare) + categorical (sex, embarked, pclass)
  - Missing values: ~20% of age, ~77% of cabin, a few embarked
  - Feature engineering matters: extract title from name, family size
    from sibsp+parch, deck from cabin prefix, fare bins, etc.
  - Meaningful score range: a naive model gets ~0.72 F1, good feature
    engineering + tuned model reaches ~0.82+

Usage:
    python examples/titanic.py

Requires codex CLI installed and authenticated.
"""

import logging
import os
import sys

from sklearn.datasets import fetch_openml

from aglearn import TaskConfig, evolve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def prepare_data() -> str:
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "titanic.csv")
    if not os.path.exists(path):
        X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
        df = X.copy()
        df["survived"] = y
        df.to_csv(path, index=False)
        print(f"Downloaded Titanic dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return os.path.abspath(path)


def main() -> None:
    data_path = prepare_data()

    task = TaskConfig(
        description=(
            "Binary classification: predict passenger survival on the Titanic. "
            "The 'survived' column is the target (0 = died, 1 = survived). "
            "The dataset has mixed types and missing values. "
            "Drop the 'boat', 'body', and 'home.dest' columns — they leak "
            "survival information. "
            "Feature engineering is important — consider extracting titles "
            "from names, grouping cabin letters, computing family size, etc."
        ),
        data_path=data_path,
        target_column="survived",
        metric="f1",
    )

    best = evolve(
        task,
        model=os.getenv("MLE_MODEL") or None,
        max_steps=10,
        output_dir="./output/titanic",
    )

    if best:
        print(f"\nBest F1: {best.metric_value:.4f}")
        print("Code saved to: ./output/titanic/best_solution.py")
    else:
        print("\nNo successful solutions found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
