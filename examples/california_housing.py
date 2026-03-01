"""California housing price prediction — regression.

Why this benchmark:
  - 20,640 samples, 8 numeric features (income, house age, rooms,
    bedrooms, population, occupancy, latitude, longitude)
  - Spatial structure: lat/lon encode geography — the agent can discover
    location-based features or interactions
  - Feature engineering opportunities: rooms per household, bedroom ratio,
    population density, income bins, location clusters
  - Meaningful score range: linear regression gets ~0.60 R², gradient
    boosting with engineered features reaches ~0.85+

Usage:
    python examples/california_housing.py

Requires codex CLI installed and authenticated.
"""

import logging
import os
import sys

from sklearn.datasets import fetch_california_housing

from agentic_learn import TaskConfig, evolve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def prepare_data() -> str:
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "california_housing.csv")
    if not os.path.exists(path):
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame  # type: ignore[union-attr]
        df.to_csv(path, index=False)
        print(f"Prepared California Housing dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return os.path.abspath(path)


def main() -> None:
    data_path = prepare_data()

    task = TaskConfig(
        description=(
            "Regression: predict median house value (MedHouseVal) for "
            "California districts. Features include median income, house age, "
            "average rooms, average bedrooms, population, average occupancy, "
            "latitude, and longitude. "
            "Consider spatial features, income interactions, and "
            "ratio features (e.g. bedrooms/rooms, population/households)."
        ),
        data_path=data_path,
        target_column="MedHouseVal",
        metric="r2",
    )

    best = evolve(
        task,
        model=os.getenv("MLE_MODEL") or None,
        max_steps=10,
        output_dir="./output/california_housing",
    )

    if best:
        print(f"\nBest R²: {best.metric_value:.4f}")
        print("Code saved to: ./output/california_housing/best_solution.py")
    else:
        print("\nNo successful solutions found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
