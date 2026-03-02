"""Synthetic regression — leak-free benchmark.

Usage:
    python examples/synth_regression.py
    python examples/synth_regression.py --seed 123
"""

import argparse
import logging
import os
import sys

from aglearn import TaskConfig, evolve
from aglearn.synth import SyntheticTask, generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic regression benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.2)
    args = parser.parse_args()

    synth = SyntheticTask(
        name="reg",
        task_type="regression",
        metric="r2",
        n_samples=args.samples,
        noise_level=args.noise,
        seed=args.seed,
    )
    data_path = generate(synth, output_dir="data")
    print(f"Generated synthetic dataset: {data_path}")

    task = TaskConfig(
        description=(
            "Regression on a synthetic dataset. "
            "The 'target' column is a continuous value. "
            "Features include income, age, hours_worked, distance_km, "
            "region (categorical), education (categorical), and satisfaction (ordinal). "
            "There may also be noise features — identify and handle them. "
            "Some values are missing. "
            "Feature engineering is important: look for non-linear effects "
            "and interactions between features."
        ),
        data_path=data_path,
        target_column="target",
        metric="r2",
    )

    best = evolve(
        task,
        model=os.getenv("MLE_MODEL") or None,
        max_steps=args.steps,
        output_dir="./output/synth_reg",
    )

    if best:
        print(f"\nBest R²: {best.metric_value:.4f}")
        print("Code saved to: ./output/synth_reg/best_solution.py")
    else:
        print("\nNo successful solutions found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
