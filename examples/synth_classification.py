"""Synthetic binary classification — leak-free benchmark.

Usage:
    python examples/synth_classification.py
    python examples/synth_classification.py --seed 123
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
    parser = argparse.ArgumentParser(description="Synthetic classification benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.1)
    args = parser.parse_args()

    synth = SyntheticTask(
        name="clf",
        task_type="classification",
        metric="f1",
        n_samples=args.samples,
        noise_level=args.noise,
        seed=args.seed,
    )
    data_path = generate(synth, output_dir="data")
    print(f"Generated synthetic dataset: {data_path}")

    task = TaskConfig(
        description=(
            "Binary classification on a synthetic dataset. "
            "The 'target' column is the label (0 or 1). "
            "Features include income, age, hours_worked, distance_km, "
            "region (categorical), education (categorical), and satisfaction (ordinal). "
            "There may also be noise features — identify and handle them. "
            "Some values are missing. "
            "Feature engineering is important: look for non-linear effects "
            "and interactions between features."
        ),
        data_path=data_path,
        target_column="target",
        metric="f1",
    )

    best = evolve(
        task,
        model=os.getenv("MLE_MODEL") or None,
        max_steps=args.steps,
        output_dir="./output/synth_clf",
    )

    if best:
        print(f"\nBest F1: {best.metric_value:.4f}")
        print("Code saved to: ./output/synth_clf/best_solution.py")
    else:
        print("\nNo successful solutions found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
