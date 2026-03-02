"""Synthetic multi-class classification — harder benchmark.

Challenges beyond the basic classification benchmark:
- 5 imbalanced classes (minority class is only 5% of data)
- 3-way feature interactions
- Conditional effects (feature only matters within a subgroup)
- Correlated noise features that look informative
- Non-random missing values (MNAR)

Usage:
    python examples/synth_multiclass.py
    python examples/synth_multiclass.py --seed 123 --steps 15
"""

import argparse
import logging
import os
import sys

from aglearn import TaskConfig, evolve
from aglearn.synth_hard import HardSyntheticTask, generate_multiclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic multi-class benchmark (hard)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--noise", type=float, default=0.2)
    args = parser.parse_args()

    synth = HardSyntheticTask(
        name="multiclass",
        task_type="multiclass",
        metric="f1_macro",
        n_samples=args.samples,
        noise_level=args.noise,
        seed=args.seed,
    )
    data_path = generate_multiclass(synth, output_dir="data")
    print(f"Generated synthetic dataset: {data_path}")

    task = TaskConfig(
        description=(
            "Multi-class classification on a synthetic dataset. "
            "The 'target' column has 5 classes (0-4) with imbalanced distribution. "
            "Features include salary, experience_years, weekly_hours, project_count, "
            "team_size, department (8 categories), seniority (5 levels), location, "
            "performance_score, and commute_minutes. "
            "There are also several noise features — some are correlated with inputs "
            "but NOT with the target, making them deceptive. "
            "Missing values are NOT random — higher values are more likely missing. "
            "Feature engineering is critical: look for multi-way interactions, "
            "conditional effects (features that only matter for certain subgroups), "
            "and non-linear transformations. "
            "Use macro-averaged F1 to handle class imbalance fairly. "
            "Consider class weighting, oversampling, or other imbalance strategies."
        ),
        data_path=data_path,
        target_column="target",
        metric="f1_macro",
    )

    best = evolve(
        task,
        model=os.getenv("MLE_MODEL") or None,
        max_steps=args.steps,
        output_dir="./output/synth_multiclass",
    )

    if best:
        print(f"\nBest macro F1: {best.metric_value:.4f}")
        print("Code saved to: ./output/synth_multiclass/best_solution.py")
    else:
        print("\nNo successful solutions found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
