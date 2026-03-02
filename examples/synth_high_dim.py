"""Synthetic high-dimensional classification — harder benchmark.

Challenges beyond the basic classification benchmark:
- 50 features but only 8 are informative
- XOR-like interactions that defeat linear models
- Redundant features (near-copies of informative ones)
- Multicollinear feature groups
- Feature names are shuffled (no semantic hints)

Usage:
    python examples/synth_high_dim.py
    python examples/synth_high_dim.py --seed 123 --steps 15
"""

import argparse
import logging
import os
import sys

from aglearn import TaskConfig, evolve
from aglearn.synth_hard import HardSyntheticTask, generate_high_dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic high-dimensional benchmark (hard)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--noise", type=float, default=0.15)
    args = parser.parse_args()

    synth = HardSyntheticTask(
        name="high_dim", task_type="high_dim", metric="f1",
        n_samples=args.samples, noise_level=args.noise, seed=args.seed,
    )
    data_path = generate_high_dim(synth, output_dir="data")
    print(f"Generated synthetic dataset: {data_path}")

    task = TaskConfig(
        description=(
            "Binary classification on a high-dimensional synthetic dataset. "
            "The 'target' column is the label (0 or 1). "
            "There are 50 features (feat_00 through feat_49) but most are noise. "
            "Only ~8 features are truly informative — the rest are pure noise, "
            "redundant copies, or multicollinear groups. "
            "Feature names are deliberately uninformative (feat_XX). "
            "The true signal involves XOR-like interactions: pairs of features "
            "interact non-linearly so linear models will struggle. "
            "Some noise features are near-duplicates of real features (with added noise) "
            "— including both the original and copy hurts more than helps. "
            "There is a group of ~5 multicollinear features — they should be reduced. "
            "Strategy: start with aggressive feature selection (mutual information, "
            "tree-based importance, or RFECV), then look for non-linear interactions. "
            "Some values are missing across a random subset of columns."
        ),
        data_path=data_path,
        target_column="target",
        metric="f1",
    )

    best = evolve(
        task,
        model=os.getenv("MLE_MODEL") or None,
        max_steps=args.steps,
        output_dir="./output/synth_high_dim",
    )

    if best:
        print(f"\nBest F1: {best.metric_value:.4f}")
        print("Code saved to: ./output/synth_high_dim/best_solution.py")
    else:
        print("\nNo successful solutions found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
