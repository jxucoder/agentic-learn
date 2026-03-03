"""Synthetic binary classification — leak-free benchmark.

Usage:
    python examples/synth_classification.py
    python examples/synth_classification.py --seed 123
"""

import argparse
import json
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
    meta_path = os.path.join("data", "synth_clf_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    files = meta["files"]
    print(f"Generated synthetic train split: {files['train']}")
    print(f"Kaggle-style test split (no labels): {files['test']}")

    task = TaskConfig(
        description=(
            "Kaggle-style binary classification on synthetic customer-event data. "
            "The provided CSV is the TRAIN split and includes target (0/1). "
            "A hidden TEST split exists separately with no labels. "
            "Columns include mixed data types: temporal features (event_date), "
            "IDs/high-cardinality categoricals (customer_id, city_code, campaign_id), "
            "behavioral numerics (sessions_30d, tenure_months, avg_order_value, return_rate), "
            "and categorical business context (segment, plan_tier, acquisition_channel, region). "
            "Some columns are noisy or spurious under distribution shift (e.g. campaign-related proxy), "
            "and missingness is partly non-random. "
            "Treat this like a Kaggle tabular problem: strong validation strategy, "
            "robust preprocessing, and interaction-aware feature engineering."
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
