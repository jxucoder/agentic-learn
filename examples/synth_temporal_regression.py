"""Synthetic temporal regression — harder benchmark.

Challenges beyond the basic regression benchmark:
- Time-ordered data with trend and seasonal components
- High-cardinality categorical (50 store IDs)
- Heteroscedastic noise (variance depends on features)
- 5% outlier contamination in the target
- Complex price elasticity and diminishing-returns effects

Usage:
    python examples/synth_temporal_regression.py
    python examples/synth_temporal_regression.py --seed 123 --steps 15
"""

import argparse
import logging
import os
import sys

from aglearn import TaskConfig, evolve
from aglearn.synth_hard import HardSyntheticTask, generate_temporal_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic temporal regression benchmark (hard)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--noise", type=float, default=0.3)
    args = parser.parse_args()

    synth = HardSyntheticTask(
        name="temporal_reg", task_type="temporal_regression", metric="r2",
        n_samples=args.samples, noise_level=args.noise, seed=args.seed,
    )
    data_path = generate_temporal_regression(synth, output_dir="data")
    print(f"Generated synthetic dataset: {data_path}")

    task = TaskConfig(
        description=(
            "Regression on a synthetic temporal sales dataset. "
            "The 'target' column is a continuous sales value. "
            "The data has temporal structure: day_index, day_of_week, and month "
            "encode time — look for trends, weekly patterns, and seasonality. "
            "store_id is a high-cardinality categorical (50 stores) — each store "
            "has a hidden quality factor. Consider target encoding or entity embeddings. "
            "Features include price, competitor_price, promotion, foot_traffic, "
            "online_reviews, ad_spend, inventory_level, and temperature. "
            "There are noise features that may look informative (e.g., a random walk "
            "trend, fake seasonal pattern). "
            "The target contains ~5% outliers — consider robust methods. "
            "Key interactions: price elasticity (log price ratio vs competitor), "
            "promotion effect depends on foot traffic, ad spend has diminishing "
            "returns modulated by temperature, inventory has a U-shape effect. "
            "Some missing values are present across numeric features."
        ),
        data_path=data_path,
        target_column="target",
        metric="r2",
    )

    best = evolve(
        task,
        model=os.getenv("MLE_MODEL") or None,
        max_steps=args.steps,
        output_dir="./output/synth_temporal_reg",
    )

    if best:
        print(f"\nBest R²: {best.metric_value:.4f}")
        print("Code saved to: ./output/synth_temporal_reg/best_solution.py")
    else:
        print("\nNo successful solutions found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
