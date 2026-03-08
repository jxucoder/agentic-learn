"""Generate an experiment setup and public brief via Gemini CLI."""

from __future__ import annotations

import argparse
import json

from aglearn_experiments import generate_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-type",
        choices=[
            "classification",
            "regression",
            "multiclass",
            "temporal_regression",
            "high_dim",
        ],
        default="multiclass",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--output-root", default="experiments/generated")
    parser.add_argument("--gemini-model")
    args = parser.parse_args()

    manifest = generate_benchmark(
        task_type=args.task_type,
        seed=args.seed,
        samples=args.samples,
        noise=args.noise,
        output_root=args.output_root,
        gemini_model=args.gemini_model,
    )

    print(json.dumps(manifest.to_dict(), indent=2))


if __name__ == "__main__":
    main()
