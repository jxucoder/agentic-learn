"""Run multiple contestants against one generated experiment setup."""

from __future__ import annotations

import argparse
import json

from aglearn_experiments import run_arena


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--contestants", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-root", default="output/arena")
    args = parser.parse_args()

    leaderboard = run_arena(
        manifest_path=args.manifest,
        contestants_path=args.contestants,
        steps=args.steps,
        timeout=args.timeout,
        output_root=args.output_root,
    )
    print(json.dumps(leaderboard, indent=2))


if __name__ == "__main__":
    main()
