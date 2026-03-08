"""Worker entrypoint executed inside a Modal sandbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arena import ContestantSpec, load_public_manifest, run_public_contestant_loop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--contestant", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = load_public_manifest(args.manifest)
    contestant = ContestantSpec.from_dict(
        json.loads(Path(args.contestant).read_text(encoding="utf-8"))
    )
    summary = run_public_contestant_loop(
        manifest=manifest,
        contestant=contestant,
        steps=args.steps,
        timeout=args.timeout,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
