"""Run multiple contestants in isolated Modal sandboxes against one benchmark."""

from __future__ import annotations

import argparse
import json

from aglearn_experiments.modal_backend import run_modal_arena


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--contestants", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-root", default="output/modal_arena")
    parser.add_argument("--app-name", default="aglearn-arena")
    parser.add_argument("--sandbox-timeout", type=int)
    parser.add_argument("--apt-package", action="append", default=[])
    parser.add_argument("--npm-package", action="append", default=[])
    parser.add_argument("--python-package", action="append", default=[])
    parser.add_argument("--image-command", action="append", default=[])
    args = parser.parse_args()

    leaderboard = run_modal_arena(
        manifest_path=args.manifest,
        contestants_path=args.contestants,
        steps=args.steps,
        timeout=args.timeout,
        output_root=args.output_root,
        app_name=args.app_name,
        sandbox_timeout=args.sandbox_timeout,
        apt_packages=tuple(args.apt_package),
        npm_packages=tuple(args.npm_package),
        python_packages=tuple(args.python_package),
        image_commands=tuple(args.image_command),
    )
    print(json.dumps(leaderboard, indent=2))


if __name__ == "__main__":
    main()
