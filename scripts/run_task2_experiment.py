"""Run or load one immutable Task 2 experiment declaration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fashion.task2.experiments import run_or_load_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or load every declared fold and seed for one Task 2 config."
    )
    parser.add_argument("config", type=Path, help="Path to one configs/task2 JSON file.")
    parser.add_argument(
        "--mode",
        choices=("run_or_load", "run", "load"),
        default="run_or_load",
        help="Cache policy. The default verifies cached artifacts before training.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the requested config and print a compact machine-readable summary."""
    arguments = _parser().parse_args(argv)
    outputs = run_or_load_experiment(arguments.config, mode=arguments.mode)
    print(
        json.dumps(
            [
                {
                    "run_id": output.run_id,
                    "fold": output.fold,
                    "seed": output.seed,
                    "source": output.source,
                    "macro_f1": output.metrics.get("macro_f1"),
                }
                for output in outputs
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
