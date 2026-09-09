"""Run or load the two frozen Task 2 seed-2026 stability candidates."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from fashion.task2.stability import load_stability_pair, run_stability_matrix


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the eligible C2 and I2 finalists over five folds at seed 2026. "
            "P* is excluded from this stability gate."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("run_or_load", "run", "load"),
        default="run_or_load",
        help="Cache policy. The default verifies cached artifacts before reuse.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the frozen pair and print traceable fold metrics."""
    arguments = _parser().parse_args(argv)
    pair = load_stability_pair()
    roles = {
        pair.c2.experiment_id: "retained_c2_comparator",
        pair.i2.experiment_id: "selected_i2_candidate",
    }
    outputs = run_stability_matrix(pair, mode=arguments.mode)
    print(
        json.dumps(
            [
                {
                    "role": roles[output.experiment_id],
                    "experiment_id": output.experiment_id,
                    "run_id": output.run_id,
                    "fold": output.fold,
                    "seed": output.seed,
                    "source": output.source,
                    "macro_f1": output.metrics.get("macro_f1"),
                    "spring_recall": output.metrics.get("per_class", {})
                    .get("Spring", {})
                    .get("recall"),
                }
                for output in outputs
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
