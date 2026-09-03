"""Build audited Task 2 robustness and deployment-cost evidence."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from fashion.task2.robustness_evidence import build_robustness_cost_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or hash-load the frozen primary-seed robustness grid and machine-specific "
            "deployment-cost probes. This post-training command never opens holdout data."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("run", "load", "run_or_load"),
        default="run_or_load",
        help=(
            "run replaces exact probe caches; load requires them; run_or_load verifies and "
            "reuses exact caches before running a missing probe"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute only the frozen G6 robustness/cost declaration."""
    arguments = _parser().parse_args(argv)
    manifest = build_robustness_cost_evidence(mode=arguments.mode)
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "manifest_sha256": manifest["manifest_sha256"],
                "candidate_selection_affected": manifest["candidate_selection_affected"],
                "ultimate_winner_frozen": manifest["ultimate_winner_frozen"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
