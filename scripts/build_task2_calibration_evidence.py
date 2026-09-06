"""Build audited Task 2 cross-fitted calibration evidence."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from fashion.task2.calibration_evidence import build_calibration_evidence


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Build five-fold cross-fitted temperature calibration from frozen development OOF "
            "predictions. This post-training command never opens holdout data."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute only the declared G6 calibration analysis."""
    _parser().parse_args(argv)
    manifest = build_calibration_evidence()
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "manifest_sha256": manifest["manifest_sha256"],
                "cross_fitted_evaluation_claim_allowed": manifest[
                    "cross_fitted_evaluation_claim_allowed"
                ],
                "app_threshold_frozen": manifest["app_threshold_frozen"],
                "ultimate_winner_frozen": manifest["ultimate_winner_frozen"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
