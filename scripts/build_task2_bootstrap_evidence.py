"""Build audited Task 2 paired product-family bootstrap evidence."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from fashion.task2.bootstrap_evidence import build_paired_bootstrap_evidence


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Build paired product-family bootstrap intervals from four frozen development OOF "
            "packs. This post-training command never opens holdout data."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute only the declared G6 paired bootstrap analysis."""
    _parser().parse_args(argv)
    manifest = build_paired_bootstrap_evidence()
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "manifest_sha256": manifest["manifest_sha256"],
                "random_seed_generalisability_claim_allowed": manifest[
                    "random_seed_generalisability_claim_allowed"
                ],
                "ultimate_winner_frozen": manifest["ultimate_winner_frozen"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
