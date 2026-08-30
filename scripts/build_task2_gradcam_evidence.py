"""Build audited Task 2 Grad-CAM and failure-review evidence."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from fashion.task2.gradcam_evidence import build_gradcam_failure_evidence


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Build deterministic Grad-CAM contact sheets and non-causal failure diagnostics "
            "from frozen development OOF checkpoints. This command never opens holdout data."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute only the declared G6 Grad-CAM review."""
    _parser().parse_args(argv)
    manifest = build_gradcam_failure_evidence()
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "manifest_sha256": manifest["manifest_sha256"],
                "causal_failure_claim_allowed": manifest["causal_failure_claim_allowed"],
                "ultimate_winner_frozen": manifest["ultimate_winner_frozen"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
