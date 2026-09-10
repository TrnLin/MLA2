"""Build and freeze the audited Task 2 ultimate judgement."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from fashion.task2.ultimate_judgement import build_ultimate_judgement_evidence


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Apply the frozen G7 scorecard to verified development OOF evidence and write "
            "one immutable pre-holdout Task 2 selection record. This command never trains "
            "a model and never opens holdout data."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Execute only the declared G7 judgement and freeze."""
    _parser().parse_args(argv)
    manifest = build_ultimate_judgement_evidence()
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "manifest_sha256": manifest["manifest_sha256"],
                "selected_candidate": manifest["selected_candidate"],
                "selection_freeze_sha256": manifest["selection_freeze_sha256"],
                "holdout_opened": manifest["holdout_opened"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
