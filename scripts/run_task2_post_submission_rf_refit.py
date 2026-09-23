"""Refit the fixed post-submission RF and compare it on the opened holdout."""

from __future__ import annotations

import argparse
import json

from fashion.task2.post_submission_rf_evaluation import evaluate_full_development_rf
from fashion.task2.post_submission_rf_refit import refit_rf_on_all_development


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=("refit", "evaluate", "all"), default="all")
    parser.add_argument(
        "--mode", choices=("run", "load", "run_or_load"), default="run_or_load"
    )
    parser.add_argument("--refit-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--evaluation-device", choices=("auto", "cpu", "cuda"), default="cpu"
    )
    args = parser.parse_args()

    if args.step in {"refit", "all"}:
        bundle = refit_rf_on_all_development(mode=args.mode, device=args.refit_device)
        print(
            json.dumps(
                {
                    "phase": "full_development_refit",
                    "run_id": bundle.manifest["run_id"],
                    "training_rows": bundle.manifest["training_rows"],
                    "forest": str(bundle.forest_path),
                    "manifest": str(bundle.manifest_path),
                },
                indent=2,
            )
        )
    if args.step in {"evaluate", "all"}:
        result = evaluate_full_development_rf(
            mode=args.mode, device=args.evaluation_device
        )
        print(
            json.dumps(
                {
                    "phase": "retrospective_holdout_evaluation",
                    "rows": result["rows"],
                    "macro_f1": result["macro_f1"],
                    "claim_boundary": result["claim_boundary"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
