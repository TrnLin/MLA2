"""Task 1 final evaluation stages; Run All in Notebook 06 only audits saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from fashion.config import ROOT
from fashion.task1.final_evaluation_runner import (
    audit_final_evaluation,
    predict_holdout,
    predict_test,
    score_holdout,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["predict-holdout", "score", "predict-test", "audit"])
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--evaluation-unlocked", action="store_true")
    args = parser.parse_args()
    if args.evaluation_unlocked and args.stage != "score":
        parser.error("--evaluation-unlocked applies only to score")
    if args.stage == "score":
        score_holdout(args.project_root, evaluation_unlocked=args.evaluation_unlocked)
    else:
        {
            "predict-holdout": predict_holdout,
            "predict-test": predict_test,
            "audit": audit_final_evaluation,
        }[args.stage](args.project_root)
    print(f"Task 1 {args.stage}: complete")


if __name__ == "__main__":
    main()
