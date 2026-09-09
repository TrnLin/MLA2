"""Build, score, or verify the two-phase Task 4 final evaluation."""

from __future__ import annotations

import argparse
import json

from fashion.task4_evaluation import (
    build_blind_holdout_evidence,
    load_verified_holdout_evaluation,
    score_holdout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser(
        "predict",
        help="freeze label-free holdout rankings",
    )
    predict.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    score = subparsers.add_parser(
        "score",
        help="open protected labels once and score frozen rankings",
    )
    score.add_argument(
        "--evaluation-unlocked",
        action="store_true",
        help="explicit acknowledgement that holdout labels will be opened",
    )
    subparsers.add_parser("audit", help="hash-verify the completed evaluation package")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "predict":
        result = build_blind_holdout_evidence(device=args.device)
    elif args.command == "score":
        result = score_holdout(evaluation_unlocked=args.evaluation_unlocked)
    else:
        result = load_verified_holdout_evaluation()
    summary = {
        key: result[key]
        for key in ("evaluation_id", "phase", "status", "labels_opened", "coverage")
        if key in result
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
