"""Export the selected Task 4 R5 checkpoint as a portable inference package."""

from __future__ import annotations

import argparse
from pathlib import Path

from fashion.config import ROOT
from fashion.task4 import export_r5_inference_package

SELECTED_RUN_ID = "task4-candidate-r5-task9-preexec"
SELECTED_CHECKPOINT_SHA256 = "521e96f3df9e28853309bd607030523f59ef8425326da56dfcf2b754e97631b3"
DEFAULT_CHECKPOINT = (
    ROOT / "results/evidence/task4/checkpoints/task4-candidate-r5-task9-preexec/epoch-080.pt"
)
DEFAULT_DESTINATION = ROOT / "models/task4_r5"
DEFAULT_ARCHIVE = ROOT / "models/task4_r5_portable.zip"
DEFAULT_CONTRACT = ROOT / "results/evidence/task4/preprocessing_contract.json"
DEFAULT_NORMALIZATION = ROOT / "results/evidence/task4/preprocessing_normalization_fold1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the selected scratch R5 model without training state."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--normalization", type=Path, default=DEFAULT_NORMALIZATION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package, archive = export_r5_inference_package(
        args.checkpoint,
        args.destination,
        preprocessing_contract_path=args.contract,
        normalization_path=args.normalization,
        archive_path=args.archive,
        expected_source_sha256=SELECTED_CHECKPOINT_SHA256,
        expected_run_id=SELECTED_RUN_ID,
    )
    print(f"Package: {package}")
    print(f"ZIP: {archive}")


if __name__ == "__main__":
    main()
