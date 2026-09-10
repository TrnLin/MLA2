"""Run one development-safe Task 4 image search."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from fashion.config import ROOT
from fashion.task4.search import (
    CropBox,
    load_search_bundle,
    run_search,
    write_search_outputs,
)

MODEL_PACKAGE = ROOT / "models/task4_r5"
GALLERY_DIRECTORY = ROOT / "models/task4_teacher_gallery"
SPLITS_PATH = ROOT / "data/processed/splits.csv"
FIGURE_DIRECTORY = ROOT / "results/figures/task4/test-platform"
EVIDENCE_DIRECTORY = ROOT / "results/evidence/task4/test-platform"


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, json.dumps({"error": message}) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(
        description=(
            "Search only the development fold. Protected data stays closed: this command "
            "never opens holdout, quarantine, or official teacher-test images."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--query-id", type=int)
    source.add_argument("--image", type=Path)
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
    )
    parser.add_argument("--top-k", type=int, choices=range(1, 21), default=5)
    parser.add_argument("--rating", choices=("good", "mixed", "bad"))
    parser.add_argument("--note")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--evidence-directory", type=Path, default=EVIDENCE_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        crop = CropBox(*arguments.crop) if arguments.crop is not None else None
        bundle = load_search_bundle(
            model_package=MODEL_PACKAGE,
            gallery_directory=GALLERY_DIRECTORY,
            splits_path=SPLITS_PATH,
            device=arguments.device,
        )
        response = run_search(
            bundle,
            image_path=arguments.image,
            query_id=arguments.query_id,
            crop=crop,
            top_k=arguments.top_k,
            rating=arguments.rating,
            note=arguments.note,
        )
        png_path, json_path = write_search_outputs(
            response,
            figure_directory=arguments.figure_directory,
            evidence_directory=arguments.evidence_directory,
        )
        summary = {
            "query_key": response.record.query_key,
            "top_k": response.record.top_k,
            "png_path": str(png_path),
            "json_path": str(json_path),
            "warnings": list(response.record.query.warnings),
            "results": [hit.to_dict() for hit in response.record.results],
        }
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ArithmeticError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
