"""Build the verified Task 2 component handoff without unlocking final evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from fashion.config import ROOT
from fashion.task2.handoff import (
    audit_task2_artifacts,
    build_task2_handoff_evidence,
    load_verified_task2_handoff,
)
from fashion.task2.inference import load_season_bundle, predict_season
from fashion.train.artifacts import ArtifactVerificationError

DEFAULT_SMOKE_IMAGE = ROOT / "data/raw/teacher/train/images_train/1163.jpg"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the frozen Task 2 package and run one development-image inference "
            "smoke test. This command keeps Notebook 06 and holdout evaluation locked."
        )
    )
    parser.add_argument(
        "--smoke-image",
        type=Path,
        default=DEFAULT_SMOKE_IMAGE,
        help="One labelled development image listed in canonical splits.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Inference smoke device; CPU is the reproducible default.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build, reload, and summarize the locked component handoff."""
    arguments = _parser().parse_args(argv)
    try:
        audit = audit_task2_artifacts()
        bundle = load_season_bundle(device=arguments.device)
        prediction = predict_season(bundle, arguments.smoke_image)
        _, manifest_path = build_task2_handoff_evidence(audit, prediction)
        manifest, _, verified_audit, smoke = load_verified_task2_handoff(manifest_path)
    except (
        ArtifactVerificationError,
        FileNotFoundError,
        FloatingPointError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    summary = {
        "status": manifest["status"],
        "task2_component_ready": manifest["task2_component_ready"],
        "group_freeze_verified": manifest["group_freeze_verified"],
        "notebook_06_unlocked": manifest["notebook_06_unlocked"],
        "holdout_opened": manifest["holdout_opened"],
        "audit_checks_passed": int(verified_audit["status"].eq("PASS").sum()),
        "audit_checks_total": int(len(verified_audit)),
        "smoke_product_id": smoke["product_id"],
        "smoke_prediction": smoke["predicted_label"],
        "run_id": manifest["run_id"],
        "bundle_sha256": smoke["bundle_sha256"],
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "next_gate": manifest["next_gate"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
