"""Read and verify the accepted Usage E8 refit without training or inference."""

import hashlib
import json
from pathlib import Path

from fashion.config import ROOT

PACK = Path("reports/task3/usage_final_e8_refit_20260907")
RUN_ID = "t3_usage_e8_translation_teacher_all_development_refit_5553be0c138243e9"
CHECKPOINT_SHA256 = "18da75a4ec935ab0d18c9ebdf9d1dabb6fa87ee484ead5f439de0192c4af2747"
CLASS_NAMES = [
    "Casual",
    "Ethnic",
    "Formal",
    "Home",
    "NA",
    "Party",
    "Smart Casual",
    "Sports",
    "Travel",
]


def check_hash(path: Path, expected: str) -> None:
    """Fail instead of silently reading a changed model or evidence file."""
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}")


def verify_usage_final(root: Path = ROOT) -> dict:
    """Verify the exact accepted checkpoint, recipe, sources and training records."""
    pack = root / PACK
    check_hash(pack / "model_manifest.json", (pack / "model_manifest.sha256").read_text().strip())
    manifest = json.loads((pack / "model_manifest.json").read_text())
    if (
        manifest["run_id"] != RUN_ID
        or manifest["checkpoint"]["sha256"] != CHECKPOINT_SHA256
        or manifest["class_names"] != CLASS_NAMES
        or manifest["selected_epoch"] != 30
        or manifest["artifact_type"] != "single_all_development_scratch_refit"
    ):
        raise ValueError("Final Usage model identity differs from the accepted E8 refit")
    for relative, expected in manifest["files"].items():
        check_hash(root / relative, expected)
    for key in ("checkpoint", "config", "normalization", "training_manifest"):
        entry = manifest[key]
        check_hash(root / entry["path"], entry["sha256"])
    config = json.loads((root / manifest["config"]["path"]).read_text())
    if (
        config["class_names"] != CLASS_NAMES
        or config["training_rows"] != 32772
        or config["training_scope"] != "all_eligible_teacher_development"
        or not config["scratch"]
        or config["validation_used"]
        or config["epochs"] != 30
        or config["checkpoint_rule"] != "final_epoch"
        or config["training_augmentation"] != "translation_uniform_2px_p05"
        or config["effective_loss_name"] != "effective_number_cross_entropy"
    ):
        raise ValueError("Final Usage training recipe differs from the frozen contract")
    return manifest


def verify_usage_holdout_sources(root: Path = ROOT) -> dict:
    """Check saved holdout evidence; this does not open raw labels or run inference."""
    lock = json.loads((root / PACK / "holdout_sources.json").read_text())
    for relative, expected in lock["files"].items():
        check_hash(root / relative, expected)
    return lock


if __name__ == "__main__":
    accepted = verify_usage_final()
    print(f"Verified final Usage E8 refit: {accepted['run_id']}")
    print(f"Verified {len(accepted['files'])} frozen model and training references.")
