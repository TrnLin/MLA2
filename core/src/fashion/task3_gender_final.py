"""Verify the accepted Gender SAM25 refit and its separate holdout evidence."""

import hashlib
import json
from pathlib import Path

from fashion.config import ROOT

PACK = Path("reports/task3/gender_final_sam25_refit_20260907")
RUN_ID = "t3_gender_name_truth_mixup_alpha020_sam005_epoch25_refit_20260907T061926Z_db0fc1ee"
CHECKPOINT_SHA256 = "41a5f5ea027805e8edbbf667d080564776ca9873d9146e0dec15c0d4bec2b272"
CLASS_NAMES = ["Boys", "Girls", "Men", "Unisex", "Women"]


def _check_hash(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}")


def verify_gender_final(root: Path = ROOT) -> dict:
    """Check the exact accepted artifact without inference or reading holdout scores."""
    pack = root / PACK
    _check_hash(pack / "model_manifest.json", (pack / "model_manifest.sha256").read_text().strip())
    manifest = json.loads((pack / "model_manifest.json").read_text())
    if (
        manifest["status"] != "accepted_final"
        or manifest["run_id"] != RUN_ID
        or manifest["checkpoint"]["sha256"] != CHECKPOINT_SHA256
        or manifest["class_names"] != CLASS_NAMES
        or manifest["selected_epoch"] != 25
        or manifest["artifact_type"] != "single_all_development_scratch_refit"
        or manifest["inference"]["models"] != 1
    ):
        raise ValueError("Final Gender identity differs from the accepted SAM25 refit")
    for relative, expected in manifest["files"].items():
        _check_hash(root / relative, expected)
    for key in ("checkpoint", "config", "normalization", "training_manifest"):
        entry = manifest[key]
        _check_hash(root / entry["path"], entry["sha256"])
    config = json.loads((root / manifest["config"]["path"]).read_text())
    if (
        config["class_names"] != CLASS_NAMES
        or config["training_scope"] != "all_development"
        or not config["scratch"]
        or config["validation_used"]
        or config["early_stopping"]
        or config["epochs"] != 25
        or config["cosine_t_max"] != 30
        or config["checkpoint_rule"] != "final_epoch"
        or config["mixup_contract"]["training_rows"] != 32773
        or config["mixup_contract"]["policy"]["alpha"] != 0.2
        or config["sam_policy"]["rho"] != 0.05
    ):
        raise ValueError("Final Gender recipe differs from the frozen contract")
    return manifest


def verify_gender_holdout_sources(root: Path = ROOT) -> dict:
    """Check saved holdout evidence without opening protected raw labels."""
    pack = root / PACK
    _check_hash(
        pack / "holdout_sources.json", (pack / "holdout_sources.sha256").read_text().strip()
    )
    lock = json.loads((pack / "holdout_sources.json").read_text())
    for relative, expected in lock["files"].items():
        _check_hash(root / relative, expected)
    return lock


if __name__ == "__main__":
    accepted = verify_gender_final()
    print(f"Verified final Gender SAM25 refit: {accepted['run_id']}")
