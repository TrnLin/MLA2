"""Verify the selected Gender MixUp refit and its saved evaluation files."""

import json
from pathlib import Path

from fashion.config import ROOT
from fashion.task3_development import verify_mixup_selection
from fashion.task3_final import check_hash

PACK = Path("reports/task3/gender_mixup_selection_20260911/refit")
RUN_ID = "t3_gender_name_truth_mixup_alpha020_refit_20260911T041436Z_3af0b94e"
CHECKPOINT_SHA256 = "860f688162cccfcd903e8e4874b368697c0637b6e6a15baae3b4d4a3008f4ef9"
CLASS_NAMES = ["Boys", "Girls", "Men", "Unisex", "Women"]


def verify_gender_final(root: Path = ROOT) -> dict:
    """Check the selected checkpoint against its original training receipt."""
    return verify_mixup_selection(root)


def verify_gender_holdout_sources(root: Path = ROOT) -> dict:
    """Verify only the selected MixUp refit's saved evaluation outputs."""
    folder = Path("reports/task3/gender_mixup_refit_holdout_20260911")
    provenance = json.loads((root / folder / "evaluation_provenance.json").read_text())
    files = {str(folder / name): digest for name, digest in provenance["outputs"].items()}
    for relative, expected in files.items():
        check_hash(root / relative, expected)
    return {"scope": "selected Gender MixUp refit", "files": files}


if __name__ == "__main__":
    selected = verify_gender_final()
    print(f"Verified Gender MixUp refit: {selected['run_id']}")
