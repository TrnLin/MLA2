"""Package the fresh two-fold Usage screen and its existing v2 comparison evidence."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fashion.config import ROOT
from fashion.train import task3_usage_expanded as previous
from fashion.train import task3_usage_expanded_v2 as v2
from fashion.train import task3_usage_mixup_sam as screen

REPORT = ROOT / "reports/task3_usage_mixup_sam_20260906"
BUNDLE = REPORT / "usage_mixup_sam_training.zip"
NOTEBOOK = Path("notebooks/04am_task3_usage_mixup_sam_screen.ipynb")
REFERENCE_ROOT = Path("reference/usage_mixup_sam")
BASELINE_DIRECTORY = ROOT / "results/evidence/task3/usage_expanded_v2_e8_20260906"


def main():
    splits, contract = v2.validate_dataset(check_images=True)
    references = screen.check_reference(
        directory=BASELINE_DIRECTORY,
        registry_path=BASELINE_DIRECTORY / "results/runs.csv",
        splits=splits,
    )
    predictions = pd.concat([references[f]["predictions"] for f in screen.FOLDS], ignore_index=True)
    baseline_scores = v2.source_metrics(predictions)
    preflight = {
        "dataset": contract,
        "folds": list(screen.FOLDS),
        "baseline_sources": baseline_scores,
        "baseline_run_ids": [references[f]["run_id"] for f in screen.FOLDS],
        "all_development_images_decoded_and_hash_checked": contract["usage_development_rows"],
        "reference_weights_not_loaded": True,
        "production_training_started": False,
    }
    v2.write_json(preflight, REPORT / "preflight.json")
    paths = {p.relative_to(ROOT) for p in (ROOT / "src/fashion").rglob("*.py")}
    paths.update(
        {
            NOTEBOOK,
            Path("pyproject.toml"),
            Path("data/processed/splits.csv"),
            Path("data/processed/label_maps.json"),
            previous.DATA_DIRECTORY / "splits.csv",
            Path("tests/train/test_task3_usage_mixup_sam.py"),
            Path("tests/train/test_task3_usage_expanded_v2.py"),
            REPORT.relative_to(ROOT) / "README.md",
            REPORT.relative_to(ROOT) / "preflight.json",
            Path(__file__).relative_to(ROOT),
        }
    )
    paths.update(
        path.relative_to(ROOT) for path in (ROOT / v2.DATA_DIRECTORY).rglob("*") if path.is_file()
    )
    paths.update(Path(path) for path in splits.loc[splits.source_dataset.ne("teacher"), "path"])
    payload = {str(path): (ROOT / path).read_bytes() for path in sorted(paths)}
    notebook = json.loads(payload[str(NOTEBOOK)])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            if cell["execution_count"] is not None or cell["outputs"]:
                raise ValueError("The deliverable notebook must not contain training outputs")
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
    registry = pd.read_csv(
        BASELINE_DIRECTORY / "results/runs.csv", keep_default_na=False, dtype=str
    )
    run_ids = [screen.BASELINE_RUN_IDS[f] for f in screen.FOLDS]
    selected = registry.loc[registry.run_id.isin(run_ids)]
    if len(selected) != 2:
        raise ValueError("The bundle requires exactly two verified baseline rows")
    payload[str(REFERENCE_ROOT / "baseline_runs.csv")] = selected.to_csv(index=False).encode()
    for run_id in run_ids:
        for filename in sorted(
            v2.DIAGNOSTIC_FILES | {"metrics.json", "final_epoch.pt", "oof_predictions.csv"}
        ):
            payload[str(REFERENCE_ROOT / "baseline" / run_id / filename)] = (
                BASELINE_DIRECTORY / run_id / filename
            ).read_bytes()
    manifest = {
        "purpose": "Fresh Usage MixUp + SAM models on exactly folds 0 and 4",
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": contract,
        "recipe": screen.screen_spec().to_dict(),
        "teacher_images": "Reuse MyDrive/MLA2/data/task3-data.zip",
        "files": {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()},
    }
    partial = BUNDLE.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, value in payload.items():
            archive.writestr(name, value)
        archive.writestr("training_bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(partial) as archive:
        assert len(archive.namelist()) == len(payload) + 1
        assert set(archive.namelist()) == set(payload) | {"training_bundle_manifest.json"}
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
        assert archive.testzip() is None
    partial.replace(BUNDLE)
    receipt = {
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "bytes": BUNDLE.stat().st_size,
        "files": len(payload) + 1,
        "sha256": compute_hash(BUNDLE),
        "all_archive_file_hashes_verified": True,
        "production_training_started": False,
    }
    v2.write_json(receipt, REPORT / "bundle_receipt.json")
    print(json.dumps(receipt, indent=2))


def compute_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


if __name__ == "__main__":
    main()
