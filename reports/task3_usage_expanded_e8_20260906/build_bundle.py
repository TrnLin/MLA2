"""Build a self-contained Colab training snapshot without committing or pushing."""

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fashion.config import ROOT
from fashion.train.task3_usage_expanded import (
    DATA_DIRECTORY,
    E8_DIRECTORY,
    E8_RUN_IDS,
    SOURCE_MANIFEST,
    check_e8_sources,
    expanded_usage_spec,
    validate_dataset,
    write_json,
)

REPORT = ROOT / "reports/task3_usage_expanded_e8_20260906"
BUNDLE = REPORT / "teacher_plus_rare_usage_training.zip"
NOTEBOOK = Path("notebooks/04ag_task3_usage_expanded_e8.ipynb")


def write_repair_script():
    """The failed session can run this small file without replacing its frozen ZIP."""
    source = (ROOT / "src/fashion/train/task3_usage_registry.py").read_text()
    entry = """

if __name__ == "__main__":
    repair_connected_usage_session(
        globals(),
        expected_completed_folds=(0, 1),
        interrupted_run_ids=(
            "t3_usage_expanded_e8_usage_smallcnn_f2_s2753_6eb56854d557_20260906T082459Z0a0f7d",
        ),
    )
"""
    (REPORT / "repair_usage_registry.py").write_text(source + entry)


def main():
    write_repair_script()
    splits, contract = validate_dataset(check_images=False)
    check_e8_sources(
        directory=ROOT / "results/evidence/task3" / E8_DIRECTORY,
        registry_path=ROOT / "results/runs.csv",
    )
    paths = {p.relative_to(ROOT) for p in (ROOT / "src/fashion").rglob("*.py")}
    paths.update(
        {
            NOTEBOOK,
            Path("pyproject.toml"),
            Path("data/processed/splits.csv"),
            Path("data/processed/label_maps.json"),
            SOURCE_MANIFEST,
        }
    )
    paths.update(p.relative_to(ROOT) for p in (ROOT / DATA_DIRECTORY).rglob("*") if p.is_file())
    paths.update(Path(p) for p in splits.loc[splits.source_dataset.ne("teacher"), "path"])
    evidence = Path("results/evidence/task3") / E8_DIRECTORY
    for run_id in E8_RUN_IDS:
        for name in (
            "config.json",
            "metrics.json",
            "normalization.json",
            "history.csv",
            "oof_predictions.csv",
            "robustness.csv",
            "final_epoch.pt",
        ):
            paths.add(evidence / run_id / name)
    payload = {str(path): (ROOT / path).read_bytes() for path in sorted(paths)}
    # The user's executed notebook stays untouched. Ship a clean run copy inside
    # the archive so old failure output cannot look like a result of this code.
    notebook = json.loads(payload[str(NOTEBOOK)])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    payload[str(NOTEBOOK)] = (json.dumps(notebook, indent=1) + "\n").encode()
    registry = pd.read_csv(ROOT / "results/runs.csv", keep_default_na=False)
    payload["reference/e8_runs.csv"] = (
        registry.loc[registry.run_id.isin(E8_RUN_IDS)].to_csv(index=False).encode()
    )
    manifest = {
        "purpose": "One Usage model, fresh E8 scratch training on teacher plus 120 rare images",
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": contract,
        "recipe": expanded_usage_spec().to_dict(),
        "teacher_images": "Use the existing MyDrive/MLA2/data/task3-data.zip",
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
        assert set(archive.namelist()) == set(payload) | {"training_bundle_manifest.json"}
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
        assert archive.testzip() is None
    partial.replace(BUNDLE)
    with BUNDLE.open("rb") as handle:
        bundle_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    receipt = {
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "bytes": BUNDLE.stat().st_size,
        "files": len(payload) + 1,
        "sha256": bundle_digest,
        "all_archive_file_hashes_verified": True,
        "model_training_performed": False,
    }
    write_json(receipt, REPORT / "bundle_receipt.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
