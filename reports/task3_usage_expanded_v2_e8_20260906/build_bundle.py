"""Package the new Usage trainer, admitted images and frozen comparison evidence."""

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

REPORT = ROOT / "reports/task3_usage_expanded_v2_e8_20260906"
BUNDLE = REPORT / "teacher_plus_rare_usage_v2_training.zip"
NOTEBOOK = Path("notebooks/04al_task3_usage_expanded_v2_e8.ipynb")
REFERENCES = Path("reference/usage_expanded_v2")
PREVIOUS_DIRECTORY = ROOT / "results/evidence/task3/usage_expanded_e8_20260906"
REFERENCE_FILES = (
    "config.json",
    "metrics.json",
    "normalization.json",
    "history.csv",
    "oof_predictions.csv",
    "robustness.csv",
    "final_epoch.pt",
)


def main():
    splits, contract = v2.validate_dataset(check_images=True)
    references = v2.check_references(
        e8_directory=ROOT / "results/evidence/task3" / previous.E8_DIRECTORY,
        source_registry_path=ROOT / "results/runs.csv",
        previous_directory=PREVIOUS_DIRECTORY,
        previous_registry_path=PREVIOUS_DIRECTORY / "results/runs.csv",
    )
    source_scores = {}
    for name, folds in references.items():
        predictions = pd.concat([folds[f]["predictions"] for f in range(5)], ignore_index=True)
        if name == "previous_expansion":
            predictions = predictions.loc[predictions.source_dataset.eq("teacher")]
        source_scores[name] = {
            "rows": len(predictions),
            "macro_f1": v2.oof_metrics(predictions, v2.CLASSES)["macro_f1"],
            "run_ids": [folds[f]["run_id"] for f in range(5)],
        }
    v2.write_json(
        {
            "dataset": contract,
            "references": source_scores,
            "all_development_images_decoded_and_hash_checked": len(
                splits.loc[splits.partition.eq("development") & splits.has_usage_label]
            ),
            "protected_images_not_read": True,
            "production_training_started": False,
        },
        REPORT / "preflight.json",
    )

    paths = {path.relative_to(ROOT) for path in (ROOT / "src/fashion").rglob("*.py")}
    paths.update(
        {
            NOTEBOOK,
            Path("pyproject.toml"),
            Path("data/processed/splits.csv"),
            Path("data/processed/label_maps.json"),
            previous.SOURCE_MANIFEST,
            previous.DATA_DIRECTORY / "splits.csv",
            previous.DATA_DIRECTORY / "label_maps.json",
            Path("docs/decisions/0019-task3-text-supported-usage-expansion.md"),
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
            cell["execution_count"] = None
            cell["outputs"] = []
    payload[str(NOTEBOOK)] = (json.dumps(notebook, indent=1) + "\n").encode()
    for name, directory, registry_path, run_ids, extra in (
        (
            "teacher_e8",
            ROOT / "results/evidence/task3" / previous.E8_DIRECTORY,
            ROOT / "results/runs.csv",
            v2.E8_RUN_IDS,
            (),
        ),
        (
            "previous_expansion",
            PREVIOUS_DIRECTORY,
            PREVIOUS_DIRECTORY / "results/runs.csv",
            v2.PREVIOUS_RUN_IDS,
            ("training_predictions.csv", "source_metrics.json"),
        ),
    ):
        registry = pd.read_csv(registry_path, keep_default_na=False, dtype=str)
        payload[str(REFERENCES / f"{name}_runs.csv")] = (
            registry.loc[registry.run_id.isin(run_ids)].to_csv(index=False).encode()
        )
        for run_id in run_ids:
            for filename in (*REFERENCE_FILES, *extra):
                payload[str(REFERENCES / name / run_id / filename)] = (
                    directory / run_id / filename
                ).read_bytes()
    manifest = {
        "purpose": "Scratch E8 training on teacher images plus all 687 admitted Usage images",
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": contract,
        "recipe": v2.expanded_usage_spec().to_dict(),
        "teacher_images": "Reuse MyDrive/MLA2/data/task3-data.zip",
        "references": source_scores,
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
        assert len(archive.namelist()) == len(payload) + 1
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
        assert archive.testzip() is None
    partial.replace(BUNDLE)
    receipt = {
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "bytes": BUNDLE.stat().st_size,
        "files": len(payload) + 1,
        "sha256": v2.compute_sha256(BUNDLE),
        "all_archive_file_hashes_verified": True,
        "production_training_started": False,
    }
    v2.write_json(receipt, REPORT / "bundle_receipt.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
