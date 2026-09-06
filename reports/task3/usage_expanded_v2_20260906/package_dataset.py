"""Render split evidence and package the new dataset plus all added RGB images."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fashion.data.external_usage import file_sha256
from fashion.data.usage_extension import DATASET_NAME

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REPORT = Path(__file__).resolve().parent
DATA = ROOT / "data/processed" / DATASET_NAME


def main():
    validation = json.loads((DATA / "validation.json").read_text())
    assert validation["passed"]
    folds = pd.read_csv(DATA / "fold_counts.csv")
    classes = pd.read_csv(DATA / "fold_class_counts.csv", keep_default_na=False)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.35]}
    )
    fig.patch.set_facecolor("#f7f8f5")
    left.axis("off")
    left.set_title("Usage training and validation", loc="left", fontsize=17, pad=18)
    table = left.table(
        cellText=[
            [str(row.fold), f"{row.train:,}", f"{row.validation:,}"] for row in folds.itertuples()
        ],
        colLabels=["Fold", "Train", "Validate"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0.2, 0.98, 0.64],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d6dfdf")
        cell.set_facecolor("#dfedeb" if row == 0 else "white")
    left.text(
        0,
        0.06,
        "39,299 rows  •  687 external images\n567 new images across 440 whole families",
        transform=left.transAxes,
        fontsize=12,
        color="#43585c",
    )
    rare = ["Home", "Party", "Smart Casual", "Travel"]
    colours = ["#259b88", "#c66242", "#4c6aaa", "#ac8331"]
    width = 0.18
    for index, (label, colour) in enumerate(zip(rare, colours, strict=True)):
        frame = classes.loc[classes.side.eq("validation") & classes.usage.eq(label)].sort_values(
            "fold"
        )
        right.bar(
            frame.fold + (index - 1.5) * width, frame.images, width=width, color=colour, label=label
        )
    right.set_title("Rare-class validation images", loc="left", fontsize=17, pad=18)
    right.set_xticks(range(5), [f"Fold {n}" for n in range(5)])
    right.set_ylabel("Images, including teacher and both added sets")
    right.spines[["top", "right"]].set_visible(False)
    right.grid(axis="y", alpha=0.15)
    right.set_axisbelow(True)
    right.legend(frameon=False, ncols=2, loc="upper right", fontsize=10)
    right.set_ylim(
        0, max(classes.loc[classes.side.eq("validation") & classes.usage.isin(rare), "images"]) + 25
    )
    fig.text(
        0.04,
        0.035,
        "Family safety comes first: the 49-image Smart Casual family stays in one fold. "
        "Earlier rows and folds are preserved.",
        fontsize=11,
        color="#43585c",
    )
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.20, top=0.87, wspace=0.3)
    fig.savefig(REPORT / "split_summary.png", dpi=140)
    plt.close(fig)
    (DATA / "README.md").write_text(
        "# Teacher + rare Usage v2\n\n"
        "39,299 rows: 38,612 teacher, 120 earlier additions, and 567 new images.\n\n"
        "Use splits.csv as the authority with the normal load_splits/get_cv_split APIs. "
        "train.csv and validation.csv are fold-0 Usage-only exports. The cv directory has "
        "all five fold ID lists. New images have Usage labels only.\n\n"
        "[Full instructions and checks]"
        "(../../../reports/task3/usage_expanded_v2_20260906/README.md).\n"
    )
    shutil.copyfile(
        ROOT / "data/external/rare_usage_expansion_20260906/manifest.csv",
        DATA / "new_source_evidence.csv",
    )
    old_evidence = ROOT / "data/external/rare_usage_20260906/splits.csv"
    shutil.copyfile(old_evidence, DATA / "previous_source_evidence.csv")
    added = pd.read_csv(DATA / "added_images.csv", keep_default_na=False)
    files = {p for p in DATA.rglob("*") if p.is_file()}
    files.update(resolve_task3_path(path, root=ROOT) for path in added.path)
    files.update(
        REPORT / name
        for name in [
            "README.md",
            "validation.json",
            "linked_external_families.csv",
            "family_edges.csv",
            "split_summary.png",
            "build_dataset.py",
            "audit_family_links.py",
            "package_dataset.py",
        ]
    )
    files.update(
        resolve_task3_path(name, root=ROOT)
        for name in [
            "src/fashion/data/usage_extension.py",
            "tests/data/test_usage_extension.py",
            "docs/decisions/0019-task3-text-supported-usage-expansion.md",
        ]
    )
    intake = ROOT / "reports/task3/rare_expansion_20260906"
    files.update(
        intake / name
        for name in [
            "README.md",
            "summary.json",
            "accepted_manifest.csv",
            "class_counts.csv",
            "prepared_contact_index.csv",
            "prepared_home.png",
            "prepared_party.png",
            "prepared_smart_casual.png",
            "prepared_travel.png",
            "home/visual_qa.json",
            "party/acquisition_log.md",
            "smart_casual/ACQUISITION_LOG.md",
            "travel/ACQUISITION_LOG.md",
            "internal_near_duplicate_matches.csv",
            "internal_duplicate_summary.json",
            "gallery.html",
            "preview.png",
        ]
    )
    files.update(p for p in (intake / "licenses").iterdir() if p.is_file())
    files.add(ROOT / "reports/task3/rare_external_intake_20260906/ATTRIBUTION.md")
    hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(files)}
    destination = REPORT / "teacher_plus_rare_usage_v2.zip"
    partial = destination.with_suffix(".zip.tmp")
    with ZipFile(partial, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            entry = ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 9, 6, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
        archive.writestr("usage_v2_bundle_hashes.json", json.dumps(hashes, indent=2) + "\n")
    with ZipFile(partial) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(hashes) + 1
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        assert all(row.path in archive.namelist() for row in added.itertuples())
    partial.replace(destination)
    result = {
        "bundle": str(destination.relative_to(ROOT)),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "files": len(hashes) + 1,
        "external_images": len(added),
        "total_dataset_rows": validation["total_rows"],
        "teacher_images_reused_from_existing_project": True,
        "archive_hashes_and_crc_verified": True,
        "dataset_split_sha256": validation["split_sha256"],
        "model_training_performed": False,
    }
    (REPORT / "bundle_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
