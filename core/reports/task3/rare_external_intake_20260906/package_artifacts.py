"""Build the local prepared-data bundle and readable HTML evidence."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import hashlib
import json
import zipfile
from pathlib import Path

import mistune
import pandas as pd
from render_contacts import render

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/external/rare_usage_20260906"
COMBINED = ROOT / "data/processed/teacher_plus_rare_usage_20260906"
CSS = """
body{margin:0;background:#f2f5f7;color:#172b35;font:17px/1.55 system-ui,sans-serif}
main{max-width:1060px;margin:26px auto;padding:32px 40px;background:#fff;
border:1px solid #dce3e8;border-radius:10px}
h1{font-size:32px;line-height:1.2;margin-top:0}h2{font-size:23px;margin-top:30px}
a{color:#006b80;overflow-wrap:anywhere}p{margin:15px 0}
code{font-size:.87em;overflow-wrap:anywhere;background:#f1f4f6;padding:1px 3px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f6;padding:14px}
table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:16px}
th,td{border:1px solid #d9e2e7;text-align:left;padding:10px 12px;vertical-align:top}
th{background:#e5f1f1}tr:nth-child(even){background:#f8fafb}th:first-child{width:26%}
img{max-width:100%;height:auto}li{margin:7px 0}
@media(max-width:800px){main{padding:18px;margin:0}table{font-size:14px}th,td{padding:6px}}
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = pd.read_csv(DATA / "splits.csv", keep_default_na=False)
    merged = pd.read_csv(COMBINED / "splits.csv", keep_default_na=False, low_memory=False)
    folds = merged.loc[merged.source_dataset.ne("teacher")].set_index("external_id").cv_fold
    rows = manifest.to_dict("records")
    for row in rows:
        row["source_label"] += f" | CV fold {int(folds.loc[row['external_id']])}"
    figure_dir = ROOT / "results/figures/task3/rare_external_intake_20260906"
    render(rows, figure_dir, "combined_accepted")
    figures = sorted(figure_dir.glob("combined_accepted_*.png"))
    gallery = [
        "# All accepted images\n",
        "Each row shows the original photo and the actual saved 60×80 PNG enlarged 2×. "
        "Names and labels come from the source. All 120 prepared images were visually inspected. "
        "Captions show their fold in the combined teacher + rare-class dataset. "
        "See [source attribution](ATTRIBUTION.md) and [the intake summary](README.md).\n",
    ]
    for index, figure in enumerate(figures, 1):
        gallery.append(
            f"\n## Page {index}\n\n![Original and prepared images]"
            f"(../../{figure.relative_to(ROOT)})\n"
        )
    (OUT / "GALLERY.md").write_text("\n".join(gallery))
    (DATA / "README.md").write_text(
        "# Rare Usage image cache and source records\n\n"
        "120 RGB PNG images, width 60 × height 80. Training now uses the combined teacher + "
        "rare-class dataset at data/processed/teacher_plus_rare_usage_20260906/splits.csv. "
        "Use its direct usage labels and the normal FashionDataset. The source-only split "
        "and statistics in this cache are acquisition evidence, not the training contract.\n\n"
        "Full [instructions](../../../reports/task3/rare_external_intake_20260906/USAGE.md), "
        "[evidence](../../../reports/task3/rare_external_intake_20260906/README.md) and "
        "[attribution](../../../reports/task3/rare_external_intake_20260906/ATTRIBUTION.md).\n"
    )
    renderer = mistune.create_markdown(plugins=["table"])
    for name in ("README", "USAGE", "GALLERY", "ATTRIBUTION"):
        body = renderer((OUT / (name + ".md")).read_text())
        (OUT / (name + ".html")).write_text(
            f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{name} — rare source intake</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>"
        )
    files = [resolve_task3_path(str(p), root=ROOT) for p in manifest.path]
    files += [
        DATA / name
        for name in (
            "README.md",
            "splits.csv",
            "validation.json",
        )
    ]
    files += [
        OUT / name
        for name in (
            "README.md",
            "USAGE.md",
            "ATTRIBUTION.md",
            "GALLERY.md",
            "validation.json",
            "combined_validation.json",
            "build_combined_dataset.py",
            "verify_ready.py",
            "visual_decisions.csv",
            "withheld.csv",
            "sources/abo/ATTRIBUTION.md",
            "sources/abo/raw/LICENSE-CC-BY-4.0.txt",
        )
    ]
    files += [
        resolve_task3_path(name, root=ROOT)
        for name in (
            "src/fashion/data/external_usage.py",
            "src/fashion/data/external_usage_audit.py",
            "src/fashion/data/expanded_usage.py",
            "src/fashion/data/dataset.py",
            "tests/data/test_external_usage.py",
            "tests/data/test_external_usage_audit.py",
            "tests/data/test_expanded_usage.py",
            "docs/decisions/0018-task3-external-source-label-intake.md",
        )
    ]
    files += figures
    files += [path for path in COMBINED.rglob("*") if path.is_file()]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    bundle = OUT / "teacher_plus_rare_usage.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            entry = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 9, 6, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
        archive.writestr("prepared_bundle_hashes.json", json.dumps(hashes, indent=2) + "\n")
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        for path, expected in hashes.items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected
    # Keep the previously shared download link pointed at the corrected combined package.
    (OUT / "rare_usage_60x80.zip").write_bytes(bundle.read_bytes())
    report = {
        "bundle": bundle.name,
        "sha256": sha(bundle),
        "bytes": bundle.stat().st_size,
        "files": len(hashes),
        "image_files": len(manifest),
        "combined_dataset_rows": len(merged),
        "image_bytes": sum((resolve_task3_path(path, root=ROOT)).stat().st_size for path in manifest.path),
        "zip_hash_readback_passed": True,
        "scope": "Combined dataset package plus 120 new PNGs; uses existing project teacher images",
    }
    (OUT / "bundle.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
