"""Create the coverage receipt, offline photo review, figure and training delta ZIP."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import base64
import hashlib
import html
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fashion.data.external_usage import file_sha256

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REPORT = Path(__file__).resolve().parent
DATASET = ROOT / "data/processed/teacher_plus_rare_usage_v3_20260906"


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main():
    old = read(REPORT / "audit/external_687_audit.csv")
    retired = read(REPORT / "retirement_review.csv")
    new = read(REPORT / "accepted_manifest.csv")
    combined = read(DATASET / "splits.csv")
    after_old = old.loc[~old.id.isin(retired.id)]
    lookup = combined.loc[combined.external_id.ne("")].set_index("external_id")
    new["id"] = new.external_id.map(lookup.id)
    new["cv_fold"] = new.external_id.map(lookup.cv_fold)
    before_types = old.groupby(["usage", "audit_product_type"]).size()
    after_types = after_old.groupby(["usage", "audit_product_type"]).size()
    additions = new.groupby(["usage", "product_type"]).size()
    coverage = []
    for key in sorted(set(before_types.index) | set(additions.index)):
        coverage.append(
            {
                "usage": key[0],
                "audit_product_type": key[1],
                "before": int(before_types.get(key, 0)),
                "retired": int(before_types.get(key, 0) - after_types.get(key, 0)),
                "new": int(additions.get(key, 0)),
                "after": int(after_types.get(key, 0) + additions.get(key, 0)),
            }
        )
    pd.DataFrame(coverage).to_csv(REPORT / "type_coverage_before_after.csv", index=False)
    external = combined.loc[combined.source_dataset.ne("teacher")]
    counts = []
    for label in ["Home", "Party", "Smart Casual", "Travel"]:
        a, b = old.loc[old.usage.eq(label)], external.loc[external.usage.eq(label)]
        counts.append(
            {
                "usage": label,
                "images_before": len(a),
                "images_after": len(b),
                "families_before": a.extension_family_group.nunique(),
                "families_after": b.extension_family_group.nunique(),
                "largest_family_before": int(a.groupby("extension_family_group").size().max()),
                "largest_family_after": int(b.groupby("extension_family_group").size().max()),
            }
        )
    pd.DataFrame(counts).to_csv(REPORT / "family_coverage_before_after.csv", index=False)
    for version, frame in [("before", old), ("after", external)]:
        frame.groupby(["usage", "source_dataset", "cv_fold"]).size().rename(
            "images"
        ).reset_index().to_csv(REPORT / f"source_fold_counts_{version}.csv", index=False)

    def photo(path):
        raw = (resolve_task3_path(path, root=ROOT)).read_bytes()
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(raw).decode()

    def card(row, status):
        is_new = status == "New"
        title = row["product_name"] if is_new else row["productDisplayName"]
        kind = row["product_type"] if is_new else row["audit_product_type"]
        url = row["product_url"] if is_new else row["source_url"]
        evidence = row["evidence_text"] if is_new else row["reason"]
        return (
            f'<article data-status="{status}" data-usage="{html.escape(row["usage"])}">'
            f'<div class="tag">{status} · {html.escape(row["usage"])} · fold {row["cv_fold"]}</div>'
            f'<img src="{photo(row["path"])}" alt="{html.escape(title)}">'
            f"<h3>{html.escape(kind)}</h3><b>{html.escape(row['id'])}</b>"
            f"<p>{html.escape(title)}</p><details><summary>Source and reason</summary>"
            f'<p>{html.escape(evidence)}</p><a href="{html.escape(url)}">'
            "Product source</a></details></article>"
        )

    cards = "\n".join(card(row, "New") for row in new.to_dict("records"))
    cards += "\n" + "\n".join(card(row, "Retired") for row in retired.to_dict("records"))
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rare Usage replacement review</title>
<style>
body{font:16px/1.5 system-ui;margin:0;background:#eef2f5;color:#162d38}
main{max-width:1280px;margin:auto;padding:28px}h1{line-height:1.15}
header{max-width:920px}select{padding:10px;margin:8px 12px 20px 0}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}
article{background:white;border-radius:12px;padding:16px;overflow-wrap:anywhere}
article img{width:120px;height:160px;object-fit:contain;display:block;
margin:16px auto;image-rendering:pixelated}
h3{margin:8px 0;font-size:18px}.tag{color:#23675b;font-size:13px;font-weight:700}
p{font-size:14px}summary{cursor:pointer}details{font-size:14px}
article[hidden]{display:none}a{color:#135ba0}</style>
<main><header><h1>130 photos replaced to fill rare Usage gaps</h1>
<p>687 outside photos remain. The teacher rows and folds stay fixed.
New photos: Smart Casual 62, Party 44, Travel 17, Home 7.
These are product-type and photo coverage improvements; model improvement has not been tested.</p>
<p>Photos below are the exact 60×80 training files, enlarged 2×.
Source text supports the proposed Usage but may differ from the teacher's meaning.
Type and audience notes are audit fields, not training labels.
“Retired” means excluded from v3; old files and old datasets remain available.</p></header>
<label>Status <select id="status"><option>New</option><option>Retired</option>
<option>All</option></select></label>
<label>Usage <select id="usage"><option>All</option><option>Home</option><option>Party</option>
<option>Smart Casual</option><option>Travel</option></select></label><span id="count"></span>
<div id="grid">"""
        + cards
        + """</div></main><script>
function filter(){
  let count=0;
  document.querySelectorAll('article').forEach(a=>{
    a.hidden=!((statusSelect.value==='All'||a.dataset.status===statusSelect.value)&&
      (usageSelect.value==='All'||a.dataset.usage===usageSelect.value));
    if(!a.hidden)count++;
  });
  document.getElementById('count').textContent=count+' photos';
}
const statusSelect=document.getElementById('status');
const usageSelect=document.getElementById('usage');
statusSelect.onchange=filter;usageSelect.onchange=filter;filter();</script></html>"""
    )
    (REPORT / "photo_review.html").write_text(page)
    labels = [
        "Smart: watches",
        "Smart: wallets",
        "Smart: ties",
        "Smart: sandals",
        "Party: clutches",
        "Party: perfumes",
        "Party: watches",
        "Travel: camera bags",
    ]
    values = [35, 8, 4, 3, 18, 8, 6, 6]
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    fig.subplots_adjust(left=0.25, right=0.95, bottom=0.13, top=0.79)
    fig.patch.set_facecolor("#f5f7fa")
    ax.set_facecolor("#f5f7fa")
    bars = ax.barh(labels[::-1], values[::-1], color="#237c73")
    ax.bar_label(bars, padding=5, fontsize=12)
    ax.set_xlim(0, 40)
    ax.set_xlabel("New outside images in previously missing visual groups")
    fig.text(0.035, 0.95, "Better coverage with the same 687 outside images", fontsize=18)
    fig.text(
        0.035, 0.89, "130 replaced · teacher split fixed · no measured model gain yet", fontsize=12
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    figure = ROOT / "results/figures/task3/rare_usage_replacement_20260906.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    archive = REPORT / "teacher_plus_rare_usage_v3_delta.zip"
    files = set(DATASET.rglob("*"))
    files.update(resolve_task3_path(path, root=ROOT) for path in new.path)
    files.add(REPORT / "README.md")
    files.add(REPORT / "photo_review.html")
    files = sorted(path for path in files if path.is_file())
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for path in files:
            data = bundle.read(path.relative_to(ROOT).as_posix())
            assert hashlib.sha256(data).hexdigest() == file_sha256(path)
    receipt = {
        "archive": str(archive.relative_to(ROOT)),
        "sha256": file_sha256(archive),
        "files": len(files),
        "bytes": archive.stat().st_size,
        "verified_all_member_hashes": True,
        "requires_existing_teacher_and_v2_image_files": True,
        "contains_new_prepared_images": len(new),
        "contains_source_original_photos": False,
        "uploaded": False,
    }
    (REPORT / "package_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
