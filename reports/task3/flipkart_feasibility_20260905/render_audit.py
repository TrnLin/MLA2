"""Render checked source samples, summary plots, and readable audit pages."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mistune
from PIL import Image, ImageOps

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
FIG = ROOT / "results/figures/task3/flipkart_feasibility"


def csv_rows(name):
    return list(csv.DictReader((OUT / name).open()))


def contact_sheets():
    paths = []
    manifests = [
        ("broad", "sample_manifest.csv", "image_access.csv", 24),
        ("candidate", "candidate_access_manifest.csv", "candidate_image_access.csv", None),
    ]
    for prefix, manifest_name, results_name, limit in manifests:
        manifest_rows = csv_rows(manifest_name)
        results = {row["product_id"]: row for row in csv_rows(results_name)}
        selected = manifest_rows[:limit] if limit is not None else manifest_rows
        for start in range(0, len(selected), 12):
            page = selected[start : start + 12]
            nrows = (len(page) + 3) // 4
            fig, axes = plt.subplots(nrows, 4, figsize=(16, 4.35 * nrows), layout="constrained")
            for ax, record in zip(axes.ravel(), page):
                result = results[record["product_id"]]
                ax.axis("off")
                occasion = record.get("source_occasion", record.get("occasions", ""))
                label = record.get("teacher_article_type_candidate", record["root_category"])
                ax.set_title(
                    textwrap.fill(f"{label} | source: {occasion}", 37)
                    + "\n"
                    + record["product_id"],
                    fontsize=10,
                )
                if result["usable"] == "True":
                    with Image.open(resolve_task3_path(result["local_path"], root=ROOT)) as opened:
                        rgb = ImageOps.exif_transpose(opened).convert("RGB")
                        full = ImageOps.contain(rgb, (350, 285))
                        panel = Image.new("RGB", (480, 330), "#edf1f3")
                        panel.paste(full, ((350 - full.width) // 2, (330 - full.height) // 2))
                        small = rgb.resize((60, 80), Image.Resampling.BILINEAR)
                        panel.paste(small.resize((120, 160), Image.Resampling.NEAREST), (355, 85))
                        ax.imshow(panel)
                else:
                    ax.text(
                        0.5, 0.5, "Image unavailable", ha="center", va="center", color="#99552f"
                    )
            for ax in axes.ravel()[len(page) :]:
                ax.axis("off")
            fig.suptitle(
                f"Flipkart {prefix} access sample — page {start // 12 + 1}\n"
                "Left: source aspect ratio. Right: full image at 60×80, enlarged 2×. "
                "Source labels are displayed unchanged; no image labels are assigned here.",
                fontsize=14,
            )
            path = FIG / f"{prefix}_contact_{start // 12 + 1}.png"
            fig.savefig(path, dpi=100)
            plt.close(fig)
            paths.append(path)
    return paths


def summary_plot():
    rows = csv_rows("candidate_counts.csv")
    labels = [row["source_occasion"] for row in rows]
    counts = [int(row["metadata_group_representatives"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    colors = ["#287680" if value else "#adb4b7" for value in counts]
    bars = axes[0].barh(labels, counts, color=colors)
    axes[0].bar_label(bars, padding=5)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 2850)
    axes[0].set_xlabel("Metadata groups after conservative filtering")
    axes[0].set_title(
        "Source occasion names matching teacher names\nLabels remain unverified seller tags"
    )
    axes[0].spines[["top", "right"]].set_visible(False)
    noncasual = [(label, count) for label, count in zip(labels, counts) if label != "Casual"]
    bars = axes[1].barh(
        [label for label, _ in noncasual],
        [count for _, count in noncasual],
        color=["#287680" if count else "#adb4b7" for _, count in noncasual],
    )
    axes[1].bar_label(bars, padding=5)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 345)
    axes[1].set_xlabel("Metadata groups (Casual omitted for scale)")
    axes[1].set_title("A closer look at the smaller labels\nBest new counts: Formal and Party")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "candidate_counts.png", dpi=140)
    plt.close(fig)


STYLE = """
body {max-width:1120px; margin:32px auto; padding:0 24px 60px;
font:16px/1.55 Arial,sans-serif; color:#1b303a;}
h1,h2 {color:#205d66;} h2 {margin-top:32px;}
table {width:100%;border-collapse:collapse;font-size:14px;margin:20px 0;}
th,td {text-align:left;border:1px solid #cedade;padding:8px;}
th {background:#e6f0f1;} tr:nth-child(even) {background:#f6f8f9;}
img {max-width:100%;height:auto;display:block;margin:24px auto;}
a {color:#146b78;} code {font-size:.88em;overflow-wrap:anywhere;}
pre {background:#edf2f4;padding:14px;white-space:pre-wrap;}
"""


def render_pages():
    render = mistune.create_markdown(plugins=["table"])
    names = ["README", "REPORT", "COVERAGE", "TERMS_LABELS", "ACCESS", "CANDIDATE_ACCESS"]
    for name in names:
        path = OUT / f"{name}.md"
        if not path.exists():
            continue
        body = render(path.read_text())
        for target in names:
            body = body.replace(f'href="{target}.md"', f'href="{target}.html"')
        html = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Flipkart feasibility — {name}</title><style>{STYLE}</style>"
            f"<main>{body}</main></html>"
        )
        (OUT / f"{name}.html").write_text(html)
    rows = csv_rows("teacher_type_coverage.csv")
    heading = "# All 124 teacher product types\n\n"
    heading += "Counts are accepted metadata matches, not verified image labels.\n\n"
    table = (
        "| Teacher type | Teacher development rows | Source rows | With Occasion "
        "| Exact single shared name |\n"
    )
    table += "|---|---:|---:|---:|---:|\n"
    for row in rows:
        table += (
            f"| {row['teacher_article_type']} | {row['development_products']} "
            f"| {row['matched_product_rows']} | {row['source_occasion_present_rows']} "
            f"| {row['exact_single_shared_occasion_rows']} |\n"
        )
    body = render(heading + table)
    (OUT / "TYPE_COVERAGE.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>All teacher types — Flipkart</title><style>{STYLE}</style>"
        f"<main>{body}</main></html>"
    )


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summary_plot()
    paths = contact_sheets()
    render_pages()
    print(json.dumps({"contact_sheets": [str(path.relative_to(ROOT)) for path in paths]}))


if __name__ == "__main__":
    main()
