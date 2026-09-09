"""Render the targeted source review and its bounded source-image sample."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mistune
from PIL import Image, ImageOps

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "pyproject.toml").is_file())
FIG = ROOT / "results/figures/task3/targeted_dataset_search"
CSS = """
body{margin:0;background:#f2f4f5;color:#172a36;font:17px/1.58 system-ui,sans-serif}
main{max-width:1120px;margin:28px auto;padding:30px 42px;background:white;
border:1px solid #dde3e6;border-radius:10px}
h1{font-size:32px;line-height:1.2;margin-top:0}h2{font-size:24px;margin-top:32px}h3{font-size:20px}
a{color:#006b80;overflow-wrap:anywhere}
code{font-size:.88em;overflow-wrap:anywhere;background:#f0f3f5;padding:1px 4px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f5;padding:14px}pre code{padding:0}
table{width:100%;border-collapse:collapse;font-size:15px;line-height:1.45;table-layout:fixed}
main[data-page="README.md"] th:first-child{width:13%}
main[data-page="README.md"] th:nth-child(2){width:24%}
main[data-page="REPORT.md"] th:first-child{width:12%}
main[data-page="REPORT.md"] th:nth-child(2){width:21%}
main[data-page="catalogue/FINDINGS.md"] th:first-child{width:55%}
td,th{text-align:left;vertical-align:top;padding:10px 12px;
border:1px solid #d7e0e4;overflow-wrap:anywhere}
th{background:#e8f2f3}tr:nth-child(even){background:#f7f9fa}img{max-width:100%;height:auto}
li{margin:6px 0}
blockquote{margin:16px 0;padding:4px 20px;border-left:4px solid #288496;background:#f5f9fa}
@media(max-width:800px){main{margin:0;padding:20px}body{font-size:16px}
td,th{padding:7px}h1{font-size:27px}}
"""


def contact_sheets():
    records = json.loads((OUT / "catalogue/image_access.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(records), 4):
        fig, axes = plt.subplots(1, 4, figsize=(16, 7.0))
        for index, (ax, record) in enumerate(zip(axes, records[start : start + 4])):
            ax.set_position([0.012 + index * 0.25, 0.21, 0.235, 0.55])
            ax.axis("off")
            ax.set_title(
                "\n".join(
                    (
                        record["source"],
                        record["id"],
                        textwrap.fill(record["signal"].replace("_", " "), 25),
                    )
                ),
                fontsize=11,
            )
            if record["usable"]:
                with Image.open(OUT / record["saved_path"]) as opened:
                    rgb = ImageOps.exif_transpose(opened).convert("RGB")
                    large = ImageOps.contain(rgb, (285, 390))
                    panel = Image.new("RGB", (425, 430), "#edf1f3")
                    panel.paste(large, ((285 - large.width) // 2, (430 - large.height) // 2))
                    tiny = rgb.resize((60, 80), Image.Resampling.BILINEAR)
                    panel.paste(tiny.resize((120, 160), Image.Resampling.NEAREST), (298, 135))
                    ax.imshow(panel)
            else:
                ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
            ax.text(
                0.5,
                -0.025,
                textwrap.fill(textwrap.shorten(record["title"], width=98, placeholder="…"), 35),
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
            )
        fig.suptitle(
            "Selected source image links — access check only\n"
            "Left: original aspect ratio. Right: full photo at 60×80, enlarged 2×. "
            "No teacher labels assigned.",
            fontsize=14,
            y=0.98,
        )
        fig.savefig(FIG / f"catalogue_sample_{start // 4 + 1}.png", dpi=110)
        plt.close(fig)


def main():
    contact_sheets()
    markdown = mistune.create_markdown(plugins=["table"])
    rendered = []
    for rel in (
        "README.md",
        "REPORT.md",
        "catalogue/FINDINGS.md",
        "home_na/findings.md",
        "smart_casual/FINDINGS.md",
        "travel/REPORT.md",
        "travel/ABO_ACCESS.md",
    ):
        path = OUT / rel
        if not path.exists():
            continue
        body = markdown(path.read_text())
        target = path.with_suffix(".html")
        target.write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{path.stem} — targeted data search</title><style>{CSS}</style>"
            f'<main data-page="{rel}">{body}</main></html>'
        )
        rendered.append(str(target.relative_to(OUT)))
    print(
        json.dumps(
            {
                "html": rendered,
                "figures": [str(p.relative_to(ROOT)) for p in sorted(FIG.glob("*.png"))],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
