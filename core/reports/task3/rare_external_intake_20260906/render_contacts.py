"""Readable, numbered original/prepared contact sheets for manual visual QA."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import csv
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/google-noto/NotoSans-Regular.ttf"


def render(rows, directory, prefix, per_page=20):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 18)
    small = ImageFont.truetype(FONT, 15)
    titlefont = ImageFont.truetype(FONT, 25)
    index = []
    for page in range(math.ceil(len(rows) / per_page)):
        batch = rows[page * per_page : (page + 1) * per_page]
        cols, cell_w, cell_h = 4, 370, 350
        canvas = Image.new(
            "RGB", (cols * cell_w, 80 + math.ceil(len(batch) / cols) * cell_h), "white"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (16, 10),
            f"{prefix}: original photo + 60×80 view — page {page + 1}",
            fill="black",
            font=titlefont,
        )
        draw.text(
            (16, 46),
            "Visual selection only. Source text supplies labels; "
            "teacher labels are not inferred from photos.",
            fill="#333333",
            font=small,
        )
        for pos, row in enumerate(batch):
            number = page * per_page + pos + 1
            x, y = pos % cols * cell_w, 80 + pos // cols * cell_h
            draw.rectangle((x + 5, y + 3, x + cell_w - 5, y + cell_h - 4), outline="#cccccc")
            draw.text((x + 12, y + 10), f"{number}. {row['source_id']}", fill="black", font=font)
            with Image.open(resolve_task3_path(row["original_path"], root=ROOT)) as im:
                original = ImageOps.exif_transpose(im).convert("RGB")
                thumbnail = ImageOps.contain(original, (222, 215), Image.Resampling.LANCZOS)
                if row.get("path"):
                    with Image.open(resolve_task3_path(row["path"], root=ROOT)) as prepared:
                        preview = prepared.convert("RGB")
                else:
                    preview = ImageOps.pad(
                        original, (60, 80), method=Image.Resampling.LANCZOS, color="white"
                    )
            canvas.paste(
                thumbnail,
                (x + 12 + (222 - thumbnail.width) // 2, y + 45 + (215 - thumbnail.height) // 2),
            )
            canvas.paste(preview.resize((120, 160), Image.Resampling.NEAREST), (x + 242, y + 70))
            lines = textwrap.wrap(row["source_title"], width=42)[:3]
            for i, line in enumerate(lines):
                draw.text((x + 12, y + 266 + i * 20), line, fill="black", font=small)
            draw.text((x + 12, y + 327), row["source_label"], fill="#125a49", font=small)
            index.append(
                {
                    "contact_number": number,
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "sheet": f"{prefix}_{page + 1:02}.png",
                }
            )
        canvas.save(directory / f"{prefix}_{page + 1:02}.png")
    with (directory / f"{prefix}_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["contact_number", "source", "source_id", "sheet"]
        )
        writer.writeheader()
        writer.writerows(index)
    return index


if __name__ == "__main__":
    with (OUT / "sources/amazon/candidates.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["source_label"] == "Home"]
    rows.sort(key=lambda r: r["source_id"])
    render(rows, OUT / "sources/amazon/contact_sheets", "home")
