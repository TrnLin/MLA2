"""Record exact external rows proposed for replacement, with review sheets."""

from fashion.task3_paths import resolve_task3_path

import math
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent


def main():
    audit = pd.read_csv(OUT / "audit/external_687_audit.csv", dtype=str, keep_default_na=False)
    shortlist = pd.read_csv(
        OUT / "audit/retirement_shortlist_128.csv", dtype=str, keep_default_na=False
    )
    selections = []

    def select(rows, reason):
        rows = rows.copy()
        rows["reason"] = reason
        selections.append(rows)

    party = audit.loc[audit.usage.eq("Party")]
    select(
        party.loc[
            party.audit_product_type.isin(
                [
                    "Bras",
                    "Other shoes",
                    "Blazers",
                    "Caps",
                    "Kurtas",
                    "Lehenga",
                    "Sarees",
                    "Suits",
                    "Tshirts",
                ]
            )
            | party.id.eq("1000000068")
        ],
        "Source supports Party, but this form is absent from teacher Party development; "
        "replace with missing accessory/perfume/watch forms. Not a mislabel finding.",
    )
    dress_ids = [572, 579, 587, 597, 622, 628, 634, 658, 669, 673, 674, 676]
    select(
        party.loc[party.id.isin([str(1000000000 + i) for i in dress_ids])],
        "Visually reviewed bright/pale gown or styled-photo example; replace with a dark mini "
        "dress closer to teacher development catalogue views. Dress count stays fixed.",
    )
    smart = shortlist.loc[shortlist.usage.eq("Smart Casual")]
    select(
        smart.loc[smart.extension_family_group.eq("external_family_9ad485e18ea86c290d4f")],
        "Retire the entire 49-image conservative shoe family, whose text-based Smart Casual "
        "inference and single-fold concentration give little independent coverage; "
        "preserve other shoes.",
    )
    select(
        smart.loc[~smart.extension_family_group.eq("external_family_9ad485e18ea86c290d4f")],
        "Jacket/coat or polo form absent from teacher Smart Casual development; prioritize "
        "missing watches/wallets/ties/sandals and full shirt views. Not a wrong-label finding.",
    )
    select(
        audit.loc[audit.id.isin(["1000000292", "1000000312"])],
        "Additional jacket form absent from teacher Smart Casual development; free room for "
        "missing forms while retaining all existing shirts and trousers.",
    )
    travel = audit.loc[audit.usage.eq("Travel")]
    select(
        travel.loc[travel.audit_product_type.eq("Wheeled luggage / hybrid bags")],
        "Wheeled/hybrid shape absent from valid teacher Travel development images; replace "
        "with shoulder/tote or camera bags. Travel label is not disputed.",
    )
    select(
        travel.loc[travel.id.isin(["1000000476", "1000000509", "1000000510", "1000000511"])],
        "Reduce colour/graphic repetition in the 27-image Hugger family; remaining family "
        "members keep their saved fold. Replace with missing bag forms.",
    )
    select(
        audit.loc[
            audit.id.isin(
                [
                    "1000000422",
                    "1000000436",
                    "1000000333",
                    "1000000338",
                    "1000000340",
                    "1000000346",
                    "1000000376",
                ]
            )
        ],
        "Single rectangular pillow view; replace with reviewed printed covers and cover sets "
        "closer to the only teacher Home development image. Home label is not disputed.",
    )
    result = pd.concat(selections, ignore_index=True).sort_values(["usage", "id"])
    assert result.id.is_unique
    assert result.usage.value_counts().to_dict() == {
        "Smart Casual": 62,
        "Party": 44,
        "Travel": 17,
        "Home": 7,
    }
    result["visual_review_status"] = "pending_parent_contact_sheet_review"
    result.to_csv(OUT / "retirement_review.csv", index=False)
    font = ImageFont.load_default(size=12)
    heading = ImageFont.load_default(size=20)
    sheets = OUT / "retirement_contact_sheets"
    sheets.mkdir(exist_ok=True)
    for label, group in result.groupby("usage"):
        rows = group.to_dict("records")
        for start in range(0, len(rows), 32):
            batch = rows[start : start + 32]
            canvas = Image.new("RGB", (1280, 48 + math.ceil(len(batch) / 8) * 205), "#f3f4f5")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 10),
                f"Proposed retirements | {label} | {start + 1}-{start + len(batch)} of {len(rows)}",
                fill="black",
                font=heading,
            )
            for i, row in enumerate(batch):
                x, y = (i % 8) * 160, 48 + (i // 8) * 205
                with Image.open(resolve_task3_path(row["path"], root=ROOT)) as image:
                    image = image.convert("RGB")
                    image.thumbnail((116, 145))
                    canvas.paste(image, (x + (160 - image.width) // 2, y))
                draw.text((x + 5, y + 148), row["id"], fill="black", font=font)
                for j, line in enumerate(textwrap.wrap(row["audit_product_type"], 23)[:2]):
                    draw.text((x + 5, y + 164 + j * 14), line, fill="black", font=font)
            name = f"{label.lower().replace(' ', '_')}_{start // 32 + 1}.jpg"
            canvas.save(sheets / name, quality=95)
    print(result.usage.value_counts().to_dict())


if __name__ == "__main__":
    main()
