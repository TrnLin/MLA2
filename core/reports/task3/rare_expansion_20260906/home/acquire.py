"""Acquire named home cushion products from the official, cached ABO catalogue."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import csv
import gzip
import hashlib
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
RAW = ROOT / "reports/task3/rare_external_intake_20260906/sources/abo/raw"
DATA = ROOT / "data/external/rare_usage_expansion_20260906/candidates/home"
BASE = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/"


def values(record, field):
    return [str(v.get("value", "")) for v in record.get(field, []) if isinstance(v, dict)]


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    selected = {}
    for path in sorted(RAW.glob("listings_?.json.gz")):
        with gzip.open(path, "rt") as source:
            for line in source:
                record = json.loads(line)
                names = record.get("item_name", [])
                title = next(
                    (v["value"] for v in names if v.get("language_tag", "").startswith("en")), ""
                )
                typ = ",".join(values(record, "product_type"))
                if typ not in {"PILLOW", "HOME", "HOME_BED_AND_BATH"}:
                    continue
                if not re.search(r"cushion|pillow", title, re.I):
                    continue
                if not re.search(r"cover|case|throw|decorative", title, re.I):
                    continue
                if re.search(
                    r"\b(pet|dog|cat|nursing|pregnan|maternity|neck|travel|orthop|seat|chair|bench|"
                    r"massage|duvet|quilt|sheet|bedspread|bed pillow|sleeping|sleep)\b",
                    title,
                    re.I,
                ):
                    continue
                if not record.get("main_image_id"):
                    continue
                selected.setdefault(record["item_id"], record)
    index = {}
    with gzip.open(RAW / "images.csv.gz", "rt") as source:
        wanted = {r["main_image_id"] for r in selected.values()}
        for row in csv.DictReader(source):
            if row["image_id"] in wanted:
                index[row["image_id"]] = row
    # Shuffle deterministically; choose different official images before repeat ASIN variants.
    records = sorted(
        selected.values(), key=lambda r: hashlib.sha256(r["item_id"].encode()).hexdigest()
    )
    chosen, image_ids = [], set()
    for record in records:
        image_id = record["main_image_id"]
        if image_id in image_ids or image_id not in index:
            continue
        image_ids.add(image_id)
        chosen.append(record)
        if len(chosen) == 180:
            break
    (OUT / "selected_native_records.json").write_text(
        json.dumps(chosen, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        f"Eligible products={len(selected)}, selected unique source images={len(chosen)}",
        flush=True,
    )

    def download(record):
        sid = record["item_id"]
        native = index[record["main_image_id"]]
        url = BASE + "images/original/" + native["path"]
        destination = DATA / (sid + Path(native["path"]).suffix)
        try:
            if not destination.exists():
                with urllib.request.urlopen(url, timeout=45) as response:
                    content = response.read(12_000_001)
                if len(content) > 12_000_000:
                    raise ValueError("12 MB per-image cap")
                destination.write_bytes(content)
            with Image.open(destination) as image:
                image.load()
                width, height = image.size
                if min(width, height) < 100:
                    raise ValueError("original is too small")
            title = next(
                v["value"]
                for v in record["item_name"]
                if v.get("language_tag", "").startswith("en")
            )
            description = " | ".join(
                values(record, "bullet_point") + values(record, "product_description")
            )
            model = "|".join(values(record, "model_number"))
            brand = "|".join(values(record, "brand"))
            return {
                "source_dataset": "amazon_berkeley_objects",
                "source_record_id": sid,
                "product_name": title,
                "description": description,
                "proposed_usage": "Home",
                "evidence_text": title,
                "evidence_basis": "product_text_inference",
                "product_url": "https://www.amazon.com/dp/" + sid,
                "image_url": url,
                "original_path": str(destination.relative_to(ROOT)),
                "file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "width": width,
                "height": height,
                "source_family_id": "abo:" + (brand + ":" + model if model else sid),
                "confidence": "high",
                "notes": (
                    "Home inferred from cushion/throw-pillow text; no official usage label. "
                    "Family grouping receives central image/name checks."
                ),
                "rights_basis": (
                    "ABO release licence CC BY 4.0; registry/paper conflict says CC BY-NC 4.0; "
                    "academic use with attribution. See original cached licence and report."
                ),
            }
        except Exception as exc:
            return {"source_record_id": sid, "image_url": url, "rejection_reason": str(exc)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, chosen))
    accepted = [r for r in results if "original_path" in r]
    rejected = [r for r in results if "original_path" not in r]
    frame = pd.DataFrame(accepted)
    frame.insert(0, "contact_number", range(1, len(frame) + 1))
    frame.to_csv(OUT / "all_downloaded.csv", index=False)
    pd.DataFrame(rejected, columns=["source_record_id", "image_url", "rejection_reason"]).to_csv(
        OUT / "download_rejected.csv", index=False
    )
    for start in range(0, len(frame), 36):
        page = Image.new("RGB", (1200, 1200), "#eaeaea")
        draw = ImageDraw.Draw(page)
        for n, row in enumerate(frame.iloc[start : start + 36].to_dict("records")):
            x, y = (n % 6) * 200, (n // 6) * 200
            with Image.open(resolve_task3_path(row["original_path"], root=ROOT)) as im:
                thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"), (184, 157))
            page.paste(thumb, (x + (200 - thumb.width) // 2, y + 4))
            draw.text(
                (x + 5, y + 164),
                f"{row['contact_number']:03d} {row['source_record_id']}",
                fill="black",
            )
            draw.text((x + 5, y + 181), row["product_name"][:28], fill="black")
        page.save(OUT / f"contact_{start // 36 + 1}.jpg", quality=93)
    print(f"Downloaded and decoded {len(frame)}; failed {len(rejected)}", flush=True)


if __name__ == "__main__":
    main()
