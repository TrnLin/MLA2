"""Collect product-supported Home and Travel candidates for visual review.

This script does not admit images or change a training split.
"""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/external/rare_usage_replacement_20260906/candidates"


def plain(value):
    return BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)


def fetch(url, path):
    if not path.exists():
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def product_row(source, product, url, usage, kind, evidence_file):
    description = plain(product.get("body_html"))
    return {
        "source_dataset": source,
        "source_record_id": str(product["id"]),
        "product_name": product["title"],
        "description": description,
        "proposed_usage": usage,
        "evidence_text": description,
        "evidence_basis": "product_text_inference",
        "product_url": url,
        "image_url": product["images"][0]["src"],
        "source_family_id": source + ":" + product["handle"],
        "confidence": "medium",
        "notes": "Source product text supports the proposed purpose; not a teacher label.",
        "rights_status": "copyright_permission_not_established",
        "product_type": kind,
        "source_evidence_path": str(evidence_file.relative_to(ROOT)),
    }


def travel_rows():
    path = OUT / "travel/pacsafe_catalog.json"
    products = json.loads(path.read_text())["products"]
    rows = []
    allowed = {"Crossbody Bags", "Totes"}
    for product in products:
        title = product["title"]
        desc = plain(product.get("body_html"))
        if product["product_type"] not in allowed:
            continue
        if re.search(r"hip pack|3.in.1 sling|totepack", title, re.I):
            continue
        if not re.search(r"travel|trips around the world", desc, re.I):
            continue
        kind = "Mobile Pouch" if "Tech Crossbody" in title else "Handbags"
        row = product_row(
            "Pacsafe official catalogue",
            product,
            "https://pacsafe.com/products/" + product["handle"],
            "Travel",
            kind,
            path,
        )
        row["evidence_text"] = " | ".join(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", desc)
            if re.search(r"travel|trips around the world", sentence, re.I)
        )
        rows.append(row)

    pages = [
        (
            "travelon_phone.html",
            "https://www.travelonbags.com/shop/packables-and-packing/packables/pi-everyway-phone-sling-belt-bag/43633.html",
            "Mobile Pouch",
        ),
        (
            "travelon_slim.html",
            "https://www.travelonbags.com/shop/world-travel-essentials-slim-crossbody-bag/43508-51T.html",
            "Mobile Pouch",
        ),
        (
            "travelon_roam_small.html",
            "https://www.travelonbags.com/shop/anti-theft-bags/anti-theft-roam-bags/roam-anti-theft-small-crossbody/43675.html",
            "Handbags",
        ),
        (
            "travelon_roam_medium.html",
            "https://www.travelonbags.com/shop/anti-theft-bags/anti-theft-roam-bags/roam-anti-theft-medium-crossbody/43677-516.html",
            "Handbags",
        ),
    ]
    for name, url, kind in pages:
        path = fetch(url, OUT / "travel" / name)
        soup = BeautifulSoup(path.read_text(), "html.parser")
        products = [
            json.loads(script.string)
            for script in soup.find_all("script", type="application/ld+json")
            if script.string
        ]
        product = next(p for p in products if p.get("@type") == "Product")
        description = plain(product.get("description"))
        image = product["image"]
        if isinstance(image, list):
            image = image[0]
        rows.append(
            {
                "source_dataset": "Travelon official catalogue",
                "source_record_id": product.get("model") or product["sku"],
                "product_name": product["name"],
                "description": description,
                "proposed_usage": "Travel",
                "evidence_text": description,
                "evidence_basis": "product_text_inference",
                "product_url": url,
                "image_url": image,
                "source_family_id": "travelon:" + str(product.get("model") or product["sku"]),
                "confidence": "medium",
                "notes": "Travel purpose from native product description. Phone-sized small bags "
                "annotated separately from larger handbags for the coverage audit.",
                "rights_status": "copyright_permission_not_established",
                "product_type": kind,
                "source_evidence_path": str(path.relative_to(ROOT)),
            }
        )
    path = OUT / "travel/oce_phone.html"
    soup = BeautifulSoup(path.read_text(), "html.parser")

    def meta(key):
        return soup.find("meta", property=key)["content"]

    rows.append(
        {
            "source_dataset": "OCE GEAR official catalogue",
            "source_record_id": "og2416",
            "product_name": meta("og:title"),
            "description": meta("og:description"),
            "proposed_usage": "Travel",
            "evidence_text": meta("og:description"),
            "evidence_basis": "product_text_inference",
            "product_url": "https://ocegear.com/products/oce-gear-rfid-travel-phone-sacoche-crossbody-pouch",
            "image_url": meta("og:image").replace("http://", "https://"),
            "source_family_id": "oce:og2416",
            "confidence": "medium",
            "notes": "Explicit travel phone pouch title and native product description.",
            "rights_status": "copyright_permission_not_established",
            "product_type": "Mobile Pouch",
            "source_evidence_path": str(path.relative_to(ROOT)),
        }
    )
    return rows


def home_rows():
    path = OUT / "home/swayam_covers.json"
    products = json.loads(path.read_text())["products"]
    rows, used = [], set()
    for product in products:
        if "cushion cover" not in product["title"].lower():
            continue
        # One image per print family, not repeated pack sizes or colour variants.
        handle = product["handle"]
        code = re.search(r"swayam-([\d_]+)-cushion", handle)
        family = code.group(1) if code else "crochet_3101"
        if family.startswith("5325"):
            family = "diamond_quilt"
        elif family.startswith("570"):
            family = "ethnic_floral_5700"
        elif family.startswith("572"):
            family = "animal_5720"
        if family in used:
            continue
        used.add(family)
        row = product_row(
            "Swayam official catalogue",
            product,
            "https://www.swayamindia.com/products/" + handle,
            "Home",
            "Cushion Covers",
            path,
        )
        row["source_family_id"] = "swayam:cover:" + family
        row["evidence_text"] = product["title"] + " | " + plain(product.get("body_html"))
        row["notes"] += (
            " Multi-cover set product; one image per conservative print family. "
            "Actual photo is reviewed separately. Same brand as teacher Home, "
            "but no teacher-product identity is admitted."
        )
        rows.append(row)
    return rows


def download(row):
    slug = row["proposed_usage"].lower()
    asset = hashlib.sha256(
        (row["source_dataset"] + ":" + row["source_record_id"]).encode()
    ).hexdigest()[:20]
    destination = DATA / slug / (asset + ".jpg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fetch(row["image_url"], destination)
    with Image.open(destination) as image:
        image.load()
        row["width"], row["height"] = image.size
    row["original_path"] = str(destination.relative_to(ROOT))
    row["file_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return row


def contact_sheet(rows, slug):
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
    for start in range(0, len(rows), 30):
        subset = rows[start : start + 30]
        sheet = Image.new("RGB", (1200, ((len(subset) + 5) // 6) * 235), "#f4f4f0")
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(subset):
            x, y = offset % 6 * 200, offset // 6 * 235
            with Image.open(resolve_task3_path(row["original_path"], root=ROOT)) as original:
                image = ImageOps.exif_transpose(original).convert("RGBA")
                image.thumbnail((185, 180))
                sheet.paste(image, (x + (200 - image.width) // 2, y), image)
            draw.text(
                (x + 3, y + 183),
                f"{start + offset:02d}: {row['product_type']}",
                fill="#22343a",
                font=font,
            )
            for line, text in enumerate([row["product_name"][:27], row["product_name"][27:54]]):
                draw.text((x + 3, y + 200 + line * 15), text, fill="#22343a", font=font)
        sheet.save(OUT / slug / f"candidates_{start // 30 + 1}.png")


def main():
    for slug, acquire in (("home", home_rows), ("travel", travel_rows)):
        candidates = acquire()
        with ThreadPoolExecutor(max_workers=6) as pool:
            rows = list(pool.map(download, candidates))
        pd.DataFrame(rows).to_csv(OUT / slug / "candidates.csv", index=False)
        contact_sheet(rows, slug)
        print(slug, len(rows), pd.DataFrame(rows).product_type.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
