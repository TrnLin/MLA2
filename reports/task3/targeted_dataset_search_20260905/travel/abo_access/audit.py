"""Audit three already-downloaded ABO images against official metadata."""

from fashion.task3_paths import resolve_task3_path

import csv
import gzip
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
BASE = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/"
ids = {"B07255FKTS", "B0727R3L19", "B07PGMRKSC"}
catalogue = ROOT.parents[1] / "catalogue/raw/abo_shard_0.bin"
with gzip.open(catalogue, "rt") as f:
    products = [json.loads(line) for line in f]
products = [r for r in products if r["item_id"] in ids]
with gzip.open(ROOT / "images.csv.gz", "rt") as f:
    images = {r["image_id"]: r for r in csv.DictReader(f)}
result = []
for p in products:
    entry = images[p["main_image_id"]]
    local = resolve_task3_path(p["item_id"] + ".jpg", root=ROOT)
    with Image.open(local) as image:
        image.load()
        size = list(image.size)
    result.append(
        {
            "item_id": p["item_id"],
            "main_image_id": p["main_image_id"],
            "image_index_row": entry,
            "url": BASE + "images/original/" + entry["path"],
            "local_file": local.name,
            "bytes": local.stat().st_size,
            "sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
            "decoded_width_height": size,
            "dimensions_match_index": size == [int(entry["width"]), int(entry["height"])],
            "original_item_name": p.get("item_name"),
            "original_bullet_point": p.get("bullet_point"),
            "product_type": p.get("product_type"),
            "visually_inspected": True,
            "teacher_duplicate_check": False,
            "independence_established": False,
        }
    )
audit = {
    "observed_date": "2026-09-05",
    "index_url": BASE + "images/metadata/images.csv.gz",
    "index_bytes": (ROOT / "images.csv.gz").stat().st_size,
    "index_sha256": hashlib.sha256((ROOT / "images.csv.gz").read_bytes()).hexdigest(),
    "downloaded_image_bytes": sum(r["bytes"] for r in result),
    "images": result,
}
(ROOT / "access_evidence.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
