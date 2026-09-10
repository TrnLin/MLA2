"""Fetch small public metadata only. No image or teacher-data access."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = "https://huggingface.co/datasets/tsazan/ikea-us-commercetxt/raw/main/"
records = []


def fetch(url, name):
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise RuntimeError("Small metadata cap exceeded")
    (OUT / name).write_bytes(data)
    records.append(
        {"url": url, "file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    )
    return data.decode()


stats = {}
for category in ["throw-pillow-covers", "accent-and-throw-pillows"]:
    body = fetch(BASE + "categories/" + category + ".txt", category + ".txt")
    paths = re.findall(r"(/products/[^\s]+\.txt)", body)
    ids = [p.rsplit("/", 1)[-1][:-4] for p in paths]
    stats[category] = {
        "declared_product_count": int(re.search(r"ProductCount: (\d+)", body)[1]),
        "listed_rows": len(paths),
        "unique_product_ids": len(set(ids)),
    }
    for path in paths[:2]:
        fetch(BASE + path.lstrip("/"), category + "_" + path.rsplit("/", 1)[-1])
fetch(
    "https://raw.githubusercontent.com/yumingj/DeepFashion-MultiModal/main/README.md",
    "deepfashion_multimodal_README.md",
)
(OUT / "metadata_checks.json").write_text(
    json.dumps(
        {
            "observed_date": "2026-09-05",
            "downloads": records,
            "total_download_bytes": sum(x["bytes"] for x in records),
            "category_counts": stats,
            "note": (
                "CommerceTXT is derivative of Jeffrey Zhou IKEA scrape, "
                "not a second independent source. Counts are products, "
                "not independently deduplicated families."
            ),
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(stats, indent=2))
