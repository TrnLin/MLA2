"""Save exact seller evidence and probe a small deterministic image sample."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

OUT = Path(__file__).resolve().parent
CAT = OUT / "catalogue"
BAG_LEAVES = {
    "Drawstring Bags",
    "Casual Daypacks",
    "Sports Duffels",
    "Satchels",
    "Totes",
    "Carry-Ons",
    "Hiking Daypacks",
    "Dry Bags",
    "Equipment Bags",
    "Mat Bags",
    "Backpacks",
    "Internal Frame Backpacks",
}


def amazon_rows(source):
    data = (CAT / "raw" / (source + ".bin")).read_bytes()
    for line in data[: data.rfind(b"\n")].splitlines():
        row = json.loads(line)
        if isinstance(row.get("details"), str):
            row["details"] = json.loads(row["details"])
        yield row


def image_url(row):
    images = row.get("images") or []
    main = next((r for r in images if r.get("variant") == "MAIN"), images[0] if images else {})
    return main.get("large") or main.get("hi_res") or main.get("thumb") or ""


def probe(case):
    path = CAT / "images" / (case["id"] + ".jpg")
    record_path = path.with_suffix(".json")
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["usable"]:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        return record
    record = dict(case, observed_on="2026-09-05", usable=False, byte_cap=1_000_000)
    try:
        with requests.get(case["url"], timeout=(15, 25), stream=True) as response:
            record["status"] = response.status_code
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_content(32768):
                if len(body) + len(chunk) > record["byte_cap"]:
                    raise ValueError("image exceeds sample cap")
                body.extend(chunk)
        image = Image.open(io.BytesIO(body))
        image.load()
        record.update(
            usable=True,
            bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            width=image.width,
            height=image.height,
            format=image.format,
            saved_path=str(path.relative_to(OUT)),
        )
        path.write_bytes(body)
    except (requests.RequestException, ValueError, OSError) as exc:
        record["error"] = str(exc)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    (CAT / "images").mkdir(exist_ok=True)
    candidates, pools = [], {"home_occasion": [], "travel_title": [], "sport_purpose": []}
    for source in ("amazon_home", "amazon_clothing", "amazon_sports"):
        for row in amazon_rows(source):
            details, cats = row.get("details") or {}, row.get("categories") or []
            rules = []
            if (
                cats
                and cats[-1] == "Throw Pillow Covers"
                and str(details.get("Occasion", "")).casefold() == "home"
            ):
                rules.append(("home_occasion", "details.Occasion", details["Occasion"]))
            if (
                cats
                and cats[-1] in BAG_LEAVES
                and re.search(r"\btravel(?:ling|ing)?\b", row["title"], re.I)
                and re.search(
                    r"\b(?:backpacks?|rucksacks?|duffels?|duffles?|handbags?)\b", row["title"], re.I
                )
            ):
                rules.append(("travel_title", "title", row["title"]))
            sport = details.get("Sport Type") or details.get("Sport")
            if (
                sport
                and "Fan Shop" not in cats
                and ("Clothing" in cats or "Shoes" in cats)
                and re.search(
                    r"\b(?:shirt|pants|shorts|sock|shoe|bra|jacket|cap|hat)s?\b", row["title"], re.I
                )
            ):
                rules.append(("sport_purpose", "details.Sport Type or Sport", str(sport)))
            for signal, field, value in rules:
                evidence = {
                    "source": source,
                    "id": row["parent_asin"],
                    "signal": signal,
                    "field": field,
                    "value": value,
                    "title": row["title"],
                    "categories": cats,
                    "image_url": image_url(row),
                    "teacher_usage_assigned": False,
                }
                candidates.append(evidence)
                pools[signal].append(evidence)
    cases = []
    for signal in pools:
        for row in sorted(pools[signal], key=lambda r: r["id"])[:2]:
            cases.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "signal": signal,
                    "title": row["title"],
                    "source_evidence": row["value"],
                    "url": row["image_url"],
                }
            )
    wardrobe_zip = OUT / "home_na/wardrobe_assistant.zip"
    with zipfile.ZipFile(wardrobe_zip) as archive:
        wardrobe = list(
            csv.DictReader(io.StringIO(archive.read("Wardrobe Assistant.csv").decode("utf-8-sig")))
        )
    for category in ("Men's Ethnic Wear", "Women's Ethnic Wear"):
        row = next(r for r in wardrobe if r["main_category"] == category)
        cases.append(
            {
                "id": "wardrobe_" + row[""],
                "source": "wardrobe_assistant",
                "signal": "source_category_only",
                "title": row["product_name"],
                "source_evidence": category + "; occasion=" + row["occasion"],
                "url": row["Image URL-src"],
            }
        )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(probe, cases))
    summary = {
        "sample_scope": (
            "All complete rows in fixed first 8MB of three Amazon category files; "
            "no population estimate or teacher label assignment."
        ),
        "rule_counts": {name: len(rows) for name, rows in pools.items()},
        "unique_parent_asins_per_rule": {
            name: len({r["id"] for r in rows}) for name, rows in pools.items()
        },
        "image_probe_rows": len(results),
        "image_probe_usable": sum(r["usable"] for r in results),
        "wardrobe_image_domains": dict(
            Counter(urlparse(r["Image URL-src"]).netloc for r in wardrobe)
        ),
        "wardrobe_source_page_domains": dict(
            Counter(urlparse(r["web-scraper-start-url"]).netloc for r in wardrobe)
        ),
        "overlap_check_performed": False,
        "human_labels_assigned": False,
    }
    (CAT / "qualified_metadata_candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (CAT / "image_access.json").write_text(json.dumps(results, indent=2) + "\n")
    (CAT / "candidate_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
