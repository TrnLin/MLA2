"""Audit bounded public product metadata; never assign teacher Usage labels."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "catalogue"
RAW = OUT / "raw"
AMAZON = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/"
ABO = "https://amazon-berkeley-objects.s3.amazonaws.com/"
SOURCES = {
    "amazon_clothing": (AMAZON + "meta_Clothing_Shoes_and_Jewelry.jsonl", 8_000_000, True),
    "amazon_home": (AMAZON + "meta_Home_and_Kitchen.jsonl", 8_000_000, True),
    "amazon_sports": (AMAZON + "meta_Sports_and_Outdoors.jsonl", 8_000_000, True),
    "abo_shard_0": (ABO + "listings/metadata/listings_0.json.gz", 6_000_000, False),
    "abo_readme": (ABO + "README.md", 100_000, False),
    "abo_license": (ABO + "LICENSE-CC-BY-4.0.txt", 100_000, False),
    "amazon_card": (
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/README.md",
        100_000,
        False,
    ),
}
TERMS = {
    "smart_casual": re.compile(r"\bsmart[ -]casual\b", re.I),
    "business_casual": re.compile(r"\bbusiness[ -]casual\b", re.I),
    "travel": re.compile(r"\btravel(?:ling|ing)?\b", re.I),
    "pillow_covers": re.compile(r"\b(?:cushion|pillow)\s+covers?\b", re.I),
    "sport": re.compile(r"\b(?:sport|sports|athletic|running|workout|yoga)\b", re.I),
    "ethnic": re.compile(r"\b(?:ethnic|kurta|kurti|saree|sari)\b", re.I),
}


def fetch(name):
    url, cap, partial = SOURCES[name]
    path = RAW / (name + ".bin")
    meta_path = RAW / (name + ".fetch.json")
    if path.exists() and meta_path.exists():
        data = path.read_bytes()
        meta = json.loads(meta_path.read_text())
        assert hashlib.sha256(data).hexdigest() == meta["sha256"]
        return name, data, meta
    headers = {"Range": f"bytes=0-{cap - 1}"} if partial else {}
    with requests.get(url, headers=headers, timeout=(15, 40), stream=True) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            take = chunk[: cap - size]
            chunks.append(take)
            size += len(take)
            if size == cap:
                break
        data = b"".join(chunks)
        if not partial and len(data) == cap:
            raise RuntimeError(f"Complete-file cap reached: {name}")
        meta = {
            "requested_url": url,
            "observed_on": "2026-09-05",
            "status": response.status_code,
            "content_range": response.headers.get("Content-Range"),
            "etag": response.headers.get("ETag"),
            "bytes_saved": len(data),
            "byte_cap": cap,
            "selection": "first byte prefix; not a representative sample"
            if partial
            else "complete named file",
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    path.write_bytes(data)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return name, data, meta


def english(value):
    if not isinstance(value, list):
        return str(value or "")
    return " | ".join(
        str(v.get("value", ""))
        for v in value
        if isinstance(v, dict) and v.get("language_tag", "en_US").startswith("en")
    )


def run():
    RAW.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(fetch, SOURCES))
    summary, hits, details_rows = {}, [], []
    for name, data, provenance in fetched:
        if name.startswith("abo_") and name != "abo_shard_0" or name == "amazon_card":
            continue
        if name == "abo_shard_0":
            rows = [json.loads(line) for line in gzip.decompress(data).splitlines() if line]
        else:
            complete = data[: data.rfind(b"\n")]
            rows = [json.loads(line) for line in complete.splitlines() if line]
        keys, detail_keys, relevant_details, term_counts = (
            Counter(),
            Counter(),
            Counter(),
            Counter(),
        )
        unique_ids, image_rows, examples = set(), 0, []
        for row in rows:
            keys.update(row.keys())
            if name == "abo_shard_0":
                item_id, title = row["item_id"], english(row.get("item_name"))
                details = {
                    k: english(row.get(k))
                    for k in ("style", "product_type", "item_keywords")
                    if row.get(k)
                }
                description = (
                    english(row.get("bullet_point"))
                    + " | "
                    + english(row.get("product_description"))
                )
                categories = [v.get("node_name", v.get("path", "")) for v in row.get("node", [])]
                image_ref = row.get("main_image_id", "")
            else:
                item_id, title = row["parent_asin"], row["title"]
                details = row.get("details") or {}
                if isinstance(details, str):
                    details = json.loads(details)
                description = " | ".join(row.get("features", []) + row.get("description", []))
                categories = row.get("categories", [])
                imgs = row.get("images", [])
                main = next(
                    (im for im in imgs if im.get("variant") == "MAIN"), imgs[0] if imgs else {}
                )
                image_ref = main.get("large") or main.get("hi_res") or main.get("thumb") or ""
            unique_ids.add(item_id)
            image_rows += bool(image_ref)
            detail_keys.update(details.keys())
            for key, value in details.items():
                if re.search(r"occasion|style|sport|use|purpose|activity", key, re.I):
                    relevant_details[(key, str(value))] += 1
            fields = {
                "title": title,
                "categories": json.dumps(categories),
                "details": json.dumps(details),
                "description": description,
            }
            matched = set()
            for term, pattern in TERMS.items():
                for field, value in fields.items():
                    match = pattern.search(value)
                    if match:
                        matched.add(term)
                        hits.append(
                            {
                                "source": name,
                                "id": item_id,
                                "term": term,
                                "field": field,
                                "title": title,
                                "context": value[max(0, match.start() - 90) : match.end() + 150],
                                "categories": json.dumps(categories),
                                "details": json.dumps(details),
                                "image_reference": image_ref,
                            }
                        )
            term_counts.update(matched)
            if len(examples) < 3:
                examples.append(
                    {
                        "id": item_id,
                        "title": title,
                        "details": details,
                        "image_reference": image_ref,
                    }
                )
        for (key, value), count in relevant_details.most_common():
            details_rows.append({"source": name, "key": key, "value": value, "rows": count})
        summary[name] = {
            "provenance": provenance,
            "parsed_rows": len(rows),
            "unique_product_ids": len(unique_ids),
            "rows_with_image_reference": image_rows,
            "top_level_keys": dict(keys),
            "detail_keys": dict(detail_keys.most_common()),
            "term_hit_rows": dict(term_counts),
            "examples": examples,
            "interpretation": (
                "Metadata retrieval hits only. No assigned Usage labels, image access checks, "
                "duplicate clearance, or whole-dataset count estimate."
            ),
        }
    for filename, rows in (("term_hits.csv", hits), ("detail_values.csv", details_rows)):
        with (OUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "metadata_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: {
                    k: v
                    for k, v in value.items()
                    if k in ("parsed_rows", "unique_product_ids", "term_hit_rows")
                }
                for key, value in summary.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
