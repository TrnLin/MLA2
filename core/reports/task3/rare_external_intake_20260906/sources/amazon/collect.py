"""Bounded, cached Amazon catalogue intake using original rare-class evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/external/rare_usage_20260906/amazon"
OLD = ROOT / "reports/task3/targeted_dataset_search_20260905/catalogue"
BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/"
SOURCES = {
    "amazon_home": ("Home_and_Kitchen", 64_000_000),
    "amazon_clothing": ("Clothing_Shoes_and_Jewelry", 32_000_000),
    "amazon_sports": ("Sports_and_Outdoors", 32_000_000),
}
BAG = re.compile(r"\b(?:backpacks?|rucksacks?|duffels?|duffles?|handbags?|travel bags?)\b", re.I)
TRAVEL = re.compile(r"\btravel(?:ling|ing)?\b", re.I)
EXCLUDED_BAG = re.compile(r"\b(?:covers?|organizers?|inserts?|tags?|straps?|charms?)\b", re.I)
BYTE_LOCK = threading.Lock()
NETWORK_BYTES = 0


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def prefix(source):
    category, cap = SOURCES[source]
    path = DATA / "metadata" / f"{source}.bin"
    evidence_path = path.with_suffix(".json")
    if path.exists() and evidence_path.exists():
        data = path.read_bytes()
        meta = json.loads(evidence_path.read_text())
        assert digest(data) == meta["sha256"]
        return source, data, meta
    url = BASE + f"meta_{category}.jsonl"
    with requests.get(
        url, headers={"Range": f"bytes=0-{cap - 1}"}, stream=True, timeout=(15, 45)
    ) as response:
        response.raise_for_status()
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(
            "bytes 0-"
        ):
            raise ValueError("Source did not honour the bounded byte prefix")
        body = bytearray()
        for chunk in response.iter_content(65536):
            body.extend(chunk[: cap - len(body)])
            if len(body) >= cap:
                break
        data = bytes(body)
        old = (OLD / "raw" / (source + ".bin")).read_bytes()
        if data[: len(old)] != old:
            raise ValueError("Previously researched metadata prefix changed")
        meta = {
            "url": url,
            "status": response.status_code,
            "content_range": response.headers.get("Content-Range"),
            "etag": response.headers.get("ETag"),
            "bytes": len(data),
            "sha256": digest(data),
            "old_prefix_matches": True,
            "scope": "Fixed first-byte prefix; not random or full category coverage",
        }
    path.write_bytes(data)
    write_json(evidence_path, meta)
    return source, data, meta


def candidates(source, data):
    matches, counts = [], Counter()
    for line in data[: data.rfind(b"\n")].splitlines():
        row = json.loads(line)
        counts["complete_rows"] += 1
        title, cats = row.get("title", ""), row.get("categories") or []
        details = row.get("details") or {}
        if isinstance(details, str):
            details = json.loads(details)
        occasion = str(details.get("Occasion", "")).strip()
        style = str(details.get("Style", "")).strip()
        native_label = ""
        strength, evidence = "", ""
        if cats and cats[-1] == "Throw Pillow Covers" and occasion.casefold() == "home":
            native_label, strength, evidence = (
                "Home",
                "source_exact_occasion",
                "details.Occasion=" + occasion,
            )
        elif occasion.casefold() == "smart casual" or style.casefold() == "smart casual":
            native_label, strength = "Smart Casual", "source_exact_style"
            evidence = "details=" + json.dumps(details, ensure_ascii=False)
        elif BAG.search(title) and not EXCLUDED_BAG.search(title):
            if occasion.casefold() == "travel":
                native_label, strength, evidence = (
                    "Travel",
                    "source_exact_occasion",
                    "details.Occasion=" + occasion,
                )
            elif TRAVEL.search(title):
                native_label, strength, evidence = "Travel", "weak_title", "title=" + title
        if not native_label:
            continue
        images = row.get("images") or []
        main = next((r for r in images if r.get("variant") == "MAIN"), images[0] if images else {})
        url = main.get("large") or main.get("hi_res") or main.get("thumb") or ""
        if not url:
            counts["matched_without_image_url"] += 1
            continue
        source_id = row["parent_asin"]
        record = {
            "source": "amazon_reviews_2023",
            "source_category": source,
            "source_id": source_id,
            "source_group_id": "amazon_asin:" + source_id,
            "source_title": title,
            "source_article_type": cats[-1] if cats else "",
            "source_label": native_label,
            "label_evidence": evidence,
            "label_strength": strength,
            "source_url": "https://www.amazon.com/dp/" + source_id,
            "image_url": url,
            "source_brand": str(details.get("Brand") or row.get("store") or ""),
            "rights_basis": (
                "Public publisher research release; retailer photo rights retained; "
                "local academic experiment, no blanket photo redistribution grant established"
            ),
            "native_metadata": row,
        }
        matches.append(record)
        counts[native_label + ":" + strength] += 1
    return matches, dict(counts)


def fetch_image(row):
    global NETWORK_BYTES
    path = DATA / "originals" / (row["source_id"] + ".jpg")
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous.get("usable"):
            assert digest(path.read_bytes()) == previous["original_sha256"]
        return previous
    record = {k: v for k, v in row.items() if k != "native_metadata"}
    record.update(usable=False, network_bytes=0)
    try:
        old_path = OLD / "images" / (row["source_id"] + ".jpg")
        if old_path.exists():
            old_meta = json.loads(old_path.with_suffix(".json").read_text())
            body = old_path.read_bytes()
            assert old_meta["url"] == row["image_url"] and digest(body) == old_meta["sha256"]
            record["reused_cache"] = True
        else:
            with requests.get(row["image_url"], stream=True, timeout=(15, 25)) as response:
                record["http_status"] = response.status_code
                response.raise_for_status()
                chunks = bytearray()
                for chunk in response.iter_content(32768):
                    with BYTE_LOCK:
                        NETWORK_BYTES += len(chunk)
                        record["network_bytes"] += len(chunk)
                        if NETWORK_BYTES > 100_000_000:
                            raise ValueError("Total image-byte cap reached")
                    if len(chunks) + len(chunk) > 2_097_152:
                        raise ValueError("Image exceeds 2 MiB cap")
                    chunks.extend(chunk)
                body = bytes(chunks)
        with Image.open(io.BytesIO(body)) as im:
            im.load()
            record.update(width=im.width, height=im.height, image_format=im.format)
        path.write_bytes(body)
        record.update(
            usable=True, original_path=str(path.relative_to(ROOT)), original_sha256=digest(body)
        )
    except (requests.RequestException, ValueError, OSError) as exc:
        record["error"] = str(exc)
    write_json(meta_path, record)
    return record


def main():
    for p in (OUT, DATA / "metadata", DATA / "originals"):
        p.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        inputs = list(pool.map(prefix, SOURCES))
    all_rows, audits, seen = [], {}, set()
    for source, data, metadata in inputs:
        rows, counts = candidates(source, data)
        audits[source] = {"input": metadata, "counts": counts}
        for row in rows:
            key = (row["source_id"], row["source_label"], row["label_strength"])
            if key not in seen:
                all_rows.append(row)
                seen.add(key)
    write_json(OUT / "metadata_candidates.json", all_rows)
    selected = []
    for label, limit in (("Home", 120), ("Smart Casual", 80), ("Travel", 40)):
        pool = [r for r in all_rows if r["source_label"] == label]
        pool.sort(
            key=lambda r: (
                r["label_strength"] == "weak_title",
                digest(("2753:" + r["source_id"]).encode()),
            )
        )
        selected.extend(pool[:limit])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch_image, selected))
    write_json(OUT / "access_results.json", results)
    good = [r for r in results if r["usable"]]
    fields = (
        [k for k in good[0] if k not in {"usable", "http_status", "network_bytes", "reused_cache"}]
        if good
        else ["source", "source_id"]
    )
    with (OUT / "candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(good)
    summary = {
        "metadata": audits,
        "selected": len(selected),
        "decoded": len(good),
        "decoded_by_label_strength": dict(
            Counter(r["source_label"] + ":" + r["label_strength"] for r in good)
        ),
        "new_image_network_bytes": NETWORK_BYTES,
        "status": "Original source candidates only; root label/duplicate/visual gate pending",
        "excluded_classes": ["Ethnic", "Sports", "Formal", "Casual", "NA"],
    }
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
