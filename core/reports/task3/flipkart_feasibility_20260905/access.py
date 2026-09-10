"""Bounded source-URL image access and image-only cached duplicate audit."""

from fashion.task3_paths import resolve_task3_path
# ruff: noqa: E501

import argparse
import csv
import hashlib
import io
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
DEST = ROOT / "data/raw/external/flipkart_products_v1/access_sample"
sys.path.insert(0, str(ROOT / "src"))
from fashion.data.perceptual import compute_image_hashes, compute_pair_pixel_metrics  # noqa: E402

LIMIT = 2 * 1024 * 1024
TOTAL_LIMIT = 100_000_000
BYTE_LOCK = threading.Lock()
BYTES_RECEIVED = 0
ROOTS = [
    "Clothing",
    "Footwear",
    "Jewellery",
    "Watches",
    "Bags, Wallets & Belts",
    "Sunglasses",
    "Beauty and Personal Care",
    "Sports & Fitness",
]


def write_csv(name, rows, fields):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select(rows):
    eligible = sorted(
        (r for r in rows if r["root_category"] in ROOTS and r["image_urls"]),
        key=lambda r: hashlib.sha256(("2753:" + r["product_id"]).encode()).hexdigest(),
    )
    chosen, seen = [], set()

    def add(row, stratum):
        if row["product_id"] not in seen:
            chosen.append({**row, "sampling_stratum": stratum})
            seen.add(row["product_id"])

    # Three products per broad root, then 24 rare exact occasion values.
    for root in ROOTS:
        for row in [r for r in eligible if r["root_category"] == root][:3]:
            add(row, "root:" + root)
    counts = Counter(v for r in eligible for v in set(r["occasions"]))
    for value in sorted(counts, key=lambda v: (counts[v], v)):
        row = next(
            (r for r in eligible if value in r["occasions"] and r["product_id"] not in seen), None
        )
        if row:
            add(row, "rare_exact_occasion:" + value)
        if len(chosen) == 48:
            break
    assert len(chosen) == len(seen) == 48
    return chosen


def fetch(row):
    global BYTES_RECEIVED
    original = row["image_urls"][0]
    attempts = []
    urls = [original]
    if original.startswith("http://"):
        urls.append("https://" + original[len("http://") :])
    result = dict(
        product_id=row["product_id"],
        source_url=original,
        sampling_stratum=row["sampling_stratum"],
        usable=False,
        local_path="",
        sha256="",
        width="",
        height="",
        dhash_hex="",
        ahash_hex="",
    )
    for url in urls:
        attempt = dict(
            request_url=url,
            scheme_only_upgrade=url != original,
            http_status=None,
            final_url="",
            content_type="",
            error="",
            bytes_read=0,
        )
        try:
            with requests.get(url, stream=True, timeout=(15, 15), allow_redirects=True) as response:
                attempt.update(
                    http_status=response.status_code,
                    final_url=response.url,
                    content_type=response.headers.get("Content-Type", ""),
                )
                response.raise_for_status()
                payload = bytearray()
                for chunk in response.iter_content(16384):
                    with BYTE_LOCK:
                        BYTES_RECEIVED += len(chunk)
                        if BYTES_RECEIVED > TOTAL_LIMIT:
                            raise ValueError("100 MB total response cap exceeded")
                    if len(payload) + len(chunk) > LIMIT:
                        raise ValueError("2 MiB response cap exceeded")
                    payload.extend(chunk)
                attempt["bytes_read"] = len(payload)
                with Image.open(io.BytesIO(payload)) as im:
                    im.load()
                    width, height = im.size
                path = DEST / (
                    hashlib.sha256(row["product_id"].encode()).hexdigest()[:20] + ".image"
                )
                path.write_bytes(payload)
                dhash, ahash = compute_image_hashes(path)
                result.update(
                    usable=True,
                    local_path=str(path.relative_to(ROOT)),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    width=width,
                    height=height,
                    dhash_hex=f"{dhash:016x}",
                    ahash_hex=f"{ahash:016x}",
                )
        except Exception as exc:
            attempt["error"] = str(exc)
        attempts.append(attempt)
        if result["usable"]:
            break
    result["attempts_json"] = json.dumps(attempts)
    result.update(
        {k: attempts[-1][k] for k in ["http_status", "final_url", "content_type", "error"]}
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Explicitly allow network probing.")
    args = parser.parse_args()
    existing = OUT / "image_access.csv"
    if existing.exists() and not args.refresh:
        with existing.open() as stream:
            saved = list(csv.DictReader(stream))
        count = 0
        for row in saved:
            if row["usable"].lower() != "true":
                continue
            path = resolve_task3_path(row["local_path"], root=ROOT)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
            with Image.open(path) as image:
                image.load()
            count += 1
        print(
            json.dumps(
                dict(
                    mode="cache_readback_only",
                    network_requests_this_run=0,
                    verified_usable_images=count,
                    saved_summary=json.loads((OUT / "access_summary.json").read_text()),
                ),
                indent=2,
            )
        )
        return
    if not args.refresh:
        raise SystemExit("No saved cache. Use --refresh explicitly for the bounded network probe.")
    rows = json.loads((OUT / "parsed_metadata.json").read_text())
    chosen = select(rows)
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            k: json.dumps(r[k]) if k in ["occasions", "image_urls"] else r[k]
            for k in [
                "product_id",
                "source_id",
                "row_index",
                "root_category",
                "name",
                "occasions",
                "image_urls",
                "sampling_stratum",
            ]
        }
        for r in chosen
    ]
    write_csv("sample_manifest.csv", manifest, list(manifest[0]))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(fetch, chosen))
    write_csv("image_access.csv", results, list(results[0]))
    good = [r for r in results if r["usable"]]
    matches = []
    coverage = {}
    if good:
        # Explicit allowlists prevent reading protected label fields.
        cached = pd.read_csv(
            ROOT / "data/processed/audit/perceptual_hashes.csv.gz",
            usecols=["id", "path", "sha256", "role", "dhash_hex", "ahash_hex"],
            dtype=str,
        )
        for role, filename in [
            ("labelled", "splits.csv"),
            ("prediction", "prediction_manifest.csv"),
        ]:
            inventory = pd.read_csv(
                ROOT / "data/processed" / filename, usecols=["id", "path", "sha256"], dtype=str
            )
            subset = cached[cached.role == role]
            assert set(map(tuple, inventory[["id", "path", "sha256"]].values)) == set(
                map(tuple, subset[["id", "path", "sha256"]].values)
            ), "Cache does not exactly cover current image inventory"
        splits = pd.read_csv(
            ROOT / "data/processed/splits.csv", usecols=["id", "partition"], dtype=str
        )
        roles = dict(zip(splits.id, splits.partition))
        cached["audit_role"] = [
            roles.get(r.id, r.role) if r.role != "prediction" else "prediction"
            for r in cached.itertuples()
        ]
        coverage = cached.audit_role.value_counts().to_dict()
        for sample in good:
            dh = int(sample["dhash_hex"], 16)
            for teacher in cached.itertuples():
                distance = (dh ^ int(teacher.dhash_hex, 16)).bit_count()
                exact = sample["sha256"] == teacher.sha256
                if exact or distance <= 2:
                    metrics = compute_pair_pixel_metrics(
                        resolve_task3_path(sample["local_path"], root=ROOT), resolve_task3_path(teacher.path, root=ROOT)
                    )
                    matches.append(
                        dict(
                            product_id=sample["product_id"],
                            teacher_id=teacher.id,
                            teacher_role=teacher.audit_role,
                            teacher_path=teacher.path,
                            exact_sha256=exact,
                            dhash_distance=distance,
                            ahash_distance=(
                                int(sample["ahash_hex"], 16) ^ int(teacher.ahash_hex, 16)
                            ).bit_count(),
                            pixel_metrics_json=json.dumps(metrics),
                            status="exact_bytes" if exact else "candidate_requires_review",
                        )
                    )
    write_csv(
        "duplicate_matches.csv",
        matches,
        [
            "product_id",
            "teacher_id",
            "teacher_role",
            "teacher_path",
            "exact_sha256",
            "dhash_distance",
            "ahash_distance",
            "pixel_metrics_json",
            "status",
        ],
    )
    summary = dict(
        sample_products=len(results),
        usable_images=len(good),
        failed_products=len(results) - len(good),
        attempts=sum(len(json.loads(r["attempts_json"])) for r in results),
        final_status_counts=dict(Counter(str(r["http_status"]) for r in results)),
        downloaded_usable_bytes=sum((resolve_task3_path(r["local_path"], root=ROOT)).stat().st_size for r in good),
        teacher_cached_role_counts=coverage,
        duplicate_candidates=len(matches),
        duplicate_status="cached image-only comparison"
        if good
        else "blocked: no usable sample images",
        limits={
            "products": 48,
            "bytes_per_file": LIMIT,
            "workers": 4,
            "connect_and_read_timeout_seconds": 15,
        },
        note="One source-listed URL per product; failed HTTP permits scheme-only HTTPS retry. No inference of whole-dataset availability or absence of leakage from this sample. No training admission.",
    )
    (OUT / "access_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "ACCESS.md").write_text(
        "# Bounded Flipkart image access audit\n\n"
        + f"Tested {len(results)} distinct products: three per each of eight broad roots, then rare exact occasion values to reach 48. Hash ordering uses salt 2753. This is a purposive diagnostic sample, not a population estimate.\n\nUsable decoded images: {len(good)}. Failed products: {len(results) - len(good)}. Requests: {summary['attempts']}.\n\n"
        + "Each product uses its first source-listed image URL. A failed HTTP URL gets one HTTPS scheme-only retry. URLs, responses, errors, decoded dimensions and SHA-256 are in image_access.csv. No alternate CDN paths, authentication bypass, or unofficial copies were attempted. Four workers; 15-second connection/read timeout; 2 MiB per response; 48 products bound saved image data to 96 MiB.\n\n"
        + f"Duplicate audit: {summary['duplicate_status']}. Candidate count: {len(matches)}. Cached teacher role counts: {coverage}. Existing repository dHash algorithm is reused; dHash distance <=2 is only a candidate flag, with aHash and full-canvas/foreground pixel metrics recorded. It does not prove a duplicate or prove the absence of leakage. No protected labels are read. No external images enter training.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
