"""One bounded candidate probe; later default executions only verify saved results."""

from fashion.task3_paths import resolve_task3_path
# ruff: noqa: E501

import argparse
import csv
import hashlib
import io
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from access import (
    DEST,
    LIMIT,
    OUT,
    ROOT,
    compute_image_hashes,
    compute_pair_pixel_metrics,
    write_csv,
)
from PIL import Image

LOCK = threading.Lock()
BUDGET = 40_000_000
spent = 0
reserved = 0
requests_sent = 0


def verified(row):
    if str(row["usable"]).lower() != "true":
        return False
    path = resolve_task3_path(row["local_path"], root=ROOT)
    assert path.is_file(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
    with Image.open(path) as image:
        image.load()
    return True


def fetch(row):
    global spent, reserved, requests_sent
    original = row["source_image_url"]
    urls = [original] + (["https://" + original[7:]] if original.startswith("http://") else [])
    result = dict(
        product_id=row["product_id"],
        source_occasion=row["source_occasion"],
        teacher_article_type_candidate=row["teacher_article_type_candidate"],
        source_url=original,
        usable=False,
        reused=False,
        local_path="",
        sha256="",
        width="",
        height="",
        dhash_hex="",
        ahash_hex="",
    )
    attempts = []
    for url in urls:
        attempt = dict(
            request_url=url,
            scheme_only_upgrade=url != original,
            http_status=None,
            final_url="",
            content_type="",
            bytes_read=0,
            error="",
        )
        with LOCK:
            if spent + reserved + LIMIT > BUDGET:
                attempt["error"] = "40 MB body budget prevents request"
                attempts.append(attempt)
                break
            reserved += LIMIT
            requests_sent += 1
        payload = bytearray()
        try:
            with requests.get(url, stream=True, timeout=(15, 15)) as response:
                attempt.update(
                    http_status=response.status_code,
                    final_url=response.url,
                    content_type=response.headers.get("Content-Type", ""),
                )
                # Error bodies are not consumed; their HTTP outcome is retained.
                response.raise_for_status()
                while attempt["bytes_read"] < LIMIT:
                    chunk = response.raw.read(
                        min(16384, LIMIT - attempt["bytes_read"]), decode_content=True
                    )
                    if not chunk:
                        break
                    attempt["bytes_read"] += len(chunk)
                    payload.extend(chunk)
                if attempt["bytes_read"] == LIMIT:
                    raise ValueError("2 MiB file cap reached; conservatively rejected")
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    width, height = image.size
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
        finally:
            with LOCK:
                reserved -= LIMIT
                spent += attempt["bytes_read"]
        attempts.append(attempt)
        if result["usable"]:
            break
    result.update(
        {k: attempts[-1][k] for k in ["http_status", "final_url", "content_type", "error"]}
    )
    result["attempts_json"] = json.dumps(attempts)
    return result


def duplicates(results):
    cached = pd.read_csv(
        ROOT / "data/processed/audit/perceptual_hashes.csv.gz",
        usecols=["id", "path", "sha256", "role", "dhash_hex", "ahash_hex"],
        dtype=str,
    )
    for role, filename in [("labelled", "splits.csv"), ("prediction", "prediction_manifest.csv")]:
        inventory = pd.read_csv(
            ROOT / "data/processed" / filename, usecols=["id", "path", "sha256"], dtype=str
        )
        subset = cached[cached.role == role]
        assert set(map(tuple, inventory[["id", "path", "sha256"]].values)) == set(
            map(tuple, subset[["id", "path", "sha256"]].values)
        )
    splits = pd.read_csv(ROOT / "data/processed/splits.csv", usecols=["id", "partition"], dtype=str)
    roles = dict(zip(splits.id, splits.partition))
    cached["audit_role"] = [
        roles.get(r.id, r.role) if r.role != "prediction" else "prediction"
        for r in cached.itertuples()
    ]
    matches = []
    for sample in results:
        if not sample["usable"]:
            continue
        for teacher in cached.itertuples():
            distance = (int(sample["dhash_hex"], 16) ^ int(teacher.dhash_hex, 16)).bit_count()
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
        "candidate_duplicate_matches.csv",
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
    return matches, cached.audit_role.value_counts().to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Explicitly permit the single bounded network probe; valid saved product images are still reused.",
    )
    args = parser.parse_args()
    path = OUT / "candidate_image_access.csv"
    if path.exists() and not args.refresh:
        saved = list(csv.DictReader(path.open()))
        count = sum(verified(r) for r in saved)
        print(
            json.dumps(
                dict(
                    mode="cache_readback_only",
                    network_requests_this_run=0,
                    verified_usable_images=count,
                    saved_summary=json.loads((OUT / "candidate_access_summary.json").read_text()),
                ),
                indent=2,
            )
        )
        return
    if not args.refresh:
        raise SystemExit(
            "No candidate cache. Use --refresh explicitly for the bounded network probe."
        )
    rows = list(csv.DictReader((OUT / "candidate_access_manifest.csv").open()))
    assert len(rows) <= 41 and len({r["product_id"] for r in rows}) == len(rows)
    prior = list(csv.DictReader((OUT / "image_access.csv").open()))
    if path.exists():
        prior += list(csv.DictReader(path.open()))
    previous = {r["product_id"]: r for r in prior}

    def process(row):
        old = previous.get(row["product_id"])
        if old and verified(old):
            return dict(
                product_id=row["product_id"],
                source_occasion=row["source_occasion"],
                teacher_article_type_candidate=row["teacher_article_type_candidate"],
                source_url=row["source_image_url"],
                usable=True,
                reused=True,
                **{
                    k: old[k]
                    for k in [
                        "local_path",
                        "sha256",
                        "width",
                        "height",
                        "dhash_hex",
                        "ahash_hex",
                        "http_status",
                        "final_url",
                        "content_type",
                    ]
                },
                error="",
                attempts_json="[]",
            )
        return fetch(row)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process, rows))
    fields = list(results[0])
    write_csv("candidate_image_access.csv", results, fields)
    matches, coverage = duplicates(results)
    labels = {}
    for label in sorted({r["source_occasion"] for r in rows}):
        subset = [r for r in results if r["source_occasion"] == label]
        labels[label] = dict(
            sampled=len(subset),
            usable=sum(r["usable"] for r in subset),
            reused=sum(r["reused"] for r in subset),
        )
    summary = dict(
        sample_products=len(results),
        usable_images=sum(r["usable"] for r in results),
        failed_products=sum(not r["usable"] for r in results),
        reused_images=sum(r["reused"] for r in results),
        network_requests=requests_sent,
        response_body_bytes_read=spent,
        saved_new_image_bytes=sum(
            (resolve_task3_path(r["local_path"], root=ROOT)).stat().st_size
            for r in results
            if r["usable"] and not r["reused"]
        ),
        by_source_occasion=labels,
        final_http_status_counts=dict(Counter(str(r["http_status"]) for r in results)),
        teacher_cached_role_counts=coverage,
        duplicate_candidates=len(matches),
        limits=dict(
            products=41,
            response_body_bytes=40_000_000,
            per_file_bytes=LIMIT,
            workers=4,
            timeout_seconds=15,
        ),
        cache_verified_against_current_manifests=True,
        no_training_admission=True,
        scope="Purposive representatives of exact-label/type metadata candidates; not a population estimate or leakage clearance.",
    )
    (OUT / "candidate_access_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "CANDIDATE_ACCESS.md").write_text(
        "# Exact-label/type candidate image probe\n\n"
        + f"{summary['usable_images']} of {len(results)} representative products yielded usable decoded images; {summary['failed_products']} failed. {summary['reused_images']} verified saved images reused. Additional requests: {requests_sent}; response body bytes consumed: {spent}.\n\n"
        + "\n".join(
            f"- {label}: {v['usable']}/{v['sampled']} usable." for label, v in labels.items()
        )
        + "\n\nSelection comes from candidate_access_manifest.csv: at most eight deterministic metadata-group representatives per available exact source label. These source labels and candidate types remain metadata, not confirmed training labels. Availability supports further feasibility review only for the observed sample. It does not measure the entire pool or clear leakage.\n\n"
        + f"Compared image SHA and dHash against all {sum(coverage.values())} existing cached teacher images, including all split roles and prediction. Cache ID/path/SHA coverage verified against image-only manifest columns. {len(matches)} exact-byte or dHash <=2 candidates; candidate pixel metrics are recorded where applicable. No target columns read. No images admitted to training.\n\n"
        + "The script defaults to saved-file verification with zero network requests. The initial probe requires --refresh. Verified prior images are reused even with refresh. Source first image URL only; failed HTTP permits one scheme-only HTTPS retry; no alternative CDN paths. Four workers, 15-second connect/read timeouts, 2 MiB per file and 40 MB additional body budget. Error response bodies are not consumed. No further sampling is planned.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
