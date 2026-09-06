"""Bounded Party original-image intake. Default verifies cache, no network."""

from fashion.task3_paths import resolve_task3_path

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
OLD = ROOT / "reports/task3/flipkart_feasibility_20260905"
DEST = ROOT / "data/external/rare_usage_20260906/flipkart_party/originals"
LIMIT = 2 * 1024 * 1024
BUDGET = 150_000_000


def read(path):
    return list(csv.DictReader(path.open()))


def write(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    if (OUT / "candidates.csv").exists() and not args.acquire:
        rows = read(OUT / "candidates.csv")
        for r in rows:
            p = resolve_task3_path(r["original_path"], root=ROOT)
            assert hashlib.sha256(p.read_bytes()).hexdigest() == r["original_sha256"]
            with Image.open(p) as im:
                im.load()
                assert im.size == (int(r["width"]), int(r["height"]))
        print(json.dumps({"verified": len(rows), "new_requests": 0}))
        return
    if not args.acquire:
        raise SystemExit("No intake cache; pass --acquire for bounded retrieval.")
    DEST.mkdir(parents=True, exist_ok=True)
    metadata = {r["product_id"]: r for r in json.loads((OLD / "parsed_metadata.json").read_text())}
    selected = [
        r for r in read(OLD / "deduplicated_candidates.csv") if r["source_occasion"] == "Party"
    ]
    assert len(selected) <= 186
    cache = {}
    for name in ["image_access.csv", "candidate_image_access.csv"]:
        for r in read(OLD / name):
            if r["usable"].lower() == "true":
                cache[r["product_id"]] = r
    # Stable source order, with previously verified images first.
    selected.sort(key=lambda r: (r["product_id"] not in cache, r["metadata_group"]))
    results, failures, attempts = [], [], []
    spent = reused = 0
    for r in selected:
        if len(results) >= 150:
            failures.append(
                {"source_id": r["product_id"], "reason": "not_attempted_150_success_target"}
            )
            continue
        m = metadata[r["product_id"]]
        assert m["occasion_tokens"] == ["Party"], m["occasion_tokens"]
        assert r["source_image_url"] == m["image_urls"][0]
        payload = None
        c = cache.get(r["product_id"])
        if c and c["source_url"] == r["source_image_url"]:
            p = resolve_task3_path(c["local_path"], root=ROOT)
            b = p.read_bytes()
            if hashlib.sha256(b).hexdigest() == c["sha256"]:
                payload = b
                reused += 1
        if payload is None:
            urls = [r["source_image_url"]]
            if urls[0].startswith("http://"):
                urls.append("https://" + urls[0][7:])
            for url in urls:
                a = {
                    "source_id": r["product_id"],
                    "request_url": url,
                    "http_status": "",
                    "final_url": "",
                    "bytes_read": 0,
                    "error": "",
                }
                if spent + LIMIT > BUDGET:
                    a["error"] = "total_body_budget"
                    attempts.append(a)
                    break
                try:
                    with requests.get(url, stream=True, timeout=(10, 15)) as response:
                        a.update(http_status=response.status_code, final_url=response.url)
                        response.raise_for_status()
                        b = response.raw.read(LIMIT, decode_content=True)
                        a["bytes_read"] = len(b)
                        spent += len(b)
                        if len(b) >= LIMIT:
                            raise ValueError("2MiB_cap")
                        with Image.open(io.BytesIO(b)) as im:
                            im.load()
                        payload = b
                except Exception as exc:
                    a["error"] = str(exc)
                attempts.append(a)
                if payload is not None:
                    break
        if payload is None:
            failures.append({"source_id": r["product_id"], "reason": "image_access_failed"})
            continue
        with Image.open(io.BytesIO(payload)) as im:
            im.load()
            width, height = im.size
        sha = hashlib.sha256(payload).hexdigest()
        path = DEST / (r["product_id"] + ".image")
        path.write_bytes(payload)
        results.append(
            dict(
                source="flipkart_party",
                source_id=r["product_id"],
                source_group_id=r["metadata_group"],
                source_title=r["name"],
                source_article_type=r["teacher_article_type_candidate"],
                source_label="Party",
                label_evidence=json.dumps(m["occasion_entries"]),
                label_strength="source_exact_occasion",
                source_url=r["source_product_url"],
                image_url=r["source_image_url"],
                original_path=str(path.relative_to(ROOT)),
                original_sha256=sha,
                width=width,
                height=height,
                rights_basis=(
                    "Publisher metadata CC BY-SA 4.0; remote photo rights scope not established"
                ),
                source_record_id=r["source_id"],
                source_category_path=m["category_path"],
                source_specs=json.dumps(m["specs"]),
                attribution="Flipkart Products, PromptCloud / PromptCloudHQ, version 1, via Kaggle",
                license_url="https://creativecommons.org/licenses/by-sa/4.0/",
                dataset_url="https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products",
            )
        )
        print(f"{len(results)} decoded; {spent} new body bytes", flush=True)
    write("candidates.csv", results)
    write("exclusions.csv", failures)
    write("attempts.csv", attempts)
    for start in range(0, len(results), 30):
        sheet = Image.new("RGB", (1000, 1080), "white")
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(results[start : start + 30]):
            with Image.open(resolve_task3_path(r["original_path"], root=ROOT)) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (190, 150))
            x, y = (i % 5) * 200, (i // 5) * 180
            sheet.paste(thumb, (x + (190 - thumb.width) // 2, y))
            draw.text(
                (x + 3, y + 151), f"{start + i + 1}: {r['source_article_type']}", fill="black"
            )
            draw.text((x + 3, y + 164), r["source_id"], fill="black")
        sheet.save(OUT / f"contact_{start // 30 + 1:02d}.jpg")
    summary = {
        "source_candidates": len(selected),
        "decoded": len(results),
        "reused": reused,
        "new_body_bytes": spent,
        "attempts": len(attempts),
        "exclusions": len(failures),
        "workers": 1,
        "label_strength": (
            "source_exact_occasion; seller labels remain weak relative to teacher annotations"
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
