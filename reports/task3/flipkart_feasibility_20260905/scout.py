"""Read Flipkart's released metadata without changing any training data."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/external/flipkart_products_v1"
VIEW_URL = "https://www.kaggle.com/api/v1/datasets/view/PromptCloudHQ/flipkart-products"
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/PromptCloudHQ/flipkart-products"
    "?datasetVersionNumber=1"
)
EXPECTED_ARCHIVE_SHA = "54a91fcd0b3d1923e3adb52c27e4dde557a7cd948dba066e3cb5bca542da1b9f"


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    if not rows and fields is None:
        raise ValueError("Empty CSV requires fields")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def download(url: str, cap: int):
    request = urllib.request.Request(url, headers={"User-Agent": "FashionDataFeasibility/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(cap + 1)
    if len(data) > cap:
        raise ValueError(f"Download exceeds {cap} byte cap")
    return data


def acquire():
    RAW.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT / "publisher_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_bytes(download(VIEW_URL, 3_000_000))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["id"] == 2506 and metadata["currentVersionNumber"] == 1
    archive_path = RAW / "flipkart-products-v1.zip"
    if not archive_path.exists():
        archive_path.write_bytes(download(DOWNLOAD_URL, 15_000_000))
    archive = archive_path.read_bytes()
    archive_sha = hashlib.sha256(archive).hexdigest()
    assert archive_sha == EXPECTED_ARCHIVE_SHA, (
        "Unexpected release bytes; inspect before proceeding"
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = bundle.namelist()
        assert names == ["flipkart_com-ecommerce_sample.csv"]
        assert bundle.getinfo(names[0]).file_size < 100_000_000
        data = bundle.read(names[0])
    csv_path = RAW / names[0]
    if csv_path.exists():
        assert csv_path.read_bytes() == data
    else:
        csv_path.write_bytes(data)
    provenance_path = OUT / "source_provenance.json"
    previous = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    save_json(
        OUT / "source_provenance.json",
        {
            "source": "https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products",
            "metadata_api": VIEW_URL,
            "download_url": DOWNLOAD_URL,
            "dataset_id": metadata["id"],
            "version": metadata["currentVersionNumber"],
            "publisher": metadata["ownerName"],
            "declared_dataset_license": metadata["licenseName"],
            "license_limit": (
                "Publisher's dataset declaration; separate image rights "
                "not established by this field alone."
            ),
            "retrieved_utc": previous.get("retrieved_utc", datetime.now(timezone.utc).isoformat()),
            "archive_bytes": len(archive),
            "archive_sha256": archive_sha,
            "csv_bytes": len(data),
            "csv_sha256": hashlib.sha256(data).hexdigest(),
            "local_csv": str(csv_path.relative_to(ROOT)),
            "purpose": "Metadata and bounded image-access feasibility only; no training admission",
        },
    )
    return csv_path


def literal(value, default):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default


def normalize(value: str):
    return re.sub(r"\s+", " ", value).strip()


def parse_rows(csv_path: Path):
    raw_rows = list(csv.DictReader(csv_path.open()))
    rows = []
    for index, raw in enumerate(raw_rows):
        paths = literal(raw["product_category_tree"], [])
        category = paths[0] if paths else raw["product_category_tree"]
        segments = [normalize(value) for value in category.split(" >> ")]
        urls = literal(raw["image"], [])
        urls = [
            value
            for value in urls
            if isinstance(value, str) and value.startswith(("http://", "https://"))
        ]
        # The released field is a Ruby-style hash representation. literal_eval never executes it.
        specs = literal(raw["product_specifications"].replace("=>", ":"), {})
        entries = specs.get("product_specification", []) if isinstance(specs, dict) else []
        occasion_entries = [
            normalize(str(entry.get("value", "")))
            for entry in entries
            if isinstance(entry, dict) and str(entry.get("key", "")).strip().lower() == "occasion"
        ]
        occasions = sorted({value for value in occasion_entries if value})
        tokens = sorted(
            {
                normalize(value)
                for entry in occasions
                for value in entry.split(",")
                if normalize(value)
            }
        )
        rows.append(
            {
                "row_index": index,
                "source_id": raw["uniq_id"],
                "product_id": raw["pid"],
                "product_url": raw["product_url"],
                "name": raw["product_name"],
                "brand": raw["brand"],
                "root_category": segments[0],
                "category_path": category,
                "segments": segments,
                "image_urls": urls,
                "occasion_entries": occasion_entries,
                "occasions": occasions,
                "occasion_tokens": tokens,
                "specs": entries,
            }
        )
    return rows


def main():
    rows = parse_rows(acquire())
    assert len(rows) == 20_000
    save_json(OUT / "parsed_metadata.json", rows)
    roots = Counter(row["root_category"] for row in rows)
    segments = Counter(segment for row in rows for segment in set(row["segments"]))
    paths = Counter(row["category_path"] for row in rows)
    occasions = Counter(entry for row in rows for entry in row["occasion_entries"])
    tokens = Counter(token for row in rows for token in row["occasion_tokens"])
    for name, counts, field in [
        ("root_counts", roots, "root"),
        ("segment_counts", segments, "segment"),
        ("category_path_counts", paths, "path"),
        ("occasion_value_counts", occasions, "occasion"),
        ("occasion_token_counts", tokens, "token"),
    ]:
        write_csv(
            OUT / f"{name}.csv",
            [{field: key, "rows": value} for key, value in counts.most_common()],
        )
    summary = {
        "rows": len(rows),
        "unique_source_ids": len({row["source_id"] for row in rows}),
        "unique_product_ids": len({row["product_id"] for row in rows}),
        "rows_with_image_urls": sum(bool(row["image_urls"]) for row in rows),
        "rows_with_occasion": sum(bool(row["occasions"]) for row in rows),
        "occasion_entries": sum(len(row["occasion_entries"]) for row in rows),
        "single_occasion_token_rows": sum(len(row["occasion_tokens"]) == 1 for row in rows),
        "multiple_occasion_token_rows": sum(len(row["occasion_tokens"]) > 1 for row in rows),
        "teacher_article_types": len(
            json.loads((ROOT / "data/processed/taxonomy.json").read_text())["targets"][
                "articleType"
            ]["classes"]
        ),
        "canonical_split_sha256": hashlib.sha256(
            (ROOT / "data/processed/splits.csv").read_bytes()
        ).hexdigest(),
        "training_rows_admitted": 0,
    }
    save_json(OUT / "metadata_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
