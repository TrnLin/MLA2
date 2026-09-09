"""Recheck source bytes, reported metadata counts, and decoded sample images."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "pyproject.toml").is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = []

    def handle_starttag(self, tag, attrs):
        self.values.extend(value for key, value in attrs if key in {"href", "src"})


def main():
    raw_hashes = {}
    for meta_path in (OUT / "catalogue/raw").glob("*.fetch.json"):
        meta = json.loads(meta_path.read_text())
        path = meta_path.with_name(meta_path.name.replace(".fetch.json", ".bin"))
        assert sha(path) == meta["sha256"]
        assert path.stat().st_size == meta["bytes_saved"]
        raw_hashes[str(path.relative_to(OUT))] = sha(path)

    amazon_rows, count_rows = {}, {}
    for source in ("amazon_clothing", "amazon_home", "amazon_sports"):
        data = (OUT / "catalogue/raw" / (source + ".bin")).read_bytes()
        rows = [json.loads(line) for line in data[: data.rfind(b"\n")].splitlines()]
        amazon_rows[source] = rows
        count_rows[source] = len(rows)
    assert count_rows == {"amazon_clothing": 2797, "amazon_home": 2191, "amazon_sports": 2789}
    home = []
    for row in amazon_rows["amazon_home"]:
        details = row.get("details") or {}
        if isinstance(details, str):
            details = json.loads(details)
        if (row.get("categories") or [""])[-1] == "Throw Pillow Covers":
            if str(details.get("Occasion", "")).lower() == "home":
                home.append(row["parent_asin"])
    assert len(home) == len(set(home)) == 7
    abo = [
        json.loads(line)
        for line in gzip.decompress(
            (OUT / "catalogue/raw/abo_shard_0.bin").read_bytes()
        ).splitlines()
    ]
    smart = {
        row["item_id"]
        for row in abo
        if any(
            v.get("value", "").casefold() == "smart casual"
            and v.get("language_tag", "").startswith("en")
            for v in (row.get("bullet_point") or [])
        )
    }
    assert smart == {"B07255FKTS", "B0727R3L19"}

    image_count = 0
    for record in json.loads((OUT / "catalogue/image_access.json").read_text()):
        assert record["usable"]
        path = OUT / record["saved_path"]
        assert sha(path) == record["sha256"]
        with Image.open(path) as im:
            im.load()
            assert list(im.size) == [record["width"], record["height"]]
        image_count += 1
    abo_access = json.loads((OUT / "travel/abo_access/access_evidence.json").read_text())
    for record in abo_access["images"]:
        path = OUT / "travel/abo_access" / record["local_file"]
        assert sha(path) == record["sha256"]
        with Image.open(path) as im:
            im.load()
            assert list(im.size) == record["decoded_width_height"]
        assert record["dimensions_match_index"]
        image_count += 1
    assert image_count == 11

    with zipfile.ZipFile(OUT / "home_na/wardrobe_assistant.zip") as archive:
        wardrobe = list(
            csv.DictReader(io.StringIO(archive.read("Wardrobe Assistant.csv").decode("utf-8-sig")))
        )
    counts = Counter(row["main_category"] for row in wardrobe)
    assert len(wardrobe) == 5443
    assert counts["Men's Ethnic Wear"] + counts["Women's Ethnic Wear"] == 4570
    smart_rows = list(
        csv.DictReader((OUT / "smart_casual/fashionstylist_smart_casual_matches.csv").open())
    )
    assert len(smart_rows) == 48
    id_rows, short_rows, non_url_rows = [], [], []
    for row in smart_rows:
        url = urlparse(row["URL link"])
        ids = parse_qs(url.query).get("id", [])
        if ids:
            id_rows.append(ids[0])
        elif url.scheme in {"http", "https"}:
            short_rows.append(row["URL link"])
        else:
            non_url_rows.append(row["URL link"])
    assert len(id_rows) == 29 and len(set(id_rows)) == 23
    assert len(short_rows) == 18 and non_url_rows == ["AI生成"]
    ikea = (OUT / "home_na/throw-pillow-covers.txt").read_text()
    assert re.search(r"ProductCount:\s*183\b", ikea)

    split_sha = sha(ROOT / "data/processed/splits.csv")
    assert split_sha == "d76a49c6dc7999b4f286e94838a92c603d68c4f66179fde081028948f6a187db"
    missing, html_count = [], 0
    for path in OUT.rglob("*.html"):
        parser = Links()
        parser.feed(path.read_text())
        html_count += 1
        for value in parser.values:
            url = urlparse(value)
            if not url.scheme and url.path:
                target = (path.parent / unquote(url.path)).resolve()
                if not target.exists() and target != OUT / "verification.json":
                    missing.append({"page": str(path.relative_to(OUT)), "link": value})
    assert not missing, missing
    inputs = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.suffix in {
            ".bin",
            ".csv",
            ".gz",
            ".parquet",
            ".txt",
            ".jpg",
            ".zip",
        }:
            if "checks" not in path.relative_to(OUT).parts:
                inputs[str(path.relative_to(OUT))] = sha(path)
    result = {
        "observed_date": "2026-09-05",
        "passed": True,
        "amazon_prefix_complete_rows": count_rows,
        "home_exact_occasion_cover_ids": sorted(home),
        "abo_smart_casual_bullet_ids": sorted(smart),
        "image_hash_and_decode_checks": image_count,
        "wardrobe_rows": 5443,
        "wardrobe_ethnic_category_rows": 4570,
        "fashionstylist_smart_casual_phrase_rows": 48,
        "fashionstylist_explicit_id_rows": 29,
        "fashionstylist_known_product_ids": 23,
        "fashionstylist_unresolved_shortlink_rows": 18,
        "fashionstylist_ai_marked_rows": 1,
        "fashionstylist_total_unique_product_count": None,
        "canonical_split_sha256": split_sha,
        "html_pages_checked": html_count,
        "missing_local_links": missing,
        "input_hash_count": len(inputs),
        "limits": "No teacher duplicate clearance, new labels, training, or model gain claim.",
    }
    (OUT / "input_hashes.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
