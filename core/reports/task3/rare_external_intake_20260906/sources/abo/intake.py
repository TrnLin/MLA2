"""Bounded official ABO intake. Cached inputs require no network."""

from fashion.task3_paths import resolve_task3_path

import csv
import gzip
import hashlib
import json
import re
import shutil
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/external/rare_usage_20260906/abo"
OLD = ROOT / "reports/task3/targeted_dataset_search_20260905"
BASE = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)


def fetch(url, path, limit):
    if path.exists():
        return path.read_bytes()
    if "--offline" in sys.argv:
        raise FileNotFoundError("offline cache miss: " + str(path))
    with urllib.request.urlopen(url, timeout=90) as r:
        b = r.read(limit + 1)
    if len(b) > limit:
        raise ValueError("byte cap exceeded: " + url)
    path.write_bytes(b)
    return b


for src, dst in [
    ("abo_readme.bin", "README.md"),
    ("abo_license.bin", "LICENSE-CC-BY-4.0.txt"),
    ("abo_shard_0.bin", "listings_0.json.gz"),
]:
    if not (RAW / dst).exists():
        shutil.copyfile(OLD / "catalogue/raw" / src, RAW / dst)
fetch(BASE + "listings/README.md", RAW / "listings_README.md", 100000)
records = []
shards = []
for i in range(16):
    p = RAW / f"listings_{i:x}.json.gz"
    used = sum(f.stat().st_size for f in RAW.iterdir())
    b = fetch(BASE + f"listings/metadata/listings_{i:x}.json.gz", p, 150_000_000 - used)
    rr = [json.loads(line) for line in gzip.decompress(b).splitlines()]
    records.extend(rr)
    shards.append(dict(shard=i, bytes=len(b), rows=len(rr), sha256=hashlib.sha256(b).hexdigest()))
    print("shard", i, len(rr), flush=True)


def values(r, key):
    return [str(v.get("value", "")) for v in r.get(key, []) if isinstance(v, dict)]


hits = []
groups = defaultdict(list)
for r in records:
    groups[r["item_id"]].append(r)
    evidence = []
    for k in ["item_name", "bullet_point", "style", "occasion", "product_description"]:
        for v in r.get(k, []):
            if not isinstance(v, dict):
                continue
            s = str(v.get("value", ""))
            if re.search(r"smart[ -]?casual|\btravel\b", s, re.I):
                evidence.append(dict(field=k, **v))
    if evidence:
        hits.append(dict(item_id=r["item_id"], record=r, evidence=evidence))
(OUT / "term_hits.json").write_text(json.dumps(hits, ensure_ascii=False, indent=2))
(OUT / "metadata_audit.json").write_text(
    json.dumps(
        dict(
            shards=shards,
            total_rows=len(records),
            unique_item_ids=len(groups),
            term_hit_rows=len(hits),
        ),
        indent=2,
    )
)
print("HITS", len(hits), "unique", len(set(h["item_id"] for h in hits)), flush=True)

selected = {}
excluded = []
for h in hits:
    r = h["record"]
    sid = r["item_id"]
    titles = values(r, "item_name")
    title = next(
        (v["value"] for v in r.get("item_name", []) if v.get("language_tag", "").startswith("en")),
        titles[0] if titles else "",
    )
    typ = ",".join(values(r, "product_type"))
    smart = [v for v in h["evidence"] if re.fullmatch(r"smart[ -]?casual", v["value"], re.I)]
    travel = [v for v in h["evidence"] if re.search(r"\btravel\b", v["value"], re.I)]
    # Bags only: exclude suitcases, packing inserts, hygiene cases and non-fashion goods.
    bag = typ in ["BACKPACK", "HANDBAG", "TOTE_BAG", "DUFFEL_BAG"] or (
        typ == "LUGGAGE" and re.search(r"\b(duffel|backpack|travel bag)\b", title, re.I)
    )
    if smart:
        label = "Smart Casual"
        strength = "source_exact_style"
        ev = smart
    elif travel and bag:
        label = "Travel"
        ev = travel
        strength = "weak_style_text" if any(v["field"] == "style" for v in travel) else "weak_title"
    else:
        excluded.append(
            dict(
                source_id=sid,
                reason="No standalone Smart Casual evidence or not a compatible bag type",
                source_article_type=typ,
                source_title=title,
            )
        )
        continue
    mixed = (
        bool(
            re.search(
                r"casual|school|gym|hiking|camping|shopping|work,|work and|everyday",
                " ".join(v["value"] for v in ev),
                re.I,
            )
        )
        if label == "Travel"
        else False
    )
    if sid not in selected or strength == "weak_style_text":
        selected[sid] = dict(
            record=r,
            source_id=sid,
            source_title=title,
            source_article_type=typ,
            source_label=label,
            label_evidence=json.dumps(ev, ensure_ascii=False),
            label_strength=strength,
            mixed_use=mixed,
        )

# Keep strong source fields first, then single-purpose title evidence, then mixed use.
items = sorted(
    selected.values(),
    key=lambda x: (
        x["source_label"],
        x["label_strength"] == "weak_title",
        x["mixed_use"],
        x["source_id"],
    ),
)
counts = Counter()
chosen = []
for c in items:
    if counts[c["source_label"]] >= 50:
        excluded.append(dict(source_id=c["source_id"], reason="50 image class cap"))
        continue
    counts[c["source_label"]] += 1
    chosen.append(c)
index_path = RAW / "images.csv.gz"
if not index_path.exists():
    shutil.copyfile(OLD / "travel/abo_access/images.csv.gz", index_path)
with gzip.open(index_path, "rt") as f:
    index = {r["image_id"]: r for r in csv.DictReader(f)}
rows = []
failures = []
native = {}
for c in chosen:
    r = c.pop("record")
    sid = c["source_id"]
    native[sid] = groups[sid]
    entry = index.get(r.get("main_image_id"))
    if not entry:
        failures.append(dict(source_id=sid, error="missing official main_image_id index join"))
        continue
    path = DATA / (sid + ".jpg")
    old = OLD / "travel/abo_access" / (sid + ".jpg")
    if old.exists() and not path.exists():
        shutil.copyfile(old, path)
    url = BASE + "images/original/" + entry["path"]
    try:
        total = sum(p.stat().st_size for p in DATA.glob("*.jpg"))
        b = fetch(url, path, min(2 * 1024 * 1024, 100_000_000 - total))
        with Image.open(path) as im:
            im.load()
            width, height = im.size
        assert (width, height) == (int(entry["width"]), int(entry["height"]))
    except Exception as e:
        failures.append(dict(source_id=sid, error=str(e)))
        continue
    model = "|".join(values(r, "model_number"))
    brand = "|".join(values(r, "brand"))
    family = re.sub(r"[^a-z0-9]", "", model.lower()) or re.sub(
        r"[^a-z0-9]", "", c["source_title"].lower()
    )
    # Collapse obvious model/color siblings; retain original model and all native rows separately.
    for stem in [
        "nc1712056",
        "nc1712002",
        "nc1808277",
        "zh1603233",
        "zh1603219",
        "wxd0989",
        "sollgge",
        "nc1504157r1",
        "zh1510024r3",
    ]:
        if family.startswith(stem):
            family = stem
    brand_group = re.sub(r"[^a-z0-9]", "", brand.lower())
    if "amazonbasics" in brand_group:
        brand_group = "amazonbasics"
    group = "abo:" + brand_group + ":" + family
    rows.append(
        dict(
            source="Amazon Berkeley Objects",
            source_group_id=group,
            source_url="https://" + r["domain_name"] + "/dp/" + sid,
            image_url=url,
            original_path=str(path.relative_to(ROOT)),
            original_sha256=hashlib.sha256(b).hexdigest(),
            width=width,
            height=height,
            rights_basis="Official ABO release CC BY 4.0; see source license and attribution",
            main_image_id=r["main_image_id"],
            source_brand=brand,
            source_model_number=model,
            **c,
        )
    )
    print("image", sid, c["source_label"], flush=True)
qa_path = OUT / "visual_qa.json"
qa = json.loads(qa_path.read_text()) if qa_path.exists() else {}
for row in rows:
    flags = [
        key for key, value in qa.items() if isinstance(value, list) and row["source_id"] in value
    ]
    row["visual_qa_flags"] = "|".join(flags) or "gross_visual_check_pass"
with (OUT / "candidates.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
    w.writeheader()
    w.writerows(rows)
(OUT / "selected_native_records.json").write_text(json.dumps(native, ensure_ascii=False, indent=2))
(OUT / "exclusions.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2))
(OUT / "image_failures.json").write_text(json.dumps(failures, indent=2))
summary = dict(
    shards=shards,
    total_rows=len(records),
    unique_item_ids=len(groups),
    term_hit_rows=len(hits),
    eligible_before_cap=dict(Counter(c["source_label"] for c in items)),
    downloaded=dict(Counter(c["source_label"] for c in rows)),
    image_bytes=sum(p.stat().st_size for p in DATA.glob("*.jpg")),
    failures=failures,
    unique_source_groups=len(set(r["source_group_id"] for r in rows)),
    missing_smart_casual_article_types=["shirts", "watches"],
    metadata_bytes=sum(p.stat().st_size for p in RAW.iterdir()),
    scan_scope=(
        "Literal Smart Casual and Travel in original title, bullet, style, occasion, "
        "description; no translated synonyms"
    ),
)
(OUT / "metadata_audit.json").write_text(json.dumps(summary, indent=2))
for start in range(0, len(rows), 30):
    batch = rows[start : start + 30]
    sheet = Image.new("RGB", (1000, ((len(batch) + 4) // 5) * 175), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for j, r in enumerate(batch):
        with Image.open(resolve_task3_path(r["original_path"], root=ROOT)) as im:
            im.thumbnail((185, 135))
            x = (j % 5) * 200
            y = (j // 5) * 175
            sheet.paste(im, (x + (190 - im.width) // 2, y))
        draw.text((x + 4, y + 137), r["source_id"], fill="black")
        draw.text((x + 4, y + 152), r["source_label"], fill="black")
    sheet.save(OUT / f"contact_sheet_{start // 30 + 1}.jpg")
print(json.dumps(summary, indent=2))
