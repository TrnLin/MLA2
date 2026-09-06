"""Small public metadata audit. No images or teacher labels are downloaded/read."""

from fashion.task3_paths import resolve_task3_path

import collections
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

ROOT = Path(__file__).parent
evidence = (
    json.loads((ROOT / "evidence.json").read_text()) if (ROOT / "evidence.json").exists() else {}
)
evidence.update({"checked_date": "2026-09-05", "downloads": [], "fashionstylist": {}})


def fetch(url, name):
    if (resolve_task3_path(name, root=ROOT)).exists():
        b = (resolve_task3_path(name, root=ROOT)).read_bytes()
        evidence["downloads"].append({"url": url, "file": name, "bytes": len(b)})
        return b
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    assert len(r.content) < 26_000_000
    (resolve_task3_path(name, root=ROOT)).write_bytes(r.content)
    evidence["downloads"].append({"url": url, "file": name, "bytes": len(r.content)})
    return r.content


base = "https://raw.githubusercontent.com/recsys-benchmark/FashionStylist/main/"
fetch(base + "LICENSE", "fashionstylist_LICENSE.txt")
for subset in ["Female", "Male", "Child"]:
    item = pd.read_csv(
        io.BytesIO(
            fetch(base + f"Dataset/{subset}/label_en.csv", f"fashionstylist_{subset}_items.csv")
        )
    )
    look = pd.read_csv(
        io.BytesIO(
            fetch(base + f"Dataset/{subset}/look_en.csv", f"fashionstylist_{subset}_looks.csv")
        )
    )
    occasion = collections.Counter(
        x.strip() for val in look["occasion"].dropna() for x in val.split("/")
    )
    evidence["fashionstylist"][subset] = {
        "item_rows": len(item),
        "outfit_rows": len(look),
        "item_columns": list(item),
        "outfit_columns": list(look),
        "occasion_base_counts": dict(occasion),
        "occasion_exact_counts": look.occasion.value_counts().to_dict(),
        "style_exact_counts": item["style"].value_counts().to_dict(),
        "examples": item.head(2).fillna("").to_dict(orient="records"),
    }
en = pd.read_csv(ROOT / "fashionstylist_Male_items.csv")
zh = pd.read_csv(ROOT / "fashionstylist_Male_items_zh.csv")
matched = en[en["style"].str.contains("smart casual", case=False, na=False)].merge(
    zh[["itemID", "style", "URL link"]], on="itemID", suffixes=("_en", "_zh")
)
urls = matched["URL link"].fillna("")
is_url = urls.str.startswith(("https://", "http://"))
product_ids = urls.map(lambda value: parse_qs(urlparse(value).query).get("id", [None])[0])
record = evidence.setdefault("fashionstylist_smart_casual", {})
record.pop("unique_source_product_ids", None)
record.pop("unique_source_urls", None)
record.update(
    {
        "rows": len(matched),
        "explicit_product_id_rows": int(product_ids.notna().sum()),
        "distinct_nonempty_explicit_product_ids": int(product_ids.dropna().nunique()),
        "shortlink_rows_unresolved": int((is_url & product_ids.isna()).sum()),
        "ai_generated_marker_rows": int((urls == "AI生成").sum()),
        "ai_generated_item_ids": matched.loc[urls == "AI生成", "itemID"].tolist(),
        "real_url_rows": int(is_url.sum()),
        "distinct_real_urls": int(urls[is_url].nunique()),
        "unique_product_count_all_matches": None,
        "unique_product_count_limit": (
            "Unknown: unresolved shortlinks and one AI-generated marker; "
            "nonempty explicit IDs cover only 29 rows"
        ),
    }
)
(ROOT / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False))
print(
    {
        subset: {"items": row["item_rows"], "outfits": row["outfit_rows"]}
        for subset, row in evidence["fashionstylist"].items()
    }
)
