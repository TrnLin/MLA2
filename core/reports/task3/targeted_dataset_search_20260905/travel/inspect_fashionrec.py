"""Read publisher metadata without executing publisher code.

Requires duckdb, installed in raw/_reader or available to the active interpreter.
Use ./.venv/bin/python to run this file.
"""

import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "raw/_reader"))
sys.path.insert(0, "/tmp/travel-dataset-reader-20260905")
duckdb = importlib.import_module("duckdb")

path = ROOT / "raw/outfits_lite.parquet"
relation = duckdb.read_parquet(str(path))
columns = relation.columns
rows = [dict(zip(columns, r)) for r in relation.fetchall()]
fashion = [r for r in rows if r["source"] == "fashion32"]
travel = [
    r
    for r in fashion
    if "travel" in str(r["occasion"]).lower()
    or "旅行" in str(r["occasion"])
    or "旅游" in str(r["occasion"])
]
evidence = {
    "source_url": "https://huggingface.co/datasets/Anony100/FashionRec/resolve/main/meta/outfits_lite.parquet",
    "observed_date": "2026-09-05",
    "file_bytes": path.stat().st_size,
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "columns": columns,
    "rows": len(rows),
    "source_counts": dict(Counter(r["source"] for r in rows)),
    "fashion32_occasion_counts": dict(Counter(str(r["occasion"]) for r in fashion)),
    "travel_match_rows": len(travel),
    "travel_category_counts": dict(Counter(str(r["categories"]) for r in travel)),
    "travel_examples": travel[:5],
}
(ROOT / "fashionrec_metadata_audit.json").write_text(
    json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
)
print(
    json.dumps(
        {
            k: v
            for k, v in evidence.items()
            if k not in ("travel_examples", "travel_category_counts")
        },
        ensure_ascii=False,
        indent=2,
    )
)
