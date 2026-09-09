"""Bounded release and sample checks; no teacher access."""

import csv
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
log = []


def get(url, name, cap=1_000_000):
    with urllib.request.urlopen(url, timeout=20) as response:
        data = response.read(cap + 1)
    if len(data) > cap:
        raise RuntimeError("Download cap exceeded")
    (OUT / name).write_bytes(data)
    log.append({"url": url, "name": name, "bytes": len(data)})
    return data


for slug, name in [("jeffreyszhou/ikea-us-products-2025", "ikea_original_revision.json")]:
    get("https://huggingface.co/api/datasets/" + slug + "/revision/main", name)
with urllib.request.urlopen(
    urllib.request.Request(
        "https://huggingface.co/datasets/tsazan/ikea-us-commercetxt/resolve/main/README.md",
        method="HEAD",
    ),
    timeout=20,
) as response:
    (OUT / "ikea_derivative_revision.json").write_text(json.dumps(dict(response.headers), indent=2))
get(
    "https://www.kaggle.com/api/v1/datasets/list?search=wardrobe-assistant", "wardrobe_listing.json"
)
data = get(
    "https://www.kaggle.com/api/v1/datasets/download/shahzaibmalik44/wardrobe-assistant",
    "wardrobe_assistant.zip",
)
summary = {}
with zipfile.ZipFile(io.BytesIO(data)) as z:
    summary["archive_files"] = [{"name": f.filename, "bytes": f.file_size} for f in z.infolist()]
    for name in z.namelist():
        if name.endswith(".csv"):
            rows = list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig"))))
            summary[name] = {
                "rows": len(rows),
                "columns": list(rows[0]),
                "first_rows": rows[:2],
                "ethnic_values": {
                    key: sum("ethnic" in row[key].lower() for row in rows) for key in rows[0]
                },
            }
summary["images"] = []
for file in sorted(OUT.glob("throw-pillow-covers_*.txt"))[:2]:
    url = re.search(r"Main: (https://[^\s]+)", file.read_text())[1]
    name = file.stem + ".jpg"
    try:
        get(url, name, 500_000)
        summary["images"].append(name)
    except Exception as exc:
        summary["images"].append({"error": str(exc), "url": url})
summary["downloads"] = log
(OUT / "followup_checks.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
