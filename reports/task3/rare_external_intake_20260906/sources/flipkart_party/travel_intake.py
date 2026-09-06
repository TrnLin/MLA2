"""Reuse the one verified native singleton Travel candidate; no network."""

from fashion.task3_paths import resolve_task3_path

import hashlib
import json

from acquire import DEST, OLD, OUT, ROOT, read, write
from PIL import Image


def main():
    rows = [
        r for r in read(OLD / "deduplicated_candidates.csv") if r["source_occasion"] == "Travel"
    ]
    assert len(rows) == 1
    r = rows[0]
    m = next(
        x
        for x in json.loads((OLD / "parsed_metadata.json").read_text())
        if x["product_id"] == r["product_id"]
    )
    assert m["occasion_tokens"] == ["Travel"]
    cached = next(
        x
        for x in read(OLD / "candidate_image_access.csv")
        if x["product_id"] == r["product_id"] and x["usable"].lower() == "true"
    )
    assert cached["source_url"] == r["source_image_url"] == m["image_urls"][0]
    payload = (resolve_task3_path(cached["local_path"], root=ROOT)).read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    assert sha == cached["sha256"]
    with Image.open(resolve_task3_path(cached["local_path"], root=ROOT)) as image:
        image.load()
        width, height = image.size
    path = DEST / (r["product_id"] + ".image")
    path.write_bytes(payload)
    row = dict(
        source="flipkart_travel",
        source_id=r["product_id"],
        source_group_id=r["metadata_group"],
        source_title=r["name"],
        source_article_type=r["teacher_article_type_candidate"],
        source_label="Travel",
        label_evidence=json.dumps(m["occasion_entries"]),
        label_strength="source_exact_occasion",
        source_url=r["source_product_url"],
        image_url=r["source_image_url"],
        original_path=str(path.relative_to(ROOT)),
        original_sha256=sha,
        width=width,
        height=height,
        rights_basis=("Publisher metadata CC BY-SA 4.0; remote photo rights scope not established"),
        source_record_id=r["source_id"],
        source_category_path=m["category_path"],
        source_specs=json.dumps(m["specs"]),
        attribution="Flipkart Products, PromptCloud / PromptCloudHQ, version 1, via Kaggle",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        dataset_url="https://www.kaggle.com/datasets/PromptCloudHQ/flipkart-products",
    )
    write("travel_candidate.csv", [row])
    summary = dict(
        source_id=r["product_id"],
        decoded=1,
        reused=1,
        new_body_bytes=0,
        original_bytes=len(payload),
        width=width,
        height=height,
        visual_qa="Original inspected: one brown shoulder/duffel-style bag on white background; "
        "no model, collage, or gross object mismatch. Preserve native Travel occasion and Handbags "
        "mapping. Main product fills frame; straps approach frame edge.",
        final_status="candidate_only; root handles duplicate and admission checks",
    )
    (OUT / "travel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
