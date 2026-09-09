"""Conservative metadata groups; not proof of visual/product-family independence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from urllib.parse import urlsplit

from scout import OUT, save_json, write_csv

from fashion.data.families import normalize_product_name


class Union:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def join(self, left, right):
        left, right = self.find(left), self.find(right)
        self.parent[max(left, right)] = min(left, right)


def image_key(url):
    parts = urlsplit(url)
    # Different Flipkart image hosts can refer to the same stored image path.
    # This is a grouping hint, not an assertion that unseen image bytes match.
    if (parts.hostname or "").endswith(".flixcart.com"):
        return "flipkart:" + parts.path
    return (parts.hostname or "") + parts.path


def main():
    rows = json.loads((OUT / "parsed_metadata.json").read_text())
    union = Union(len(rows))
    seen = {}
    edges = []
    for index, row in enumerate(rows):
        keys = []
        if row["product_id"]:
            keys.append(("product_id", row["product_id"]))
        name = normalize_product_name(row["name"])
        brand = normalize_product_name(row["brand"])
        if name:
            keys.append(("exact_normalized_name_and_brand", brand + "|" + name))
        keys.extend(("shared_image_asset_path", image_key(url)) for url in row["image_urls"])
        for kind, value in sorted(set(keys)):
            key = kind, value
            if key in seen:
                prior = seen[key]
                union.join(index, prior)
                edges.append({"first_row": prior, "second_row": index, "basis": kind})
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[union.find(index)].append(index)
    output = []
    assignments = []
    for indices in groups.values():
        members = [rows[index] for index in indices]
        ids = sorted(row["source_id"] for row in members)
        group_id = "flipkart_meta_" + hashlib.sha256(",".join(ids).encode()).hexdigest()[:16]
        known = {
            tuple(sorted({token.casefold() for token in row["occasion_tokens"]}))
            for row in members
            if row["occasion_tokens"]
        }
        missing = sum(not row["occasion_tokens"] for row in members)
        output.append(
            {
                "metadata_group": group_id,
                "rows": len(indices),
                "unique_product_ids": len({row["product_id"] for row in members}),
                "known_occasion_sets": len(known),
                "occasion_conflict": len(known) > 1,
                "missing_occasion_rows": missing,
                "source_ids": ";".join(ids),
            }
        )
        for row in members:
            assignments.append(
                {
                    "source_id": row["source_id"],
                    "product_id": row["product_id"],
                    "metadata_group": group_id,
                    "group_rows": len(indices),
                    "group_occasion_conflict": len(known) > 1,
                    "group_missing_occasion_rows": missing,
                }
            )
    write_csv(OUT / "metadata_groups.csv", sorted(output, key=lambda row: row["metadata_group"]))
    write_csv(
        OUT / "metadata_group_assignments.csv",
        sorted(assignments, key=lambda row: row["source_id"]),
    )
    write_csv(OUT / "metadata_group_edges.csv", edges)
    summary = {
        "rows": len(rows),
        "metadata_groups": len(groups),
        "multirow_groups": sum(row["rows"] > 1 for row in output),
        "rows_in_multirow_groups": sum(row["rows"] for row in output if row["rows"] > 1),
        "groups_with_different_known_occasion_sets": sum(
            row["occasion_conflict"] for row in output
        ),
        "largest_group_rows": max(row["rows"] for row in output),
        "edge_basis_counts": dict(Counter(row["basis"] for row in edges)),
        "claim_limit": (
            "Grouping by product ID, exact normalized name plus brand, or shared image asset path; "
            "true independent product-family counts and image identity remain unverified. "
            "Conservative name collisions are possible."
        ),
    }
    assert len(assignments) == len(rows)
    save_json(OUT / "metadata_group_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
