"""Read-only source-label audit; no image or protected-label reads."""

import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(candidate for candidate in HERE.parents if (candidate / "pyproject.toml").is_file())


def counts(items):
    return {
        "rows": len(items),
        "unique_product_ids": len({x["product_id"] for x in items if x["product_id"]}),
    }


def main():
    rows = json.loads((HERE / "parsed_metadata.json").read_text())
    canonical = json.loads((ROOT / "data/processed/taxonomy.json").read_text())["targets"]["usage"][
        "classes"
    ]
    fashion_roots = {"Clothing", "Footwear", "Jewellery", "Watches", "Bags, Wallets & Belts"}
    exact = set(canonical) - {"NA"}

    summary = {
        "canonical_usage": canonical,
        "all": counts(rows),
        "with_occasion": counts([r for r in rows if r["occasion_tokens"]]),
        "multiple_tokens": counts([r for r in rows if len(r["occasion_tokens"]) > 1]),
        "multiple_canonical_tokens": counts(
            [r for r in rows if len(set(r["occasion_tokens"]) & exact) > 1]
        ),
        "strict_policy": (
            "Known broad fashion root AND exactly one source token AND exact canonical name; "
            "provisional weak labels, not semantic validation."
        ),
        "strict_by_label": {},
        "all_singleton_exact_by_label": {},
        "token_counts": {},
        "root_filter": sorted(fashion_roots),
        "cross_row_pid_label_conflicts": [],
    }
    for label in canonical:
        summary["all_singleton_exact_by_label"][label] = counts(
            [r for r in rows if r["occasion_tokens"] == [label]]
        )
        summary["strict_by_label"][label] = counts(
            [
                r
                for r in rows
                if r["occasion_tokens"] == [label] and r["root_category"] in fashion_roots
            ]
        )
    for token in sorted({t for r in rows for t in r["occasion_tokens"]}):
        summary["token_counts"][token] = counts([r for r in rows if token in r["occasion_tokens"]])
    pid_labels = defaultdict(set)
    for r in rows:
        pid_labels[r["product_id"]].add(tuple(r["occasion_tokens"]))
    summary["cross_row_pid_label_conflicts"] = [
        {"product_id": p, "labels": sorted(v)} for p, v in pid_labels.items() if len(v) > 1
    ]
    with (HERE / "label_candidates.csv").open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "product_id",
                "root_category",
                "occasion_tokens",
                "candidate_usage",
                "status",
            ],
        )
        writer.writeheader()
        for r in rows:
            tokens = r["occasion_tokens"]
            strict = len(tokens) == 1 and tokens[0] in exact and r["root_category"] in fashion_roots
            status = (
                "provisional_single_label"
                if strict
                else ("missing_occasion" if not tokens else "auxiliary_only_or_excluded")
            )
            writer.writerow(
                {k: r[k] for k in ["row_index", "product_id", "root_category"]}
                | {
                    "occasion_tokens": "|".join(tokens),
                    "candidate_usage": tokens[0] if strict else "",
                    "status": status,
                }
            )
    grouped = defaultdict(list)
    for r in rows:
        for t in r["occasion_tokens"]:
            grouped[r["root_category"], t].append(r)
    with (HERE / "occasion_by_product_group.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=["root_category", "occasion", "rows", "unique_product_ids"]
        )
        writer.writeheader()
        for (group, token), rr in sorted(grouped.items()):
            writer.writerow({"root_category": group, "occasion": token, **counts(rr)})
    (HERE / "label_compatibility.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "token_counts"}, indent=2))


if __name__ == "__main__":
    main()
