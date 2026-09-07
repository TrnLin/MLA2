"""Join independent metadata checks; select no training data."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict

from scout import OUT, save_json, write_csv


def read_csv(name):
    return list(csv.DictReader((OUT / name).open()))


def order(row):
    return hashlib.sha256(("2753:" + row["source_id"]).encode()).hexdigest()


def main():
    rows = json.loads((OUT / "parsed_metadata.json").read_text())
    coverage = {row["source_id"]: row for row in read_csv("coverage_assignments.csv")}
    label_decisions = {int(row["row_index"]): row for row in read_csv("label_candidates.csv")}
    groups = {row["source_id"]: row for row in read_csv("metadata_group_assignments.csv")}
    reasons = Counter()
    eligible = []
    for row in rows:
        type_result = coverage[row["source_id"]]
        label_result = label_decisions[row["row_index"]]
        group = groups[row["source_id"]]
        if type_result["status"] != "matched":
            reasons["no_accepted_type_mapping"] += 1
        elif label_result["status"] != "provisional_single_label":
            reasons["no_exact_singleton_occasion_in_coarse_fashion_scope"] += 1
        elif not row["image_urls"]:
            reasons["no_image_url"] += 1
        elif group["group_occasion_conflict"] == "True":
            reasons["metadata_group_has_different_occasion_sets"] += 1
        else:
            eligible.append(
                {
                    "source_id": row["source_id"],
                    "row_index": row["row_index"],
                    "product_id": row["product_id"],
                    "metadata_group": group["metadata_group"],
                    "teacher_article_type_candidate": type_result["teacher_article_type"],
                    "source_occasion": label_result["candidate_usage"],
                    "root_category": row["root_category"],
                    "name": row["name"],
                    "source_image_url": row["image_urls"][0],
                    "image_urls": json.dumps(row["image_urls"]),
                    "source_product_url": row["product_url"],
                    "training_status": "not_admitted_metadata_candidate_only",
                }
            )
    by_group = defaultdict(list)
    for row in eligible:
        by_group[row["metadata_group"]].append(row)
    representatives = []
    for group in by_group.values():
        # Different type assignments within one metadata group are withheld too.
        if len({row["teacher_article_type_candidate"] for row in group}) > 1:
            reasons["within_group_type_disagreement_rows"] += len(group)
            continue
        chosen = min(group, key=order)
        representatives.append(chosen)
        reasons["additional_rows_in_retained_metadata_groups"] += len(group) - 1
    representatives.sort(key=lambda row: (row["source_occasion"], order(row)))
    write_csv(OUT / "deduplicated_candidates.csv", representatives)
    usage = json.loads((next(p for p in OUT.parents if (p / "pyproject.toml").is_file()) / "data/processed/taxonomy.json").read_text())["targets"][
        "usage"
    ]["classes"]
    stages = []
    for label in usage:
        retained = [row for row in representatives if row["source_occasion"] == label]
        stages.append(
            {
                "source_occasion": label,
                "eligible_rows_before_group_representative": sum(
                    row["source_occasion"] == label for row in eligible
                ),
                "metadata_group_representatives": len(retained),
                "candidate_teacher_types": len(
                    {row["teacher_article_type_candidate"] for row in retained}
                ),
            }
        )
    write_csv(OUT / "candidate_counts.csv", stages)
    # A second availability probe targets the intended exact-label candidate pool.
    # It is capped at 8 per label (<=48), fixed before any URL outcome is known.
    probe = []
    for label in usage:
        pool = [row for row in representatives if row["source_occasion"] == label]
        # Round-robin across candidate article types under deterministic hash ordering.
        bins = defaultdict(list)
        for row in pool:
            bins[row["teacher_article_type_candidate"]].append(row)
        chosen = []
        while len(chosen) < min(8, len(pool)):
            for target in sorted(bins):
                if bins[target]:
                    chosen.append(bins[target].pop(0))
                if len(chosen) == min(8, len(pool)):
                    break
        probe.extend(chosen)
    write_csv(OUT / "candidate_access_manifest.csv", probe)
    summary = {
        "eligible_rows_before_group_representative": len(eligible),
        "candidate_metadata_groups": len(representatives),
        "candidate_teacher_types": len(
            {row["teacher_article_type_candidate"] for row in representatives}
        ),
        "counts": stages,
        "exclusions_sequential_row_counts": dict(reasons),
        "targeted_access_probe_products": len(probe),
        "selection": (
            "Exact singleton canonical-name source occasion plus both coarse fashion scope "
            "and accepted exact type mapping; URLs present; metadata groups with different "
            "known occasion sets or different accepted types excluded; "
            "one hash-selected row per remaining group."
        ),
        "claim_limit": (
            "Metadata groups are conservative hints, not proven independent products. "
            "Same-name source labels remain weak labels. Full pixel/family overlap checks "
            "and image eligibility are outstanding."
        ),
        "training_rows_admitted": 0,
    }
    assert sum(row["metadata_group_representatives"] for row in stages) == len(representatives)
    assert len(representatives) + sum(reasons.values()) == len(rows)
    save_json(OUT / "candidate_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
