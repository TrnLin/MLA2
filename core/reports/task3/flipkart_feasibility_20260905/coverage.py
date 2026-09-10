"""Conservative metadata-only type coverage; run from the repository root."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = next(candidate for candidate in BASE.parents if (candidate / "pyproject.toml").is_file())


def norm(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def write_csv(name, rows):
    with (BASE / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    taxonomy = json.loads((ROOT / "data/processed/taxonomy.json").read_text())
    classes = taxonomy["targets"]["articleType"]["classes"]
    shared = set(taxonomy["targets"]["usage"]["classes"]) - {"NA"}
    rules = json.loads((BASE / "coverage_rules.json").read_text())
    lookup = {}
    for target, aliases in rules["aliases"].items():
        assert target in classes, target
        for alias in aliases:
            lookup.setdefault(norm(alias), set()).add(target)
    products = json.loads((BASE / "parsed_metadata.json").read_text())
    assignments = []
    for row in products:
        evidence = []
        for field, values in [
            ("category_segment", row["segments"]),
            (
                "Type",
                [
                    s.get("value", "")
                    for s in row["specs"]
                    if isinstance(s, dict) and s.get("key") == "Type"
                ],
            ),
        ]:
            for value in values:
                for target in sorted(lookup.get(norm(value), [])):
                    evidence.append({"target": target, "field": field, "value": value})
        candidates = sorted({e["target"] for e in evidence})
        # Broad category bins can include mixed products and unstitched fabric.
        mixed = bool(
            re.search(
                r"\b(combo|combination|material|unstitched|semi[- ]stitched)\b", row["name"], re.I
            )
        )
        incompatible = any(
            norm(s) in {norm(v) for v in rules["ambiguous_context_segments"]}
            for s in row["segments"]
        )
        status = (
            "matched"
            if len(candidates) == 1 and not mixed and not incompatible
            else "uncertain"
            if candidates
            else "unmapped"
        )
        tokens = sorted(set(row["occasion_tokens"]))
        exact = len(tokens) == 1 and tokens[0] in shared
        assignments.append(
            {
                "row_index": row["row_index"],
                "source_id": row["source_id"],
                "product_id": row["product_id"],
                "product_name": row["name"],
                "status": status,
                "teacher_article_type": candidates[0] if status == "matched" else "",
                "candidate_types": json.dumps(candidates),
                "matched_evidence": json.dumps(evidence, ensure_ascii=False),
                "uncertainty": "incompatible_or_ambiguous_category_context"
                if incompatible and candidates
                else "mixed_or_material_name"
                if mixed and candidates
                else "conflicting_type_evidence"
                if len(candidates) > 1
                else "no_exact_rule"
                if not candidates
                else "metadata_only_not_image_validated",
                "source_occasion_present": bool(row["occasions"]),
                "occasion_values": json.dumps(row["occasions"]),
                "exact_single_shared_occasion": exact,
                "shared_occasion": tokens[0] if exact else "",
            }
        )
    development = {}
    with (ROOT / "data/processed/development_class_summary.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["target"] == "articleType":
                development[row["class"]] = int(row["development_product_count"])
    coverage = []
    for target in classes:
        matched = [r for r in assignments if r["teacher_article_type"] == target]
        uncertain = [
            r
            for r in assignments
            if r["status"] == "uncertain" and target in json.loads(r["candidate_types"])
        ]
        coverage.append(
            {
                "teacher_article_type": target,
                "development_products": development[target],
                "matched_product_rows": len(matched),
                "unique_product_ids": len({r["product_id"] for r in matched if r["product_id"]}),
                "source_occasion_present_rows": sum(r["source_occasion_present"] for r in matched),
                "exact_single_shared_occasion_rows": sum(
                    r["exact_single_shared_occasion"] for r in matched
                ),
                "uncertain_candidate_rows": len(uncertain),
                "gap": "metadata_match_only"
                if matched
                else "uncertain_candidates_only"
                if uncertain
                else "no_match_under_conservative_rules",
            }
        )
    summary = {
        "total_source_rows": len(products),
        "teacher_types": len(classes),
        "rule_accepted_metadata_types": sum(r["matched_product_rows"] > 0 for r in coverage),
        "types_with_exact_single_shared_occasion": sum(
            r["exact_single_shared_occasion_rows"] > 0 for r in coverage
        ),
        "status_rows": dict(Counter(r["status"] for r in assignments)),
        "matched_rows_with_occasion": sum(
            r["source_occasion_present"] for r in assignments if r["status"] == "matched"
        ),
        "matched_rows_exact_single_shared_occasion": sum(
            r["exact_single_shared_occasion"] for r in assignments if r["status"] == "matched"
        ),
        "missing_types": [
            r["teacher_article_type"] for r in coverage if not r["matched_product_rows"]
        ],
        "checks": {
            "all_124_types": len(coverage) == 124,
            "one_assignment_per_source_row": len(assignments) == len(products),
            "unique_row_index": len({r["row_index"] for r in assignments}) == len(products),
            "matched_count_reconciles": sum(r["matched_product_rows"] for r in coverage)
            == sum(r["status"] == "matched" for r in assignments),
        },
    }
    assert all(summary["checks"].values()), summary["checks"]
    write_csv("teacher_type_coverage.csv", coverage)
    write_csv("coverage_assignments.csv", assignments)
    (BASE / "coverage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
