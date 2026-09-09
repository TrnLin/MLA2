"""Verify saved provenance, joins, images and image-only overlap evidence offline."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from fashion.data.perceptual import compute_image_hashes

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent


def csv_rows(path):
    return list(csv.DictReader(path.open()))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    provenance = json.loads((OUT / "source_provenance.json").read_text())
    plan = json.loads((OUT / "pilot_intake_plan.json").read_text())
    raw = resolve_task3_path(provenance["local_csv"], root=ROOT)
    assert digest(raw) == provenance["csv_sha256"]
    assert digest(raw.parent / "flipkart-products-v1.zip") == plan["source_archive_sha256"]
    assert digest(ROOT / "data/processed/splits.csv") == plan["teacher_contract"]["split_sha256"]
    raw_rows = csv_rows(raw)
    parsed = json.loads((OUT / "parsed_metadata.json").read_text())
    assert len(raw_rows) == len(parsed) == 20_000
    assert [row["uniq_id"] for row in raw_rows] == [row["source_id"] for row in parsed]
    # Independent literal-field count checks against the original CSV representation.
    raw_occasion_keys = sum(
        len(re.findall(r'"key"\s*=>\s*"Occasion"', row["product_specifications"]))
        for row in raw_rows
    )
    assert raw_occasion_keys == sum(len(row["occasion_entries"]) for row in parsed) == 9954
    types = json.loads((ROOT / "data/processed/taxonomy.json").read_text())["targets"][
        "articleType"
    ]["classes"]
    coverage = csv_rows(OUT / "teacher_type_coverage.csv")
    assignments = csv_rows(OUT / "coverage_assignments.csv")
    assert [row["teacher_article_type"] for row in coverage] == types
    assert len({row["source_id"] for row in assignments}) == len(parsed)
    assert sum(int(row["matched_product_rows"]) for row in coverage) == sum(
        row["status"] == "matched" for row in assignments
    )
    candidates = csv_rows(OUT / "deduplicated_candidates.csv")
    assert len(candidates) == len({row["metadata_group"] for row in candidates}) == 3094
    by_id = {row["source_id"]: row for row in parsed}
    assignment_by_id = {row["source_id"]: row for row in assignments}
    groups = {row["source_id"]: row for row in csv_rows(OUT / "metadata_group_assignments.csv")}
    for candidate in candidates:
        row = by_id[candidate["source_id"]]
        assert row["occasion_tokens"] == [candidate["source_occasion"]]
        assert candidate["source_image_url"] == row["image_urls"][0]
        assert groups[candidate["source_id"]]["group_occasion_conflict"] == "False"
        assert assignment_by_id[candidate["source_id"]]["status"] == "matched"
        assert candidate["training_status"] == "not_admitted_metadata_candidate_only"
    # Explicit column allowlists exclude all teacher target labels.
    cached_path = ROOT / "data/processed/audit/perceptual_hashes.csv.gz"
    cached = pd.read_csv(
        cached_path,
        usecols=["id", "path", "sha256", "role", "dhash_hex", "ahash_hex"],
        dtype=str,
    )
    for role, filename in [("labelled", "splits.csv"), ("prediction", "prediction_manifest.csv")]:
        inventory = pd.read_csv(
            ROOT / "data/processed" / filename,
            usecols=["id", "path", "sha256"],
            dtype=str,
        )
        expected = set(map(tuple, inventory[["id", "path", "sha256"]].values))
        actual = set(map(tuple, cached.loc[cached.role == role, ["id", "path", "sha256"]].values))
        assert expected == actual
    assert len(cached) == 44_441
    teacher_sha = set(cached.sha256)
    teacher_dhash = [int(value, 16) for value in cached.dhash_hex]
    images = []
    sample_product_ids = set()
    for result_name, manifest_name in [
        ("image_access.csv", "sample_manifest.csv"),
        ("candidate_image_access.csv", "candidate_access_manifest.csv"),
    ]:
        results = csv_rows(OUT / result_name)
        manifest = {row["product_id"]: row for row in csv_rows(OUT / manifest_name)}
        assert {row["product_id"] for row in results} == set(manifest)
        for row in results:
            assert row["product_id"] not in sample_product_ids
            sample_product_ids.add(row["product_id"])
            original = json.loads(manifest[row["product_id"]]["image_urls"])[0]
            assert row["source_url"] == original
            for attempt in json.loads(row["attempts_json"]):
                assert attempt["request_url"] in [original, re.sub(r"^http:", "https:", original)]
            if row["usable"] != "True":
                continue
            path = resolve_task3_path(row["local_path"], root=ROOT)
            assert digest(path) == row["sha256"]
            with Image.open(path) as image:
                image.load()
                assert image.size == (int(row["width"]), int(row["height"]))
            dhash, ahash = compute_image_hashes(path)
            assert f"{dhash:016x}" == row["dhash_hex"]
            assert f"{ahash:016x}" == row["ahash_hex"]
            assert row["sha256"] not in teacher_sha
            assert all((dhash ^ teacher).bit_count() > 2 for teacher in teacher_dhash)
            images.append(path)
    assert len(sample_product_ids) == 89 and len(images) == 78
    dependencies = [
        raw,
        raw.parent / "flipkart-products-v1.zip",
        ROOT / "data/processed/splits.csv",
        ROOT / "data/processed/prediction_manifest.csv",
        ROOT / "data/processed/taxonomy.json",
        ROOT / "data/processed/development_class_summary.csv",
        cached_path,
    ] + images
    dependencies += sorted(OUT.glob("*.py"))
    dependencies += [
        OUT / "coverage_rules.json",
        OUT / "parsed_metadata.json",
        OUT / "deduplicated_candidates.csv",
        OUT / "metadata_group_assignments.csv",
        OUT / "pilot_intake_plan.json",
    ]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in dependencies}
    (OUT / "input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    summary = {
        "verified": True,
        "source_rows": len(parsed),
        "raw_occasion_fields_independently_counted": raw_occasion_keys,
        "teacher_type_rows": len(coverage),
        "candidate_metadata_groups": len(candidates),
        "sampled_products": len(sample_product_ids),
        "verified_saved_images": len(images),
        "saved_image_bytes": sum(path.stat().st_size for path in images),
        "teacher_cached_images": len(cached),
        "sample_exact_or_dhash_at_most_two_matches": 0,
        "canonical_split_unchanged": True,
        "protected_teacher_targets_read": False,
        "training_rows_admitted": 0,
        "network_requests": 0,
        "dependency_hashes": len(hashes),
        "limits": [
            "Metadata matches and exact name overlap do not validate image or occasion labels.",
            "Cached overlap check applies to these 78 files and the declared threshold only.",
            "No model was fitted or evaluated; no accuracy gain is claimed.",
        ],
    }
    (OUT / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
