"""Build the explicit auxiliary intake after source, visual, and image-only checks."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.data.external_usage import (
    ACCEPTED_EVIDENCE,
    ExternalUsageDataset,
    assign_external_partitions,
    file_sha256,
    fit_external_statistics,
    prepare_external_image,
    validate_external_manifest,
)
from fashion.data.external_usage_audit import (
    find_teacher_overlaps,
    fingerprint_image,
    link_external_families,
)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/external/rare_usage_20260906"
EXPECTED_TEACHER_SPLIT = "d76a49c6dc7999b4f286e94838a92c603d68c4f66179fde081028948f6a187db"
INPUTS = [
    "sources/flipkart_party/candidates.csv",
    "sources/flipkart_party/travel_candidate.csv",
    "sources/amazon/candidates.csv",
    "sources/abo/candidates.csv",
]


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def source_candidates():
    frames = [read_csv(OUT / path) for path in INPUTS]
    frame = pd.concat(frames, ignore_index=True).fillna("")
    frame["external_id"] = frame.source + ":" + frame.source_id
    if frame.external_id.duplicated().any():
        raise ValueError("duplicate source identifiers in combined input")
    policy = json.loads((OUT / "visual_selection.json").read_text())
    home_index = read_csv(OUT / policy["home_contact_index"])
    home_keep = set(
        home_index.loc[
            home_index.contact_number.astype(int).isin(policy["home_keep_contact_numbers"]),
            "source_id",
        ]
    )
    home_art = set(
        home_index.loc[
            home_index.contact_number.astype(int).isin(policy["home_artwork_exclusion_numbers"]),
            "source_id",
        ]
    )
    party_qa = read_csv(OUT / "sources/flipkart_party/visual_qa.csv").set_index("source_id")
    visual_rows = []
    for _, row in frame.iterrows():
        reason, decision, flags = "", "keep", ""
        if row.label_strength not in ACCEPTED_EVIDENCE:
            decision, reason = (
                "withhold",
                "Weak source purpose/type text is not an exact occasion/style label",
            )
        elif row.source_label == "Home":
            if row.source_id not in home_keep:
                decision = "withhold"
                reason = (
                    "Flat print artwork without clear product form"
                    if row.source_id in home_art
                    else policy["home_other_exclusion_reason"]
                )
            else:
                flags = "One clear cushion/cover; original and 60x80 preview inspected"
        elif row.source == "flipkart_party":
            qa = party_qa.loc[row.source_id]
            flags = str(qa.visual_flags)
            if any(flag in flags for flag in policy["party_exclude_flags"]):
                decision, reason = "withhold", "Collage or unresolved source product-type mismatch"
            if int(qa.contact_index) in policy["party_extra_exclude_contact_numbers"]:
                decision, reason = "withhold", policy["party_extra_exclude_reason"]
        elif row.source_label == "Smart Casual":
            flags = "Single boot/loafer photo; repeated source images need duplicate removal"
        elif row.source == "flipkart_travel":
            flags = "Single bag on white; original visually inspected"
        else:
            decision, reason = "withhold", "No source-specific visual acceptance rule"
        if row.label_strength in ACCEPTED_EVIDENCE:
            # Check actual existing evidence, not merely the strength label supplied by a scout.
            if row.source.startswith("flipkart"):
                assert json.loads(row.label_evidence) == [row.source_label]
            elif row.source_label == "Home":
                assert (
                    row.label_evidence == "details.Occasion=Home"
                    or row.label_evidence.casefold() == "details.occasion=home"
                )
            elif row.source_label == "Smart Casual":
                assert any(
                    r.get("value", "").casefold().strip() == "smart casual"
                    for r in json.loads(row.label_evidence)
                )
        visual_rows.append(
            {
                "external_id": row.external_id,
                "visual_decision": decision,
                "visual_flags": flags,
                "selection_reason": reason,
            }
        )
    qa = pd.DataFrame(visual_rows)
    qa.to_csv(OUT / "visual_decisions.csv", index=False)
    return frame.merge(qa, on="external_id", validate="one_to_one")


def teacher_references():
    """Read image/identity fields only; validate every cached file against its SHA."""
    columns = ["id", "path", "sha256", "partition"]
    labelled = pd.read_csv(
        ROOT / "data/processed/splits.csv", usecols=columns, dtype=str, keep_default_na=False
    )
    labelled = labelled.rename(columns={"partition": "audit_role"})
    prediction = pd.read_csv(
        ROOT / "data/processed/prediction_manifest.csv",
        usecols=["id", "path", "sha256"],
        dtype=str,
        keep_default_na=False,
    )
    prediction["audit_role"] = "prediction"
    inventory = (
        pd.concat([labelled, prediction], ignore_index=True)
        .sort_values(["audit_role", "id"])
        .reset_index(drop=True)
    )
    inventory_digest = hashlib.sha256(inventory.to_csv(index=False).encode()).hexdigest()
    cache_path, meta_path = (
        DATA / "teacher_image_fingerprints_v2.csv",
        DATA / "teacher_image_fingerprints_v2.json",
    )
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["inventory_sha256"] != inventory_digest or meta["cache_sha256"] != file_sha256(
            cache_path
        ):
            raise ValueError("Teacher image inventory changed; cached comparison is invalid")
        references = read_csv(cache_path)
        if len(references) != len(inventory):
            raise ValueError("Incomplete teacher fingerprint cache")

        def verify_current_file(row):
            if file_sha256(resolve_task3_path(row["path"], root=ROOT)) != row["sha256"]:
                raise ValueError("Teacher image bytes changed: " + row["path"])

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(verify_current_file, inventory.to_dict("records")))
        return references

    def check(row):
        fingerprint = fingerprint_image(resolve_task3_path(row["path"], root=ROOT))
        if fingerprint["file_sha256"] != row["sha256"]:
            raise ValueError("Teacher image differs from the canonical inventory: " + row["path"])
        return {**row, **fingerprint}

    print(f"Hashing and decoding all {len(inventory)} teacher-role images (no labels).", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        references = pd.DataFrame(pool.map(check, inventory.to_dict("records")))
    # Also confirm parity with the independently saved canonical perceptual cache.
    old = pd.read_csv(
        ROOT / "data/processed/audit/perceptual_hashes.csv.gz",
        usecols=["id", "path", "sha256", "dhash_hex", "ahash_hex"],
        dtype=str,
    )
    merged = references.merge(
        old, on=["id", "path", "sha256"], validate="one_to_one", suffixes=("", "_canonical")
    )
    assert len(merged) == len(references)
    assert merged.dhash_hex.eq(merged.dhash_hex_canonical).all()
    assert merged.ahash_hex.eq(merged.ahash_hex_canonical).all()
    references.to_csv(cache_path, index=False)
    save_json(
        meta_path,
        {
            "inventory_sha256": inventory_digest,
            "cache_sha256": file_sha256(cache_path),
            "image_count": len(references),
            "all_original_file_hashes_verified": True,
        },
    )
    return references


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    split_path = ROOT / "data/processed/splits.csv"
    prediction_path = ROOT / "data/processed/prediction_manifest.csv"
    if file_sha256(split_path) != EXPECTED_TEACHER_SPLIT:
        raise ValueError("Canonical split changed since the approved data search")
    sealed = {str(p.relative_to(ROOT)): file_sha256(p) for p in [split_path, prediction_path]}
    candidates = source_candidates()
    print(f"Fingerprinting {len(candidates)} external originals.", flush=True)

    def fingerprint(row):
        result = fingerprint_image(resolve_task3_path(row["original_path"], root=ROOT))
        if result["file_sha256"] != row["original_sha256"]:
            raise ValueError("External original hash changed: " + row["external_id"])
        return {**row, **result}

    with ThreadPoolExecutor(max_workers=4) as pool:
        candidates = pd.DataFrame(pool.map(fingerprint, candidates.to_dict("records")))
    candidates, edges = link_external_families(
        candidates, root=ROOT, cache=DATA / "comparison_views"
    )
    edges.to_csv(OUT / "family_edges.csv", index=False)
    references = teacher_references()
    print("Comparing original, 60x80 and foreground views against all teacher roles.", flush=True)
    matches = find_teacher_overlaps(
        candidates, references, root=ROOT, cache=DATA / "comparison_views"
    )
    matches.to_csv(OUT / "teacher_overlap_matches.csv", index=False)
    overlapped_ids = set(matches.loc[matches.accepted_overlap.eq(True), "external_id"])
    affected_groups = set(
        candidates.loc[candidates.external_id.isin(overlapped_ids), "external_group_id"]
    )
    conflicted = set(
        candidates.groupby("external_group_id").source_label.nunique().loc[lambda v: v > 1].index
    )
    candidates["teacher_overlap"] = candidates.external_group_id.isin(affected_groups)
    candidates["admission_reason"] = candidates.selection_reason
    candidates.loc[candidates.teacher_overlap, "admission_reason"] = (
        "Source family connected to a teacher-role image"
    )
    candidates.loc[candidates.external_group_id.isin(conflicted), "admission_reason"] = (
        "Conflicting source labels within linked family"
    )
    eligible = (
        candidates.visual_decision.eq("keep")
        & ~candidates.teacher_overlap
        & ~candidates.external_group_id.isin(conflicted)
    )
    ready = candidates.loc[eligible].copy().sort_values("external_id")
    # Choose one decoded image representative, not repeated exports or ASIN views.
    repeated = ready.duplicated("view_pixel_sha256", keep="first")
    repeat_ids = set(ready.loc[repeated, "external_id"])
    candidates.loc[candidates.external_id.isin(repeat_ids), "admission_reason"] = (
        "Repeated prepared RGB image; keep one deterministic source representative"
    )
    ready = ready.loc[~repeated].copy()
    selected = []
    for label, subset in ready.groupby("source_label", sort=True):
        subset = subset.assign(
            _order=subset.external_id.map(
                lambda value: hashlib.sha256(("2753:" + value).encode()).hexdigest()
            )
        )
        selected.extend(subset.sort_values("_order").head(100).external_id.tolist())
    candidates.loc[
        eligible
        & ~candidates.external_id.isin(selected)
        & ~candidates.external_id.isin(repeat_ids),
        "admission_reason",
    ] = "Per-class intake cap reached"
    ready = ready.loc[ready.external_id.isin(selected)].copy()
    prepared_rows = []
    for row in ready.to_dict("records"):
        destination = DATA / "images_60x80" / row["source"] / (row["source_id"] + ".png")
        info = prepare_external_image(resolve_task3_path(row["original_path"], root=ROOT), destination)
        prepared_rows.append(
            {
                **row,
                **info,
                "path": str(destination.relative_to(ROOT)),
                "training_target": "source_usage",
                "teacher_usage_compatible": False,
                "candidate_usage": row["source_label"],
                "admission_reason": "accepted_for_auxiliary_source_label_training",
            }
        )
    ready = assign_external_partitions(pd.DataFrame(prepared_rows))
    ready = ready.sort_values(
        ["source_label", "external_partition", "external_group_id", "external_id"]
    )
    validate_external_manifest(ready, root=ROOT, check_files=True)
    manifest_path = DATA / "splits.csv"
    ready.to_csv(manifest_path, index=False)
    ready.to_csv(OUT / "splits.csv", index=False)
    for partition in ("train", "validation"):
        ready.loc[ready.external_partition.eq(partition)].to_csv(
            DATA / (partition + ".csv"), index=False
        )
        ready.loc[ready.external_partition.eq(partition)].to_csv(
            OUT / (partition + ".csv"), index=False
        )
    accepted_ids = set(ready.external_id)
    candidates["accepted"] = candidates.external_id.isin(accepted_ids)
    candidates.to_csv(OUT / "all_candidates.csv", index=False)
    candidates.loc[~candidates.accepted].to_csv(OUT / "withheld.csv", index=False)
    train = ready.loc[ready.external_partition.eq("train")]
    stats = fit_external_statistics(train, root=ROOT)
    save_json(DATA / "normalization.json", stats)
    labels = {label: index for index, label in enumerate(sorted(set(ready.source_label)))}
    save_json(DATA / "label_map.json", {"target": "source_usage", "label_to_index": labels})
    # Exercise the actual adapter on every image in each role; this is not a training fit.
    adapter_counts = {}
    for partition in ("train", "validation"):
        dataset = ExternalUsageDataset(
            ready,
            partition=partition,
            label_to_index=labels,
            mean=stats["mean"],
            std=stats["std"],
            root=ROOT,
        )
        for index in range(len(dataset)):
            sample = dataset[index]
            assert sample["image"].shape == (3, 80, 60) and np.isfinite(sample["image"]).all()
        adapter_counts[partition] = len(dataset)
    counts = []
    for label in ("Home", "Party", "Smart Casual", "Travel", "NA"):
        subset = ready.loc[ready.source_label.eq(label)]
        count = {
            "class": label,
            "images": len(subset),
            "families": subset.external_group_id.nunique(),
            "train_images": int(subset.external_partition.eq("train").sum()),
            "validation_images": int(subset.external_partition.eq("validation").sum()),
            "train_families": subset.loc[
                subset.external_partition.eq("train"), "external_group_id"
            ].nunique(),
            "validation_families": subset.loc[
                subset.external_partition.eq("validation"), "external_group_id"
            ].nunique(),
        }
        counts.append(count)
    pd.DataFrame(counts).to_csv(OUT / "class_counts.csv", index=False)
    for path, sha in sealed.items():
        assert file_sha256(resolve_task3_path(path, root=ROOT)) == sha
    summary = {
        "date": "2026-09-06",
        "status": "validated_auxiliary_source_label_intake",
        "training_target": "source_usage",
        "teacher_usage_compatible": False,
        "downloaded_candidate_images": len(candidates),
        "accepted_images": len(ready),
        "accepted_families": int(ready.external_group_id.nunique()),
        "class_counts": counts,
        "teacher_role_counts": references.audit_role.value_counts().to_dict(),
        "teacher_comparison_rows": len(matches),
        "accepted_teacher_overlap_rows": int(matches.accepted_overlap.eq(True).sum()),
        "teacher_connected_source_families_withheld": len(affected_groups),
        "conflicting_source_families_withheld": len(conflicted),
        "withheld_reason_counts": dict(
            Counter(candidates.loc[~candidates.accepted, "admission_reason"])
        ),
        "adapter_checks": adapter_counts,
        "source_split_sha256": file_sha256(manifest_path),
        "teacher_contract_hashes_unchanged": sealed,
        "split_policy": (
            "Deterministic 80/20 whole-family allocation within source/label strata, "
            "seed2753; singleton strata train-only"
        ),
        "image_policy": (
            "EXIF->RGB->LANCZOS aspect-preserving letterbox; PNG width60 height80; "
            "content mask preserves neutral padding after normalization"
        ),
        "normalization_scope": (
            "Only external training content pixels; external validation never fitted"
        ),
        "model_fit_performed": False,
        "limits": (
            "Perceptual comparisons are heuristic, not proof of source independence. "
            "Source labels are not validated teacher-equivalent Usage targets. "
            "Source validation is separate from canonical teacher evaluation."
        ),
    }
    save_json(OUT / "validation.json", summary)
    save_json(DATA / "validation.json", summary)
    save_json(
        OUT / "input_hashes.json",
        {path: file_sha256(OUT / path) for path in INPUTS + ["visual_selection.json"]},
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
