"""Audit and prepare the user's text-supported rare-usage image collection.

This creates an unassigned collection, not a replacement training split.
"""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import argparse
import hashlib
import importlib.util
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from fashion.data.external_usage import file_sha256, prepare_external_image
from fashion.data.external_usage_audit import (
    find_teacher_overlaps,
    fingerprint_image,
    link_external_families,
)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/external/rare_usage_expansion_20260906"
OLD = ROOT / "reports/task3/rare_external_intake_20260906"
CLASSES = {"home": "Home", "party": "Party", "smart_casual": "Smart Casual", "travel": "Travel"}


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def references():
    spec = importlib.util.spec_from_file_location("old_intake", OLD / "build_intake.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    teacher = module.teacher_references()
    old = read_csv(OLD / "all_candidates.csv")
    admitted = read_csv(ROOT / "data/processed/teacher_plus_rare_usage_20260906/added_images.csv")
    old = old.loc[old.external_id.isin(admitted.external_id)].copy()
    assert len(old) == len(admitted) == 120
    for row in old.to_dict("records"):
        assert file_sha256(resolve_task3_path(row["original_path"], root=ROOT)) == row["file_sha256"]
    old["id"] = old.external_id
    old["path"] = old.original_path
    old["sha256"] = old.file_sha256
    old["audit_role"] = "previously_admitted_external"
    return pd.concat([teacher, old[teacher.columns]], ignore_index=True), admitted


def fingerprint(row):
    source = resolve_task3_path(row["original_path"], root=ROOT)
    assert file_sha256(source) == row["file_sha256"], row["original_path"]
    row["source_original_path"] = row["original_path"]
    row["source_file_sha256"] = row["file_sha256"]
    row["external_id"] = row["source_dataset"] + ":" + row["source_record_id"]
    row["asset_id"] = "rare_" + hashlib.sha256(row["external_id"].encode()).hexdigest()[:20]
    row["alpha_composited_on_white"] = False
    with Image.open(source) as image:
        image.load()
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = ImageOps.exif_transpose(image).convert("RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            white.alpha_composite(rgba)
            source = DATA / "opaque_originals" / (row["asset_id"] + ".png")
            source.parent.mkdir(parents=True, exist_ok=True)
            white.convert("RGB").save(source)
            row["original_path"] = str(source.relative_to(ROOT))
            row["alpha_composited_on_white"] = True
    row.update(fingerprint_image(source))
    row["source"] = row["source_dataset"]
    row["source_id"] = row["source_record_id"]
    row["source_group_id"] = row["source_family_id"]
    row["source_title"] = row["product_name"]
    return row


def audit_class(name):
    source = OUT / name / "candidates.csv"
    frame = read_csv(source)
    assert frame.proposed_usage.eq(CLASSES[name]).all()
    assert frame.evidence_basis.isin({"product_text_inference", "explicit_source_label"}).all()
    assert frame.evidence_text.str.len().gt(10).all()
    assert not frame.source_record_id.duplicated().any()
    with ThreadPoolExecutor(max_workers=4) as pool:
        frame = pd.DataFrame(pool.map(fingerprint, frame.to_dict("records")))
    frame.to_csv(OUT / name / "fingerprints.csv", index=False)
    refs, admitted = references()
    matches = []
    print(f"{name}: compare {len(frame)} candidates with {len(refs)} existing images", flush=True)
    for start in range(0, len(frame), 10):
        matches.append(
            find_teacher_overlaps(
                frame.iloc[start : start + 10],
                refs,
                root=ROOT,
                cache=DATA / "comparison_views" / name,
            )
        )
        print(f"{name}: checked {min(start + 10, len(frame))}/{len(frame)}", flush=True)
    match = pd.concat(matches, ignore_index=True)
    match.to_csv(OUT / name / "existing_image_matches.csv", index=False)
    overlap_ids = set(match.loc[match.accepted_overlap.eq(True), "external_id"])

    def product_key(source, identifier):
        if "amazon" in source.casefold() or "abo" in source.casefold():
            match = re.search(r"B[0-9A-Z]{9}", identifier)
            if match:
                return "amazon:" + match.group()
        return source.casefold() + ":" + identifier

    old_ids = {product_key(row.source_dataset, row.source_id) for row in admitted.itertuples()}
    old_urls = set(admitted.image_url)
    old_hashes = set(admitted.original_sha256)
    frame["existing_image_overlap"] = frame.external_id.isin(overlap_ids)
    source_matches = pd.Series(
        [
            product_key(row.source_dataset, row.source_record_id) in old_ids
            for row in frame.itertuples()
        ],
        index=frame.index,
    )
    frame["previously_admitted_identity"] = (
        source_matches | frame.image_url.isin(old_urls) | frame.source_file_sha256.isin(old_hashes)
    )
    frame.to_csv(OUT / name / "audited.csv", index=False)
    save_json(
        OUT / name / "audit_summary.json",
        {
            "input_sha256": file_sha256(source),
            "references": len(refs),
            "reference_roles": refs.audit_role.value_counts().to_dict(),
            "candidates": len(frame),
            "existing_overlap_images": len(overlap_ids),
            "previous_identity_images": int(frame.previously_admitted_identity.sum()),
            "comparison_rows": len(match),
        },
    )
    print(f"{name}: audit saved; existing overlaps={len(overlap_ids)}", flush=True)


def finalize():
    sealed_paths = [
        ROOT / "data/processed/splits.csv",
        ROOT / "data/processed/prediction_manifest.csv",
        ROOT / "data/processed/teacher_plus_rare_usage_20260906/splits.csv",
    ]
    sealed = {str(p.relative_to(ROOT)): file_sha256(p) for p in sealed_paths}
    frames = []
    for name in CLASSES:
        summary = json.loads((OUT / name / "audit_summary.json").read_text())
        assert summary["input_sha256"] == file_sha256(OUT / name / "candidates.csv")
        frames.append(read_csv(OUT / name / "audited.csv"))
    frame = pd.concat(frames, ignore_index=True).fillna("")
    assert not frame.external_id.duplicated().any()
    frame, edges = link_external_families(
        frame, root=ROOT, cache=DATA / "comparison_views" / "internal"
    )
    edges.to_csv(OUT / "family_edges.csv", index=False)
    affected = set(
        frame.loc[
            frame.existing_image_overlap.eq("True") | frame.previously_admitted_identity.eq("True"),
            "external_group_id",
        ]
    )
    conflicts = set(
        frame.groupby("external_group_id").proposed_usage.nunique().loc[lambda n: n > 1].index
    )
    frame["rejection_reason"] = ""
    frame.loc[frame.external_group_id.isin(affected), "rejection_reason"] = (
        "family_connected_to_existing_image"
    )
    frame.loc[frame.previously_admitted_identity.eq("True"), "rejection_reason"] = (
        "previously_admitted_source_identity"
    )
    frame.loc[frame.external_group_id.isin(conflicts), "rejection_reason"] = (
        "conflicting_usage_within_family"
    )
    near_edges = read_csv(OUT / "internal_near_duplicate_matches.csv")
    parents = {value: value for value in frame.external_id}

    def find(value):
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for pair in near_edges.itertuples():
        first, second = find(pair.first_id), find(pair.second_id)
        parents[max(first, second)] = min(first, second)
    eligible = frame.loc[frame.rejection_reason.eq("")].sort_values("external_id")
    representative = {}
    near_repeats = set()
    for value in eligible.external_id:
        group = find(value)
        if group in representative:
            near_repeats.add(value)
        else:
            representative[group] = value
    frame.loc[frame.external_id.isin(near_repeats), "rejection_reason"] = (
        "duplicate_under_frozen_pixel_and_hash_rule"
    )
    eligible = frame.loc[frame.rejection_reason.eq("")].copy().sort_values("external_id")
    repeats = eligible.duplicated("view_pixel_sha256", keep="first")
    repeat_ids = set(eligible.loc[repeats, "external_id"])
    frame.loc[frame.external_id.isin(repeat_ids), "rejection_reason"] = "duplicate_prepared_pixels"
    ready = frame.loc[frame.rejection_reason.eq("")].copy()
    output = []
    for row in ready.to_dict("records"):
        slug = next(k for k, v in CLASSES.items() if v == row["proposed_usage"])
        destination = DATA / "images_60x80" / slug / (row["asset_id"] + ".png")
        info = prepare_external_image(resolve_task3_path(row["original_path"], root=ROOT), destination)
        with Image.open(destination) as im:
            assert im.size == (60, 80) and im.mode == "RGB"
        assert info["decoded_sha256"] == row["view_pixel_sha256"]
        row.update(info)
        row["path"] = str(destination.relative_to(ROOT))
        row["usage"] = row["proposed_usage"]
        row["partition"] = "unassigned"
        row["label_status"] = "proposed_from_source_text_or_collection"
        is_abo = "amazon" in row["source_dataset"].casefold()
        if is_abo:
            row["rights_basis"] = (
                "Official ABO release declares CC BY 4.0; AWS registry and paper say CC BY-NC "
                "4.0. Both recorded; no claim of unrestricted commercial rights."
            )
            row["rights_status"] = "release_cc_by_4_0_registry_conflict_cc_by_nc_4_0"
            row["attribution"] = "Amazon.com and the Amazon Berkeley Objects dataset contributors"
        else:
            row["rights_basis"] = (
                "Source retailer/photographer copyright; public catalogue access; "
                "no open reuse or redistribution licence established."
            )
            row["rights_status"] = "copyright_permission_not_established"
            row["attribution"] = row["source_dataset"]
        output.append(row)
    ready = pd.DataFrame(output).sort_values(["usage", "source_dataset", "external_id"])
    assert ready.groupby("usage").size().reindex(CLASSES.values(), fill_value=0).ge(100).all(), (
        ready.usage.value_counts()
    )
    assert not ready.decoded_sha256.duplicated().any()
    assert not ready.usage.eq("NA").any()
    ready.to_csv(DATA / "manifest.csv", index=False)
    ready.to_csv(OUT / "accepted_manifest.csv", index=False)
    frame.to_csv(OUT / "all_audited_candidates.csv", index=False)
    frame.loc[frame.rejection_reason.ne("")].to_csv(OUT / "central_rejected.csv", index=False)
    counts = (
        ready.groupby("usage")
        .agg(
            images=("asset_id", "size"),
            source_products=("external_id", "nunique"),
            family_groups=("external_group_id", "nunique"),
        )
        .reset_index()
    )
    counts.to_csv(OUT / "class_counts.csv", index=False)
    for path, digest in sealed.items():
        assert file_sha256(resolve_task3_path(path, root=ROOT)) == digest
    save_json(
        OUT / "summary.json",
        {
            "status": "collected_and_audited_unassigned",
            "accepted_images": len(ready),
            "counts": counts.to_dict("records"),
            "central_rejections": frame.loc[frame.rejection_reason.ne(""), "rejection_reason"]
            .value_counts()
            .to_dict(),
            "alpha_composited_images": int(ready.alpha_composited_on_white.eq("True").sum()),
            "sealed_files": sealed,
            "no_na_collected": True,
            "training_split_changed": False,
            "training_run_started": False,
            "internal_duplicate_edges": len(near_edges),
            "internal_duplicate_images_removed": len(near_repeats),
        },
    )
    print(counts.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[*CLASSES, "finalize"])
    args = parser.parse_args()
    if args.stage == "finalize":
        finalize()
    else:
        audit_class(args.stage)
