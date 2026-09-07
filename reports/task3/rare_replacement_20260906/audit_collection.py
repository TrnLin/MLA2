"""Prepare and audit the reviewed replacement intake against every existing image."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import argparse
import hashlib
import importlib.util
import json
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
from fashion.data.usage_extension import validate_text_collection

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/external/rare_usage_replacement_20260906"
PREVIOUS = ROOT / "data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv"
INPUTS = (
    "home/selected_candidates.csv",
    "party/selected_candidates.csv",
    "smart_casual/selected_candidates.csv",
    "travel/selected_candidates.csv",
    "camera_travel/selected_candidates.csv",
)


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(row):
    source = resolve_task3_path(row["original_path"], root=ROOT)
    if file_sha256(source) != row["file_sha256"]:
        raise ValueError("Reviewed source bytes changed: " + str(source))
    row["source_original_path"] = row["original_path"]
    row["source_file_sha256"] = row["file_sha256"]
    row["external_id"] = row["source_dataset"] + ":" + row["source_record_id"]
    row["asset_id"] = "replace_" + hashlib.sha256(row["external_id"].encode()).hexdigest()[:20]
    row["alpha_composited_on_white"] = False
    with Image.open(source) as image:
        image.load()
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = ImageOps.exif_transpose(image).convert("RGBA")
            opaque = Image.new("RGBA", rgba.size, "white")
            opaque.alpha_composite(rgba)
            source = DATA / "opaque_originals" / (row["asset_id"] + ".png")
            source.parent.mkdir(parents=True, exist_ok=True)
            opaque.convert("RGB").save(source)
            row["original_path"] = str(source.relative_to(ROOT))
            row["alpha_composited_on_white"] = True
    row.update(fingerprint_image(source))
    row["source"] = row["source_dataset"]
    row["source_id"] = row["source_record_id"]
    row["source_group_id"] = row["source_family_id"]
    row["source_title"] = row["product_name"]
    row["usage"] = row["proposed_usage"]
    row["source_label"] = row["usage"]
    # Link obvious shared case/bottle designs before any fold assignment.
    title = row["product_name"].lower()
    if "bellavita" in row["source_dataset"]:
        if any(name in title for name in ("date woman", "skai aquatic", "g.o.a.t.", "glam woman")):
            row["source_group_id"] = "bellavita:rectangular_bottle_gold_cylinder_cap"
        elif "d.i.v.a." in title or "hot mess" in title:
            row["source_group_id"] = "bellavita:tapered_woman_bottle"
    if "titan" in row["source_dataset"].lower() and any(
        model in title for model in ("95319", "95320")
    ):
        row["source_group_id"] = "titan:raga_cocktail_linked_rectangle_gems"
    if "quattro" in title and row["usage"] == "Smart Casual":
        row["source_group_id"] = "mjbale:quattro_shirt"
    slug = row["usage"].lower().replace(" ", "_")
    destination = DATA / "images_60x80" / slug / (row["asset_id"] + ".png")
    row.update(prepare_external_image(source, destination))
    row["path"] = str(destination.relative_to(ROOT))
    if row["decoded_sha256"] != row["view_pixel_sha256"]:
        raise ValueError("Prepared view differs from the fingerprint")
    row["rights_basis"] = (
        "Retailer/manufacturer product photo; no open redistribution licence established."
    )
    row["attribution"] = row["source_dataset"]
    row["partition"] = "unassigned"
    row["label_status"] = "proposed_from_source_text_or_collection"
    return row


def prepare_all():
    inputs = [OUT / name for name in INPUTS]
    rows = pd.concat([read(path) for path in inputs], ignore_index=True).fillna("")
    rows["selection_input"] = [
        str(path.relative_to(ROOT)) for path in inputs for _ in range(len(read(path)))
    ]
    if rows.duplicated(["source_dataset", "source_record_id"]).any():
        raise ValueError("Repeated selected source identity")
    for column in ("evidence_text", "product_url", "image_url", "source_family_id"):
        if rows[column].str.strip().eq("").any():
            raise ValueError("Missing source evidence: " + column)
    with ThreadPoolExecutor(max_workers=6) as pool:
        frame = pd.DataFrame(pool.map(prepare, rows.to_dict("records")))
    frame.to_csv(OUT / "prepared_candidates.csv", index=False)
    save_json(
        OUT / "selection_receipt.json",
        {
            "selected_images": len(frame),
            "inputs": {str(path.relative_to(ROOT)): file_sha256(path) for path in inputs},
            "prepared_candidates_sha256": file_sha256(OUT / "prepared_candidates.csv"),
        },
    )
    print("Prepared", len(frame), frame.usage.value_counts().to_dict(), flush=True)


def references():
    module_path = ROOT / "reports/task3/rare_external_intake_20260906/build_intake.py"
    spec = importlib.util.spec_from_file_location("original_usage_intake", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    teacher = module.teacher_references()
    previous = read(PREVIOUS)
    old = read(ROOT / "reports/task3/usage_expanded_v2_20260906/linked_external_families.csv")
    old_lookup = previous.loc[previous.source_dataset.ne("teacher")].set_index("external_id")
    if set(old.external_id) != set(old_lookup.index):
        raise ValueError("Old reference membership changed")
    for row in old.itertuples():
        if file_sha256(resolve_task3_path(row.original_path, root=ROOT)) != row.file_sha256:
            raise ValueError("Old source image changed")
        if file_sha256(resolve_task3_path(row.path, root=ROOT)) != old_lookup.loc[row.external_id, "sha256"]:
            raise ValueError("Old prepared image changed")
    old["source_group_id"] = old.external_id.map(old_lookup.extension_family_group)
    old_refs = old.copy()
    old_refs["id"] = old_refs.external_id
    old_refs["path"] = old_refs.original_path
    old_refs["sha256"] = old_refs.file_sha256
    old_refs["audit_role"] = "earlier_external_all_versions"
    return pd.concat([teacher, old_refs[teacher.columns]], ignore_index=True), old, previous


def audit_existing():
    frame = read(OUT / "prepared_candidates.csv")
    refs, _, _ = references()
    print("Compare", len(frame), "candidates with", len(refs), "existing images", flush=True)
    batches = [(i, frame.iloc[i : i + 10]) for i in range(0, len(frame), 10)]

    def compare(batch):
        index, subset = batch
        result = find_teacher_overlaps(
            subset, refs, root=ROOT, cache=DATA / "comparison_views" / f"existing_{index}"
        )
        print("Compared", index, "through", index + len(subset), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        matches = pd.concat(pool.map(compare, batches), ignore_index=True)
    matches.to_csv(OUT / "existing_image_matches.csv", index=False)
    accepted = matches.loc[matches.accepted_overlap.astype(str).str.lower().eq("true")]
    save_json(
        OUT / "existing_audit.json",
        {
            "candidate_sha256": file_sha256(OUT / "prepared_candidates.csv"),
            "previous_split_sha256": file_sha256(PREVIOUS),
            "reference_count": len(refs),
            "reference_roles": refs.audit_role.value_counts().to_dict(),
            "source_and_prepared_reference_bytes_verified": True,
            "overlap_candidate_count": accepted.external_id.nunique(),
            "matches_sha256": file_sha256(OUT / "existing_image_matches.csv"),
        },
    )
    print("Existing overlap candidates:", accepted.external_id.nunique(), flush=True)


def finalize():
    receipt = json.loads((OUT / "existing_audit.json").read_text())
    selection = json.loads((OUT / "selection_receipt.json").read_text())
    for name, digest in selection["inputs"].items():
        if file_sha256(resolve_task3_path(name, root=ROOT)) != digest:
            raise ValueError("Reviewed selection changed after preparation: " + name)
    if selection["prepared_candidates_sha256"] != file_sha256(OUT / "prepared_candidates.csv"):
        raise ValueError("Prepared candidates differ from the selection receipt")
    if receipt["matches_sha256"] != file_sha256(OUT / "existing_image_matches.csv"):
        raise ValueError("Existing overlap decisions changed after audit")
    if receipt["candidate_sha256"] != file_sha256(OUT / "prepared_candidates.csv"):
        raise ValueError("Candidate input changed after overlap audit")
    if receipt["previous_split_sha256"] != file_sha256(PREVIOUS):
        raise ValueError("Previous split changed")
    frame = read(OUT / "prepared_candidates.csv")
    existing = read(OUT / "existing_image_matches.csv")
    overlap = set(existing.loc[existing.accepted_overlap.str.lower().eq("true"), "external_id"])
    frame["rejection_reason"] = ""
    frame.loc[frame.external_id.isin(overlap), "rejection_reason"] = "overlap_with_existing_image"
    frame["existing_image_overlap"] = frame.external_id.isin(overlap)
    frame["previously_admitted_identity"] = False
    # Compare all views of every pair, including source siblings, before grouping.
    refs = frame.copy()
    refs["id"], refs["path"], refs["audit_role"] = (
        refs.external_id,
        refs.original_path,
        "new_candidate",
    )
    internal = find_teacher_overlaps(
        frame, refs, root=ROOT, cache=DATA / "comparison_views/internal"
    )
    internal = internal.loc[internal.external_id.ne(internal.teacher_id)].copy()
    internal.to_csv(OUT / "internal_image_matches.csv", index=False)
    accepted = internal.loc[internal.accepted_overlap.astype(str).str.lower().eq("true")]
    for row in accepted.itertuples():
        a, b = sorted((row.external_id, row.teacher_id))
        labels = frame.set_index("external_id").usage.loc[[a, b]]
        ids = [a, b] if labels.nunique() > 1 else [b]
        frame.loc[
            frame.external_id.isin(ids) & frame.rejection_reason.eq(""), "rejection_reason"
        ] = "internal_near_duplicate_or_conflict"
    _, old, previous = references()
    old_lookup = previous.loc[previous.source_dataset.ne("teacher")].set_index("external_id")
    while True:
        eligible = frame.loc[frame.rejection_reason.eq("")].copy()
        graph_input = pd.concat([old, eligible], ignore_index=True).fillna("")
        linked, edges = link_external_families(
            graph_input, root=ROOT, cache=DATA / "comparison_views/families"
        )
        linked["previous_anchor"] = linked.external_id.map(
            old_lookup.extension_family_group
        ).fillna("")
        linked["reviewed_usage"] = linked.external_id.map(old_lookup.usage).fillna(linked.usage)
        bad = set()
        for group, members in linked.groupby("external_group_id"):
            anchors = set(members.previous_anchor) - {""}
            if len(anchors) > 1 or members.reviewed_usage.nunique() > 1:
                bad.update(set(members.external_id) & set(eligible.external_id))
        if not bad:
            break
        frame.loc[frame.external_id.isin(bad), "rejection_reason"] = (
            "conflicting_frozen_family_anchors_or_labels"
        )
    anchored_names = {}
    for group, members in linked.groupby("external_group_id"):
        anchors = set(members.previous_anchor) - {""}
        if len(anchors) > 1:
            raise ValueError("Existing-only family graph contradicts frozen anchors")
        anchored_names[group] = next(iter(anchors)) if anchors else group
    linked["external_group_id"] = linked.external_group_id.map(anchored_names)
    mapping = linked.set_index("external_id").external_group_id
    frame["external_group_id"] = frame.external_id.map(mapping).fillna("")
    frame.to_csv(OUT / "all_candidate_decisions.csv", index=False)
    eligible = frame.loc[frame.rejection_reason.eq("")].copy()
    validate_text_collection(eligible, root=ROOT)
    DATA.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(DATA / "manifest.csv", index=False)
    eligible.to_csv(OUT / "accepted_manifest.csv", index=False)
    linked.to_csv(OUT / "linked_families_all_previous.csv", index=False)
    edges.to_csv(OUT / "family_edges.csv", index=False)
    summary = {
        "reviewed_candidates": len(frame),
        "accepted_images": len(eligible),
        "accepted_by_usage": eligible.usage.value_counts().to_dict(),
        "accepted_by_type": eligible.groupby(["usage", "product_type"]).size().to_dict(),
        "rejection_counts": frame.loc[frame.rejection_reason.ne(""), "rejection_reason"]
        .value_counts()
        .to_dict(),
        "accepted_families": eligible.external_group_id.nunique(),
        "largest_new_family": int(eligible.groupby("external_group_id").size().max()),
        "duplicate_check": receipt,
        "source_original_files_retained": True,
        "teacher_labels_used_for_overlap": False,
        "manifest_sha256": file_sha256(DATA / "manifest.csv"),
    }
    summary["accepted_by_type"] = [
        {"usage": k[0], "product_type": k[1], "images": int(v)}
        for k, v in summary["accepted_by_type"].items()
    ]
    save_json(OUT / "collection_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "audit", "finalize"))
    stage = parser.parse_args().stage
    {"prepare": prepare_all, "audit": audit_existing, "finalize": finalize}[stage]()
