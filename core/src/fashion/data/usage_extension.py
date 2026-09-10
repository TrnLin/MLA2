"""Extend a frozen Usage dataset with reviewed, text-supported product images."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from fashion.config import CV_FOLD_COUNT, RANDOM_SEED, ROOT, TARGET_COLUMNS
from fashion.data.external_usage import decoded_sha256, file_sha256
from fashion.data.families import normalize_product_name
from fashion.data.splits import validate_splits

DATASET_NAME = "teacher_plus_rare_usage_v2_20260906"
USAGE_CLASSES = (
    "Casual",
    "Ethnic",
    "Formal",
    "Home",
    "NA",
    "Party",
    "Smart Casual",
    "Sports",
    "Travel",
)
TEXT_CLASSES = {"Home", "Party", "Smart Casual", "Travel"}
TEXT_EVIDENCE = {"product_text_inference", "explicit_source_label"}


def validate_text_collection(
    collection: pd.DataFrame, *, root: str | Path = ROOT, check_files: bool = True
) -> None:
    """Require the reviewed collection contract; never promote missing labels to NA."""
    required = {
        "external_id",
        "source_dataset",
        "source_record_id",
        "product_name",
        "description",
        "usage",
        "proposed_usage",
        "evidence_text",
        "evidence_basis",
        "product_url",
        "image_url",
        "path",
        "sha256",
        "decoded_sha256",
        "source_file_sha256",
        "source_original_path",
        "external_group_id",
        "partition",
        "label_status",
        "rejection_reason",
        "rights_basis",
        "rights_status",
        "attribution",
        "existing_image_overlap",
        "previously_admitted_identity",
        "content_left",
        "content_top",
        "content_width",
        "content_height",
    }
    if missing := required - set(collection):
        raise ValueError(f"text collection is missing fields: {sorted(missing)}")
    if collection.empty or collection.external_id.duplicated().any():
        raise ValueError("text collection must contain unique external IDs")
    for column in (
        "external_id",
        "source_record_id",
        "product_name",
        "external_group_id",
        "evidence_text",
        "product_url",
        "image_url",
        "rights_basis",
    ):
        if (
            collection[column].isna().any()
            or collection[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"empty source evidence field: {column}")
    if not collection.usage.isin(TEXT_CLASSES).all():
        raise ValueError("text admission permits Home, Party, Smart Casual and Travel only; no NA")
    if not collection.usage.eq(collection.proposed_usage).all():
        raise ValueError("Usage differs from the reviewed proposed label")
    if not collection.evidence_basis.isin(TEXT_EVIDENCE).all():
        raise ValueError("unsupported text evidence basis")
    if not collection.label_status.eq("proposed_from_source_text_or_collection").all():
        raise ValueError("text labels must retain their proposed-source status")
    if not collection.partition.eq("unassigned").all() or collection.rejection_reason.ne("").any():
        raise ValueError("only the accepted, unassigned collection may be appended")
    for column in ("existing_image_overlap", "previously_admitted_identity"):
        if not collection[column].astype(str).str.lower().isin({"false", "0"}).all():
            raise ValueError(f"unresolved existing-image check: {column}")
    if collection.sha256.duplicated().any() or collection.decoded_sha256.duplicated().any():
        raise ValueError("repeated prepared image in text collection")
    if collection.groupby("external_group_id").usage.nunique().gt(1).any():
        raise ValueError("a reviewed family has conflicting Usage labels")
    geometry = collection[["content_left", "content_top", "content_width", "content_height"]].apply(
        pd.to_numeric, errors="raise"
    )
    if geometry.isna().any().any() or not geometry.mod(1).eq(0).all().all():
        raise ValueError("image content geometry must contain finite integers")
    left, top, width, height = (geometry[column] for column in geometry)
    if (
        (left < 0)
        | (top < 0)
        | (width < 1)
        | (height < 1)
        | (left + width > 60)
        | (top + height > 80)
    ).any():
        raise ValueError("image content geometry lies outside 60 x 80")
    if check_files:
        root = Path(root).resolve()
        for row in collection.itertuples():
            path = (root / row.path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("prepared image path escapes the project")
            if file_sha256(path) != row.sha256:
                raise ValueError("prepared image hash changed: " + row.path)
            with Image.open(path) as image:
                image.load()
                if image.size != (60, 80) or image.mode != "RGB":
                    raise ValueError("prepared images must be RGB, width 60 and height 80")
                if decoded_sha256(image) != row.decoded_sha256:
                    raise ValueError("prepared pixel hash changed: " + row.path)


def assign_extension_folds(
    previous: pd.DataFrame,
    collection: pd.DataFrame,
    family_links: pd.DataFrame,
    *,
    seed: int = RANDOM_SEED,
) -> pd.Series:
    """Keep old folds fixed and balance whole new families against their class counts."""
    previous_external = previous.loc[previous.source_dataset.ne("teacher")]
    expected_ids = set(previous_external.external_id) | set(collection.external_id)
    if family_links.external_id.duplicated().any() or set(family_links.external_id) != expected_ids:
        raise ValueError("family links must cover the previous and new external images exactly")
    if family_links.external_group_id.isna().any() or family_links.external_group_id.eq("").any():
        raise ValueError("linked family IDs must be present")
    links = family_links.set_index("external_id").external_group_id
    sources = pd.concat(
        [
            previous_external[["external_id", "usage", "cv_fold"]],
            collection[["external_id", "usage"]],
        ],
        ignore_index=True,
    )
    sources["family"] = sources.external_id.map(links)
    if sources.groupby("family").usage.nunique().gt(1).any():
        raise ValueError("linked source family has conflicting Usage labels")
    old_families = sources.loc[sources.cv_fold.notna() & sources.cv_fold.astype(str).ne("")]
    if old_families.groupby("family").cv_fold.nunique().gt(1).any():
        raise ValueError("a linked family spans previously frozen folds")
    locked = old_families.groupby("family").cv_fold.first().astype(int).to_dict()
    new = collection.assign(family=collection.external_id.map(links))
    # The reviewed graph is atomic even if a caller supplies a finer link table.
    if new.groupby("external_group_id").family.nunique().gt(1).any():
        raise ValueError("family links split an already reviewed source family")
    if (
        previous_external.assign(family=previous_external.external_id.map(links))
        .groupby("source_group_id")
        .family.nunique()
        > 1
    ).any():
        raise ValueError("family links split an existing source family")
    groups = new.groupby("family").agg(usage=("usage", "first"), size=("external_id", "size"))
    development = previous.loc[previous.partition.eq("development")]
    totals = [
        int(pd.to_numeric(development.cv_fold).eq(fold).sum()) for fold in range(CV_FOLD_COUNT)
    ]
    counts = {
        label: [
            int((development.usage.eq(label) & pd.to_numeric(development.cv_fold).eq(fold)).sum())
            for fold in range(CV_FOLD_COUNT)
        ]
        for label in TEXT_CLASSES
    }
    assignments = {}

    def tie(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    def place(group, fold):
        row = groups.loc[group]
        size = int(row["size"])
        assignments[group] = fold
        totals[fold] += size
        counts[row.usage][fold] += size

    for group in sorted(set(groups.index) & set(locked)):
        place(group, locked[group])
    for label, labelled in groups.groupby("usage", sort=True):
        ordered = sorted(
            set(labelled.index) - set(assignments),
            key=lambda group: (-int(groups.loc[group, "size"]), tie(group)),
        )
        for group in ordered:
            fold = min(
                range(CV_FOLD_COUNT),
                key=lambda f: (counts[label][f], totals[f], tie(f"{label}:{f}")),
            )
            place(group, fold)
    return new.family.map(assignments).astype(int)


def extend_usage_dataset(
    previous: pd.DataFrame,
    collection: pd.DataFrame,
    label_maps: dict[str, dict[str, Any]],
    family_links: pd.DataFrame,
    *,
    root: str | Path = ROOT,
    check_files: bool = True,
) -> pd.DataFrame:
    """Append direct Usage rows without changing any previous row, ID or fold."""
    validate_splits(previous)
    validate_text_collection(collection, root=root, check_files=check_files)
    mapping = label_maps["usage"]
    if mapping["label_column"] != "usage" or tuple(mapping["classes"]) != USAGE_CLASSES:
        raise ValueError("the existing nine-class Usage map must be preserved")
    if mapping["label_to_index"] != {label: index for index, label in enumerate(USAGE_CLASSES)}:
        raise ValueError("the Usage class indices changed")
    if set(collection.external_id) & set(previous.external_id):
        raise ValueError("an external ID was already admitted")
    if set(collection.sha256) & set(previous.sha256):
        raise ValueError("an added image duplicates an existing prepared image")
    names = collection.product_name.map(normalize_product_name)
    if set(names) & (set(previous.product_name_key.astype(str)) - {""}):
        raise ValueError("resolve existing product-name identity before adding images")
    added = collection.sort_values("external_id").reset_index(drop=True)
    folds = assign_extension_folds(previous, added, family_links)
    first_id = max(1_000_000_000, int(previous.id.astype(int).max()) + 1)
    links = family_links.set_index("external_id").external_group_id
    old_families = previous.loc[
        previous.source_dataset.ne("teacher"), ["external_id", "product_family_group"]
    ].copy()
    old_families["linked"] = old_families.external_id.map(links)
    anchors = (
        old_families.sort_values("product_family_group")
        .groupby("linked")
        .product_family_group.first()
        .to_dict()
    )
    extra_columns = [
        "extension_family_group",
        "product_description",
        "evidence_basis",
        "label_status",
        "rights_status",
        "attribution",
        "source_original_path",
        "content_left",
        "content_top",
        "content_width",
        "content_height",
    ]
    base = previous.copy()
    for column in extra_columns:
        if column not in base:
            base[column] = ""
    base["extension_family_group"] = base.external_id.map(links).fillna("")
    records = []
    for index, row in added.iterrows():
        group = str(links.loc[row.external_id])
        record = dict.fromkeys(base.columns, "")
        record.update(
            id=str(first_id + index),
            usage=str(row.usage),
            productDisplayName=str(row.product_name),
            product_name_repaired=False,
            product_description=str(row.description),
            path=str(row.path),
            width=60,
            height=80,
            aspect_ratio=0.75,
            mode="RGB",
            format="PNG",
            file_size_bytes=(Path(root) / row.path).stat().st_size if check_files else 0,
            sha256=str(row.sha256),
            product_name_key=normalize_product_name(str(row.product_name)),
            product_family_group=anchors.get(group, "external:" + group),
            family_group_basis="source_identity_and_image_audit_v2",
            duplicate_group="external:" + str(row.decoded_sha256),
            is_cross_role_exact_duplicate=False,
            is_cross_role_near_duplicate=False,
            is_cross_role_duplicate=False,
            has_conflicting_target_labels=False,
            partition="development",
            cv_fold=str(int(folds.iloc[index])),
            source_dataset=str(row.source_dataset),
            source_id=str(row.source_record_id),
            external_id=str(row.external_id),
            source_label=str(row.usage),
            label_mapping_basis=(
                "product_text_inference_to_same_named_usage"
                if row.evidence_basis == "product_text_inference"
                else "source_collection_or_tag_to_same_named_usage"
            ),
            source_url=str(row.product_url),
            image_url=str(row.image_url),
            label_evidence=str(row.evidence_text),
            label_strength=str(row.evidence_basis),
            evidence_basis=str(row.evidence_basis),
            label_status=str(row.label_status),
            source_group_id=group,
            extension_family_group=group,
            rights_basis=str(row.rights_basis),
            rights_status=str(row.rights_status),
            attribution=str(row.attribution),
            original_sha256=str(row.source_file_sha256),
            source_original_path=str(row.source_original_path),
        )
        for column in ("content_left", "content_top", "content_width", "content_height"):
            record[column] = int(row[column])
        for target in TARGET_COLUMNS:
            record[f"has_{target}_label"] = target == "usage"
        records.append(record)
    combined = pd.concat([base, pd.DataFrame(records, columns=base.columns)], ignore_index=True)
    validate_usage_extension(combined, previous, collection, family_links)
    return combined


def validate_usage_extension(combined, previous, collection, family_links):
    """Prove frozen rows, source membership, labels and every linked-family boundary."""
    validate_splits(combined)
    retained = combined.loc[combined.id.astype(str).isin(previous.id.astype(str)), previous.columns]

    def order(frame):
        return frame.sort_values("id", key=lambda s: s.astype(int)).reset_index(drop=True)

    pd.testing.assert_frame_equal(order(retained), order(previous), check_dtype=False)
    added = combined.loc[~combined.id.astype(str).isin(previous.id.astype(str))]
    if len(added) != len(collection) or set(added.external_id) != set(collection.external_id):
        raise ValueError("new dataset must contain each new image exactly once")
    if not added.partition.eq("development").all():
        raise ValueError("new images must only enter the development folds")
    expected = collection.set_index("external_id").usage
    if not added.usage.eq(added.external_id.map(expected)).all():
        raise ValueError("added Usage labels differ from the reviewed collection")
    for target in TARGET_COLUMNS:
        flags = added[f"has_{target}_label"].astype(str).str.lower().isin({"true", "1"})
        if not flags.eq(target == "usage").all():
            raise ValueError(f"incorrect target mask: {target}")
        if target != "usage" and added[target].astype(str).ne("").any():
            raise ValueError(f"new images have no established {target} labels")
    external = combined.loc[combined.source_dataset.ne("teacher")]
    groups = family_links.set_index("external_id").external_group_id
    if not external.extension_family_group.eq(external.external_id.map(groups)).all():
        raise ValueError("combined source-family links changed")
    if external.groupby("extension_family_group").cv_fold.nunique().gt(1).any():
        raise ValueError("a linked external family crosses train and validation folds")
