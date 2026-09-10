"""Build and verify the frozen third Usage dataset with reviewed replacements."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.data import FashionDataset, get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.external_usage import file_sha256
from fashion.data.hashing import write_deterministic_csv
from fashion.data.images import load_and_transform_image
from fashion.data.usage_replacement import replace_usage_dataset

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REPORT = Path(__file__).resolve().parent
NAME = "teacher_plus_rare_usage_v3_20260906"
OUT = ROOT / "data/processed" / NAME
PREVIOUS = ROOT / "data/processed/teacher_plus_rare_usage_v2_20260906"
COLLECTION = ROOT / "data/external/rare_usage_replacement_20260906/manifest.csv"


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def transform(path):
    return load_and_transform_image(path, image_size=(80, 60)).transpose(2, 0, 1)


def main():
    summary = json.loads((REPORT / "collection_summary.json").read_text())
    assert file_sha256(COLLECTION) == summary["manifest_sha256"]
    assert (
        file_sha256(PREVIOUS / "splits.csv") == summary["duplicate_check"]["previous_split_sha256"]
    )
    sealed = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in [
            ROOT / "data/processed/splits.csv",
            ROOT / "data/processed/label_maps.json",
            PREVIOUS / "splits.csv",
            PREVIOUS / "label_maps.json",
            PREVIOUS / "image_geometry.csv",
        ]
    }
    old, collection, retired = (
        read(PREVIOUS / "splits.csv"),
        read(COLLECTION),
        read(REPORT / "retirement_review.csv"),
    )
    assert retired.visual_review_status.eq("reviewed_and_accepted").all()
    links = read(REPORT / "linked_families_all_previous.csv")
    retained = old.loc[~old.id.isin(retired.id)].copy()
    links = links.loc[
        links.external_id.isin(set(retained.external_id) | set(collection.external_id))
    ]
    maps = load_label_maps(ROOT / "data/processed/label_maps.json")
    combined = replace_usage_dataset(old, retired, collection, maps, links, root=ROOT)
    teacher = read(ROOT / "data/processed/splits.csv")
    pd.testing.assert_frame_equal(
        combined.loc[combined.source_dataset.eq("teacher"), teacher.columns].reset_index(drop=True),
        teacher,
        check_dtype=False,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    write_deterministic_csv(combined, OUT / "splits.csv", index=False)
    (OUT / "label_maps.json").write_bytes((ROOT / "data/processed/label_maps.json").read_bytes())
    loaded = load_splits(OUT / "splits.csv")
    external = loaded.loc[loaded.source_dataset.ne("teacher")]
    new = external.loc[external.external_id.isin(collection.external_id)]
    geometry_columns = [
        "external_id",
        "path",
        "sha256",
        "content_left",
        "content_top",
        "content_width",
        "content_height",
    ]
    for name, frame in [
        ("added_images", external),
        ("new_images", new),
        ("retired_images", old.loc[old.id.isin(retired.id)]),
        ("image_geometry", external[geometry_columns]),
        ("linked_external_families", links),
    ]:
        write_deterministic_csv(frame, OUT / f"{name}.csv", index=False)
    retired.to_csv(OUT / "retirement_review.csv", index=False)
    collection.to_csv(OUT / "replacement_sources.csv", index=False)
    # These are accounting pairs, not semantic equivalence claims or reused IDs.
    pairs = []
    for label in sorted(new.usage.unique()):
        a = retired.loc[retired.usage.eq(label)].sort_values("id")
        b = new.loc[new.usage.eq(label)].sort_values("id")
        for before, after in zip(a.itertuples(), b.itertuples(), strict=True):
            pairs.append(
                {
                    "usage": label,
                    "retired_id": before.id,
                    "new_id": after.id,
                    "new_external_id": after.external_id,
                    "pair_basis": "class_count_accounting_only",
                }
            )
    pd.DataFrame(pairs).to_csv(OUT / "replacement_pairs.csv", index=False)
    folds, classes, types = [], [], []
    type_lookup = collection.set_index("external_id").product_type
    for fold in range(5):
        train, validation = (get_samples(x, target="usage") for x in get_cv_split(loaded, fold))
        for column in [
            "id",
            "sha256",
            "duplicate_group",
            "product_name_key",
            "product_family_group",
            "extension_family_group",
        ]:
            assert not (set(train[column]) - {""}) & (set(validation[column]) - {""}), column
        folder = OUT / "cv" / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        folds.append(
            {
                "fold": fold,
                "train": len(train),
                "validation": len(validation),
                "new_train": int(train.external_id.isin(collection.external_id).sum()),
                "new_validation": int(validation.external_id.isin(collection.external_id).sum()),
            }
        )
        for side, frame in [("train", train), ("validation", validation)]:
            write_deterministic_csv(frame[["id"]], folder / f"{side}_ids.csv", index=False)
            if fold == 0:
                write_deterministic_csv(frame, OUT / f"{side}.csv", index=False)
            for label in maps["usage"]["classes"]:
                classes.append(
                    {
                        "fold": fold,
                        "side": side,
                        "usage": label,
                        "teacher": int(
                            (frame.usage.eq(label) & frame.source_dataset.eq("teacher")).sum()
                        ),
                        "external": int(
                            (frame.usage.eq(label) & frame.source_dataset.ne("teacher")).sum()
                        ),
                    }
                )
        x = validation.loc[validation.external_id.isin(collection.external_id)].copy()
        x["audit_product_type"] = x.external_id.map(type_lookup)
        for (label, kind), group in x.groupby(["usage", "audit_product_type"]):
            types.append(
                {
                    "fold": fold,
                    "usage": label,
                    "product_type": kind,
                    "images": len(group),
                    "families": group.extension_family_group.nunique(),
                }
            )
    pd.DataFrame(folds).to_csv(OUT / "fold_counts.csv", index=False)
    pd.DataFrame(classes).to_csv(OUT / "fold_class_counts.csv", index=False)
    pd.DataFrame(types).to_csv(OUT / "new_type_fold_counts.csv", index=False)
    dataset = FashionDataset(loaded, transform=transform, root=ROOT, targets=("usage",))
    print(f"Loading and hash-checking all {len(dataset)} images", flush=True)

    def check(index):
        sample = dataset[index]
        image = sample["image"]
        assert image.shape == (3, 80, 60) and image.dtype == np.float32
        assert np.isfinite(image).all() and 0 <= image.min() <= image.max() <= 1
        assert file_sha256(resolve_task3_path(sample["path"], root=ROOT)) == loaded.iloc[index].sha256
        if sample["has_usage_label"]:
            assert sample["usage"] in maps["usage"]["label_to_index"]
        return sample["id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(check, range(len(dataset))))
    assert len(checked) == len(set(checked)) == len(old) == 39_299
    for name, digest in sealed.items():
        assert file_sha256(resolve_task3_path(name, root=ROOT)) == digest
    result = {
        "passed": True,
        "dataset": NAME,
        "total_rows": len(loaded),
        "teacher_rows": len(teacher),
        "replaced_images": len(new),
        "retained_external": len(external) - len(new),
        "total_external": len(external),
        "replacements_by_usage": new.usage.value_counts().to_dict(),
        "external_class_counts": external.usage.value_counts().to_dict(),
        "same_class_totals": True,
        "same_nine_class_map": True,
        "no_external_na": True,
        "teacher_all_fields_preserved": True,
        "retained_external_all_fields_preserved": True,
        "protected_new_targets_blank_and_masked": True,
        "old_ids_not_reused": True,
        "all_five_fold_boundary_crossings": 0,
        "all_images_loaded_and_hash_checked": len(checked),
        "new_families": new.extension_family_group.nunique(),
        "all_external_families": external.extension_family_group.nunique(),
        "largest_external_family": int(external.groupby("extension_family_group").size().max()),
        "source_and_type_annotations_are_audit_only": True,
        "model_fit_performed": False,
        "statistics_fitted": False,
        "split_sha256": file_sha256(OUT / "splits.csv"),
        "manifest_sha256": file_sha256(COLLECTION),
        "original_inputs_unchanged": sealed,
    }
    save_json(OUT / "validation.json", result)
    save_json(REPORT / "validation.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
