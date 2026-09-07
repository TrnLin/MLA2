"""Build a new frozen Usage version and verify all rows through the normal loader."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from fashion.data import FashionDataset, get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.external_usage import file_sha256
from fashion.data.hashing import write_deterministic_csv
from fashion.data.images import load_and_transform_image
from fashion.data.usage_extension import (
    DATASET_NAME,
    extend_usage_dataset,
    validate_usage_extension,
)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REPORT = Path(__file__).resolve().parent
OUT = ROOT / "data/processed" / DATASET_NAME
COLLECTION = ROOT / "data/external/rare_usage_expansion_20260906/manifest.csv"
INTAKE_REPORT = ROOT / "reports/task3/rare_expansion_20260906"
PREVIOUS = ROOT / "data/processed/teacher_plus_rare_usage_20260906/splits.csv"


def read(path):
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def transform(path):
    return load_and_transform_image(path, image_size=(80, 60)).transpose(2, 0, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reviewed = json.loads((INTAKE_REPORT / "summary.json").read_text())
    sealed = dict(reviewed["sealed_files"])
    sealed["data/processed/label_maps.json"] = file_sha256(ROOT / "data/processed/label_maps.json")
    for name, digest in sealed.items():
        if file_sha256(resolve_task3_path(name, root=ROOT)) != digest:
            raise ValueError("a frozen input changed: " + name)
    receipt = json.loads((INTAKE_REPORT / "package_validation.json").read_text())
    bundle = resolve_task3_path(receipt["prepared_archive"], root=ROOT)
    if file_sha256(bundle) != receipt["prepared_archive_sha256"]:
        raise ValueError("the reviewed collection package changed")
    with ZipFile(bundle) as archive:
        if archive.read(str(COLLECTION.relative_to(ROOT))) != COLLECTION.read_bytes():
            raise ValueError("collection manifest differs from its reviewed package")
    collection, previous = read(COLLECTION), read(PREVIOUS)
    links = read(REPORT / "linked_external_families.csv")
    maps = load_label_maps(ROOT / "data/processed/label_maps.json")
    assert len(collection) == 567 and len(previous) == 38_732
    combined = extend_usage_dataset(previous, collection, maps, links, root=ROOT)
    # Keep reviewed source-letterbox geometry with the new dataset for future train-only stats.
    old_geometry = read(ROOT / "data/external/rare_usage_20260906/splits.csv")
    geometry_columns = [
        "external_id",
        "path",
        "sha256",
        "content_left",
        "content_top",
        "content_width",
        "content_height",
    ]
    geometry = pd.concat(
        [old_geometry[geometry_columns], collection[geometry_columns]], ignore_index=True
    )
    geometry = geometry.sort_values("external_id").reset_index(drop=True)
    assert len(geometry) == 687 and not geometry.external_id.duplicated().any()
    lookup = geometry.set_index("external_id")
    for column in geometry_columns[3:]:
        combined[column] = combined.external_id.map(lookup[column]).fillna("")
    validate_usage_extension(combined, previous, collection, links)
    write_deterministic_csv(combined, OUT / "splits.csv", index=False)
    (OUT / "label_maps.json").write_bytes((ROOT / "data/processed/label_maps.json").read_bytes())
    write_deterministic_csv(geometry, OUT / "image_geometry.csv", index=False)
    loaded = load_splits(OUT / "splits.csv")
    added = loaded.loc[loaded.source_dataset.ne("teacher")]
    new = added.loc[added.external_id.isin(collection.external_id)]
    for name, frame in (("added_images", added), ("new_images", new)):
        write_deterministic_csv(frame, OUT / f"{name}.csv", index=False)
    folds, class_folds = [], []
    for fold in range(5):
        train, validation = (
            get_samples(frame, target="usage") for frame in get_cv_split(loaded, fold)
        )
        for column in (
            "id",
            "sha256",
            "duplicate_group",
            "product_name_key",
            "product_family_group",
            "extension_family_group",
        ):
            assert not (set(train[column]) - {""}) & (set(validation[column]) - {""}), column
        directory = OUT / "cv" / f"fold_{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in (("train", train), ("validation", validation)):
            write_deterministic_csv(frame[["id"]], directory / f"{name}_ids.csv", index=False)
            if fold == 0:
                write_deterministic_csv(frame, OUT / f"{name}.csv", index=False)
        row = {"fold": fold, "train": len(train), "validation": len(validation)}
        for name, frame in (("train", train), ("validation", validation)):
            row[f"teacher_{name}"] = int(frame.source_dataset.eq("teacher").sum())
            row[f"previous_added_{name}"] = int(
                frame.source_dataset.ne("teacher").sum()
                - frame.external_id.isin(collection.external_id).sum()
            )
            row[f"new_{name}"] = int(frame.external_id.isin(collection.external_id).sum())
            for label in maps["usage"]["classes"]:
                class_folds.append(
                    {
                        "fold": fold,
                        "side": name,
                        "usage": label,
                        "images": int(frame.usage.eq(label).sum()),
                        "new_images": int(
                            (
                                frame.usage.eq(label)
                                & frame.external_id.isin(collection.external_id)
                            ).sum()
                        ),
                    }
                )
        folds.append(row)
    write_deterministic_csv(pd.DataFrame(folds), OUT / "fold_counts.csv", index=False)
    write_deterministic_csv(pd.DataFrame(class_folds), OUT / "fold_class_counts.csv", index=False)
    teacher = read(ROOT / "data/processed/splits.csv")
    pd.testing.assert_frame_equal(
        combined.loc[combined.source_dataset.eq("teacher"), teacher.columns].reset_index(drop=True),
        teacher,
        check_dtype=False,
    )
    development = get_samples(loaded, partition="development", target="usage")
    class_counts = []
    for label in maps["usage"]["classes"]:
        class_counts.append(
            {
                "usage": label,
                "teacher_development": int(
                    (teacher.partition.eq("development") & teacher.usage.eq(label)).sum()
                ),
                "previous_added": int(
                    (previous.source_dataset.ne("teacher") & previous.usage.eq(label)).sum()
                ),
                "new_added": int(new.usage.eq(label).sum()),
                "combined_development": int(development.usage.eq(label).sum()),
                "new_families": int(
                    new.loc[new.usage.eq(label), "extension_family_group"].nunique()
                ),
            }
        )
    write_deterministic_csv(pd.DataFrame(class_counts), OUT / "class_counts.csv", index=False)
    dataset = FashionDataset(loaded, transform=transform, root=ROOT, targets=("usage",))
    mapping = maps["usage"]["label_to_index"]
    print(f"Checking {len(dataset)} images with the normal RGB 60x80 loader", flush=True)

    def check(index):
        sample = dataset[index]
        image = sample["image"]
        assert image.shape == (3, 80, 60) and image.dtype == np.float32
        assert np.isfinite(image).all() and 0 <= image.min() <= image.max() <= 1
        assert file_sha256(resolve_task3_path(sample["path"], root=ROOT)) == loaded.iloc[index].sha256
        if sample["has_usage_label"]:
            assert sample["usage"] in mapping
        return sample["id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        checked_ids = list(pool.map(check, range(len(dataset))))
    assert len(checked_ids) == len(set(checked_ids)) == 39_299
    fold_zero = get_samples(get_cv_split(loaded, 0)[0], target="usage")
    examples = pd.concat(
        [
            fold_zero.loc[fold_zero.source_dataset.eq("teacher")].head(3),
            fold_zero.loc[
                fold_zero.source_dataset.ne("teacher")
                & ~fold_zero.external_id.isin(collection.external_id)
            ].head(3),
            fold_zero.loc[fold_zero.external_id.isin(collection.external_id)].head(3),
        ]
    )
    mixed = FashionDataset(examples, transform=transform, root=ROOT, targets=("usage",))
    samples = [mixed[i] for i in range(len(mixed))]
    batch = np.stack([s["image"] for s in samples])
    targets = np.array([mapping[s["usage"]] for s in samples], dtype=np.int64)
    assert batch.shape == (9, 3, 80, 60) and targets.shape == (9,)
    for name, digest in sealed.items():
        assert file_sha256(resolve_task3_path(name, root=ROOT)) == digest
    result = {
        "passed": True,
        "dataset": DATASET_NAME,
        "total_rows": len(loaded),
        "teacher_rows": len(teacher),
        "previous_added_rows": 120,
        "new_added_rows": len(new),
        "total_added_rows": len(added),
        "partition_counts": loaded.partition.value_counts().to_dict(),
        "usage_development_rows": len(development),
        "new_class_counts": new.usage.value_counts().to_dict(),
        "all_added_class_counts": added.usage.value_counts().to_dict(),
        "folds": folds,
        "class_count": len(mapping),
        "same_usage_target": True,
        "na_added": False,
        "teacher_rows_ids_and_folds_preserved": True,
        "previous_120_rows_ids_and_folds_preserved": True,
        "protected_targets_blank": True,
        "all_boundary_crossings": 0,
        "new_families": new.extension_family_group.nunique(),
        "all_added_families": added.extension_family_group.nunique(),
        "all_images_loaded_and_hash_checked": len(checked_ids),
        "mixed_batch_shape": list(batch.shape),
        "model_fit_performed": False,
        "statistics_fitted": False,
        "original_inputs_unchanged": sealed,
        "collection_manifest_sha256": file_sha256(COLLECTION),
        "linked_family_manifest_sha256": file_sha256(REPORT / "linked_external_families.csv"),
        "split_sha256": file_sha256(OUT / "splits.csv"),
        "label_map_sha256": file_sha256(OUT / "label_maps.json"),
        "source_geometry_sha256": file_sha256(OUT / "image_geometry.csv"),
    }
    save_json(OUT / "validation.json", result)
    save_json(REPORT / "validation.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
