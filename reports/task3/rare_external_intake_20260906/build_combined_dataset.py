"""Build and verify teacher + rare images using the existing nine-class Usage task."""

from __future__ import annotations

from fashion.task3_paths import resolve_task3_path

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.data import FashionDataset, get_cv_split, get_samples, load_label_maps, load_splits
from fashion.data.expanded_usage import DATASET_NAME, combine_teacher_and_rare_usage
from fashion.data.external_usage import file_sha256
from fashion.data.hashing import write_deterministic_csv
from fashion.data.images import load_and_transform_image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REPORT = Path(__file__).resolve().parent
OUT = ROOT / "data/processed" / DATASET_NAME
SOURCE = ROOT / "data/external/rare_usage_20260906"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def transformed(path):
    return load_and_transform_image(path, image_size=(80, 60)).transpose(2, 0, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_evidence = json.loads((SOURCE / "validation.json").read_text())
    frozen = dict(source_evidence["teacher_contract_hashes_unchanged"])
    frozen["data/processed/label_maps.json"] = file_sha256(ROOT / "data/processed/label_maps.json")
    for path, expected in frozen.items():
        if file_sha256(resolve_task3_path(path, root=ROOT)) != expected:
            raise ValueError(f"The teacher input changed since the image audit: {path}")
    if file_sha256(SOURCE / "splits.csv") != source_evidence["source_split_sha256"]:
        raise ValueError("The accepted source manifest changed since its image audit")
    if source_evidence["accepted_teacher_overlap_rows"] != 0:
        raise ValueError("Resolve teacher-connected source images before building")

    teacher = pd.read_csv(ROOT / "data/processed/splits.csv", keep_default_na=False, dtype=str)
    admitted = pd.read_csv(SOURCE / "splits.csv", keep_default_na=False)
    label_maps = load_label_maps(ROOT / "data/processed/label_maps.json")
    combined = combine_teacher_and_rare_usage(teacher, admitted, label_maps, root=ROOT)
    write_deterministic_csv(combined, OUT / "splits.csv", index=False)
    (OUT / "label_maps.json").write_bytes((ROOT / "data/processed/label_maps.json").read_bytes())
    loaded = load_splits(OUT / "splits.csv")
    added = loaded.loc[loaded.source_dataset.ne("teacher")]
    write_deterministic_csv(added, OUT / "added_images.csv", index=False)
    folds = []
    for fold in range(5):
        training, validation = get_cv_split(loaded, fold)
        training = get_samples(training, target="usage")
        validation = get_samples(validation, target="usage")
        assert not set(training.id).intersection(validation.id)
        assert not set(training.product_family_group).intersection(validation.product_family_group)
        folder = OUT / "cv" / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in (("train", training), ("validation", validation)):
            write_deterministic_csv(frame[["id"]], folder / f"{name}_ids.csv", index=False)
            if fold == 0:
                write_deterministic_csv(frame, OUT / f"{name}.csv", index=False)
        folds.append(
            {
                "fold": fold,
                "train": len(training),
                "validation": len(validation),
                "teacher_train": int(training.source_dataset.eq("teacher").sum()),
                "added_train": int(training.source_dataset.ne("teacher").sum()),
                "teacher_validation": int(validation.source_dataset.eq("teacher").sum()),
                "added_validation": int(validation.source_dataset.ne("teacher").sum()),
                "added_validation_classes": validation.loc[
                    validation.source_dataset.ne("teacher"), "usage"
                ]
                .value_counts()
                .to_dict(),
            }
        )

    # Exercise the normal project dataset. Every image is transformed identically;
    # no source-only head, label map, standardization or dataset adapter is used.
    dataset = FashionDataset(loaded, transform=transformed, root=ROOT, targets=("usage",))
    lookup = label_maps["usage"]["label_to_index"]
    print(f"Verifying {len(dataset)} images through FashionDataset at RGB 60x80.", flush=True)

    def check(index):
        sample = dataset[index]
        array = sample["image"]
        assert array.shape == (3, 80, 60)
        assert array.dtype == np.float32 and np.isfinite(array).all()
        assert 0 <= array.min() <= array.max() <= 1
        assert file_sha256(resolve_task3_path(sample["path"], root=ROOT)) == str(loaded.iloc[index].sha256)
        if sample["has_usage_label"]:
            assert sample["usage"] in lookup
        return sample["id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        seen = list(pool.map(check, range(len(dataset))))
    assert len(seen) == len(set(seen)) == len(loaded)
    training, _ = get_cv_split(loaded, 0)
    training = get_samples(training, target="usage")
    teacher_example = training.loc[training.source_dataset.eq("teacher")].head(4)
    added_example = training.loc[training.source_dataset.ne("teacher")].head(4)
    mixed = FashionDataset(
        pd.concat([teacher_example, added_example]),
        transform=transformed,
        root=ROOT,
        targets=("usage",),
    )
    samples = [mixed[index] for index in range(len(mixed))]
    batch_images = np.stack([sample["image"] for sample in samples])
    batch_targets = np.array([lookup[sample["usage"]] for sample in samples], dtype=np.int64)
    assert batch_images.shape == (8, 3, 80, 60) and batch_targets.shape == (8,)
    assert all(file_sha256(resolve_task3_path(path, root=ROOT)) == expected for path, expected in frozen.items())

    before = teacher.loc[teacher.partition.eq("development"), "usage"].value_counts()
    after = get_samples(loaded, partition="development", target="usage").usage.value_counts()
    class_rows = [
        {
            "usage": name,
            "teacher_development": int(before.get(name, 0)),
            "added": int(added.usage.eq(name).sum()),
            "combined_development": int(after.get(name, 0)),
        }
        for name in label_maps["usage"]["classes"]
    ]
    write_deterministic_csv(pd.DataFrame(class_rows), OUT / "class_counts.csv", index=False)
    result = {
        "passed": True,
        "dataset": DATASET_NAME,
        "training_target": "usage",
        "class_count": 9,
        "separate_model_or_head": False,
        "teacher_rows": len(teacher),
        "added_rows": len(added),
        "total_rows": len(loaded),
        "partition_counts": dict(Counter(loaded.partition)),
        "usage_development_rows": int(sum(after)),
        "added_class_counts": added.usage.value_counts().to_dict(),
        "added_families": int(added.product_family_group.nunique()),
        "folds": folds,
        "all_images_loaded_and_hash_checked": len(seen),
        "mixed_batch_shape": list(batch_images.shape),
        "mixed_target_shape": list(batch_targets.shape),
        "image_transform": "RGB, width 60 x height 80, float32 in [0, 1], CHW",
        "normalization": (
            "Range scaling only; optional standardization must fit each combined training fold"
        ),
        "teacher_rows_and_folds_preserved": True,
        "protected_labels_stay_blank": True,
        "group_crossings": 0,
        "source_overlap_audit": "reports/task3/rare_external_intake_20260906/validation.json",
        "source_split_sha256": source_evidence["source_split_sha256"],
        "teacher_inputs_unchanged": frozen,
        "combined_split_sha256": file_sha256(OUT / "splits.csv"),
        "label_map_sha256": file_sha256(OUT / "label_maps.json"),
        "model_fit_performed": False,
        "limits": (
            "Seller labels are mapped explicitly by name. Keep source provenance for error "
            "analysis; Smart Casual adds only two product families and Travel adds one."
        ),
    }
    save_json(OUT / "validation.json", result)
    save_json(REPORT / "combined_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
