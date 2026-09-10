"""Development-only data diagnostics and local-source filename overlap.

No external label file, protected image, or protected target is accessed.
The external pixel preview only opens already present versions of preselected
development IDs in folds 1/2/3. This does not admit them to Task 3 training.
"""

from fashion.task3_paths import resolve_task3_path

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyse_evidence import CLASSES, EXPERIMENTS, OUT, ROOT, csv, save_json
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

FIGURES = ROOT / "results/figures/task3/usage_investigation"


def pixel_record(row):
    path = resolve_task3_path(row["path"], root=ROOT)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert source_hash == row["sha256"]
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        decoded = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        thumb = rgb.resize((60, 80), Image.Resampling.BILINEAR)
        thumbnail_hash = hashlib.sha256(thumb.tobytes()).hexdigest()
    return {
        "id": row["id"],
        "source_hash_verified": True,
        "decoded_rgb_sha256": decoded,
        "resized_60x80_sha256": thumbnail_hash,
    }


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    dev = csv("data/processed/splits.csv")
    # Drop all labels of other partitions before constructing any statistics.
    roles = dev[["id", "partition"]].copy()
    dev = dev.loc[dev.partition.eq("development")].copy()
    dev["cv_fold"] = dev.cv_fold.astype(int)
    with ThreadPoolExecutor(max_workers=6) as pool:
        pixels = pd.DataFrame(pool.map(pixel_record, dev.to_dict("records")))
    pixels.to_csv(OUT / "development_pixel_hashes.csv.gz", index=False, compression="gzip")
    hashes = dev.merge(pixels, on="id", validate="one_to_one")
    duplicates = []
    for column in ["decoded_rgb_sha256", "resized_60x80_sha256"]:
        grouping = hashes.groupby(column)
        sizes = grouping.size()
        repeated = hashes[column].isin(sizes[sizes.gt(1)].index)
        mixed = grouping.usage.nunique().gt(1)
        duplicates.append(
            {
                "hash_kind": column,
                "duplicate_groups": int(sizes.gt(1).sum()),
                "duplicate_rows": int(repeated.sum()),
                "mixed_usage_groups": int(mixed.sum()),
                "mixed_usage_rows": int(hashes[column].isin(mixed[mixed].index).sum()),
                "fold_crossings": int(grouping.cv_fold.nunique().gt(1).sum()),
            }
        )
    save_json("pixel_duplicate_audit.json", {"verified_images": len(hashes), "hashes": duplicates})

    external = ROOT / "data/raw/external/fashion_product_images_v1/images"
    external_files = {int(p.stem): p for p in external.glob("*.jpg") if p.stem.isdigit()}
    prediction = pd.read_csv(
        ROOT / "data/processed/prediction_manifest.csv", usecols=["id"], keep_default_na=False
    )
    known = set(roles.id) | set(prediction.id)
    overlap = {
        "external_image_files": len(external_files),
        "canonical_roles": {
            str(role): len(set(rows.id) & set(external_files))
            for role, rows in roles.groupby("partition")
        },
        "teacher_prediction_id_overlap": len(set(prediction.id) & set(external_files)),
        "ids_outside_all_teacher_roles": len(set(external_files) - known),
        "outside_teacher_ids": sorted(set(external_files) - known),
        "external_labels_read": False,
        "claim_limit": "filename ID overlap only, not a full cross-source pixel/family audit",
    }
    save_json("external_local_id_overlap.json", overlap)

    predictions = {}
    for label in ["E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]:
        frame = csv(EXPERIMENTS[label] + "/oof_predictions.csv").set_index("id")
        predictions[label] = frame.predicted_index
    valid = dev.loc[dev.has_usage_label.astype(str).str.lower().eq("true")].set_index("id")
    truth = valid.usage.map(dict(zip(CLASSES, range(9))))
    correctness = pd.DataFrame(
        {model: values.loc[valid.index].eq(truth) for model, values in predictions.items()}
    )
    valid["models_correct"] = correctness.sum(axis=1)
    valid["e2_prediction"] = predictions["E2"].loc[valid.index].map(dict(enumerate(CLASSES)))
    pd.crosstab(valid.usage, valid.models_correct).to_csv(OUT / "persistent_error_counts.csv")
    persistent = valid.loc[valid.models_correct.eq(0)]
    persistent.groupby(["usage", "articleType"]).size().rename("errors").sort_values(
        ascending=False
    ).to_csv(OUT / "persistent_error_article_types.csv")
    # Different independently saved family IDs per category, no human selection.
    picks = []
    shared = valid.loc[valid.cv_fold.isin([1, 2, 3])]
    for label in ["Sports", "Formal", "Ethnic", "NA", "Party", "Smart Casual"]:
        group = shared.loc[shared.usage.eq(label)].sort_values(
            ["models_correct", "product_family_group"]
        )
        group = group.drop_duplicates("product_family_group").head(4)
        picks.extend(group.index)
    rows = shared.loc[picks].reset_index()
    rows[
        [
            "id",
            "cv_fold",
            "path",
            "usage",
            "articleType",
            "product_family_group",
            "models_correct",
            "e2_prediction",
        ]
    ].to_csv(OUT / "image_case_scope.csv", index=False)
    fig, axes = plt.subplots(6, 4, figsize=(12, 16), layout="constrained")
    for ax, row in zip(axes.ravel(), rows.itertuples(), strict=True):
        with Image.open(resolve_task3_path(row.path, root=ROOT)) as image:
            ax.imshow(image.convert("RGB"), interpolation="nearest")
        ax.set_title(
            f"{row.id} | {row.usage}\n{row.articleType}\n"
            f"E2: {row.e2_prediction}; right: {row.models_correct}/8",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle(
        "Automatically selected development cases | folds 1, 2, 3\n"
        "Original labels retained; wrong predictions do not prove wrong labels",
        fontsize=14,
    )
    fig.savefig(FIGURES / "persistent_cases.png", dpi=150)
    plt.close(fig)

    # One fixed, existing labelled product per class for resolution inspection.
    selected = rows.groupby("usage", sort=False).head(1)
    fig, axes = plt.subplots(len(selected), 2, figsize=(9, 18), layout="constrained")
    resolution = []
    for (left, right), row in zip(axes, selected.itertuples(), strict=True):
        with Image.open(resolve_task3_path(row.path, root=ROOT)) as image:
            teacher = image.convert("RGB")
        left.imshow(teacher, interpolation="nearest")
        left.set_title(f"{row.id} {row.usage}: teacher {teacher.width}x{teacher.height}")
        ext = external_files.get(row.id)
        if ext is None:
            right.text(0.5, 0.5, "No same-ID local source", ha="center")
            resolution.append({"id": row.id, "usage": row.usage, "same_id_available": False})
        else:
            with Image.open(ext) as image:
                high = ImageOps.exif_transpose(image).convert("RGB")
            right.imshow(high)
            right.set_title(f"Same-ID local source {high.width}x{high.height}")
            low = high.resize(teacher.size, Image.Resampling.BILINEAR)
            similarity = structural_similarity(
                np.asarray(teacher), np.asarray(low), channel_axis=2, data_range=255
            )
            resolution.append(
                {
                    "id": row.id,
                    "usage": row.usage,
                    "same_id_available": True,
                    "external_width": high.width,
                    "external_height": high.height,
                    "bilinear_downsample_ssim": similarity,
                    "external_image_sha256": hashlib.sha256(ext.read_bytes()).hexdigest(),
                }
            )
        left.axis("off")
        right.axis("off")
    fig.suptitle(
        "Extra pixels of the same product are not extra independent products\n"
        "Training-ID preview only; no external labels used; no new training",
        fontsize=14,
    )
    fig.savefig(FIGURES / "resolution_examples.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(resolution).to_csv(OUT / "resolution_preview_audit.csv", index=False)
    print(json.dumps(overlap, indent=2))
    print(json.dumps(duplicates, indent=2))
    print(pd.crosstab(valid.usage, valid.models_correct).to_string())


if __name__ == "__main__":
    main()
