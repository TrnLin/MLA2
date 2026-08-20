"""Read-only audit of raw metadata and image files."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from fashion.config import (
    AUDIT_DIR,
    ROOT,
    TARGET_COLUMNS,
    TEACHER_TRAIN_CSV,
    TEACHER_TRAIN_IMAGE_DIR,
    TEST_CSV,
    TEST_IMAGE_DIR,
)
from fashion.data.hashing import compute_sha256

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_AUDIT_COLUMNS = (
    "role",
    "id",
    "filename",
    "path",
    "file_size_bytes",
    "decode_ok",
    "width",
    "height",
    "aspect_ratio",
    "mode",
    "format",
    "sha256",
    "error",
)
ISSUE_COLUMNS = ("role", "id", "field", "issue_code", "severity", "action", "details")
DUPLICATE_COLUMNS = (
    "duplicate_group",
    "sha256",
    "total_count",
    "train_count",
    "test_count",
    "is_cross_role",
    "member_ids",
    "roles",
)


def _missing_mask(series: pd.Series) -> pd.Series:
    """Identify true CSV blanks without treating literal labels such as ``NA`` as missing."""
    return series.isna() | series.astype(str).str.strip().eq("")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def audit_image(role: str, path: Path, root: Path = ROOT) -> dict[str, Any]:
    """Hash and decode one image candidate without modifying it."""
    try:
        item_id: int | None = int(path.stem)
    except ValueError:
        item_id = None
    record: dict[str, Any] = {
        "role": role,
        "id": item_id,
        "filename": path.name,
        "path": _relative(path, root),
        "file_size_bytes": path.stat().st_size,
        "decode_ok": False,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "mode": None,
        "format": None,
        "sha256": None,
        "error": None,
    }
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        record["error"] = "non_image_extension"
        return record
    try:
        record["sha256"] = compute_sha256(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            record.update(
                {
                    "decode_ok": True,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / max(height, 1), 4),
                    "mode": image.mode,
                    "format": image.format,
                }
            )
    except Exception as error:  # Pillow exposes several format-specific exceptions.
        record["error"] = f"decode_error: {error}"
    return record


def _audit_images(
    train_image_dir: Path,
    prediction_image_dir: Path,
    root: Path,
    workers: int | None,
) -> pd.DataFrame:
    tasks = [
        *(("train", path) for path in sorted(train_image_dir.iterdir())),
        *(("prediction", path) for path in sorted(prediction_image_dir.iterdir())),
    ]
    worker_count = workers or min(32, (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        records = list(executor.map(lambda task: audit_image(*task, root=root), tasks))
    return pd.DataFrame(records, columns=IMAGE_AUDIT_COLUMNS).sort_values(
        ["role", "id", "filename"], na_position="last"
    )


def audit_raw_data(
    train_csv: str | Path = TEACHER_TRAIN_CSV,
    prediction_csv: str | Path = TEST_CSV,
    train_image_dir: str | Path = TEACHER_TRAIN_IMAGE_DIR,
    prediction_image_dir: str | Path = TEST_IMAGE_DIR,
    output_dir: str | Path = AUDIT_DIR,
    root: str | Path = ROOT,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    workers: int | None = None,
) -> dict[str, Path]:
    """Audit raw CSVs and images, then write rebuildable evidence under processed data."""
    train_csv = Path(train_csv)
    prediction_csv = Path(prediction_csv)
    train_image_dir = Path(train_image_dir)
    prediction_image_dir = Path(prediction_image_dir)
    output_dir = Path(output_dir)
    root = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv, keep_default_na=False)
    prediction = pd.read_csv(prediction_csv, keep_default_na=False)
    if "id" not in train or "id" not in prediction:
        raise ValueError("both raw CSV files must contain an id column")

    spill_columns = [column for column in train if column.startswith("Unnamed:")]
    spill_mask = (
        pd.concat([~_missing_mask(train[column]) for column in spill_columns], axis=1).any(axis=1)
        if spill_columns
        else pd.Series(False, index=train.index)
    )
    summary = {
        "styles_train": {
            "path": _relative(train_csv, root),
            "sha256": compute_sha256(train_csv),
            "size_bytes": train_csv.stat().st_size,
            "total_rows": len(train),
            "total_columns": len(train.columns),
            "columns": train.columns.tolist(),
            "unique_ids": int(train["id"].nunique()),
            "duplicate_ids": int(train["id"].duplicated().sum()),
            "spill_row_count": int(spill_mask.sum()),
            "spill_ids": train.loc[spill_mask, "id"].astype(int).tolist(),
            "missing_counts": {
                column: int(_missing_mask(train[column]).sum()) for column in train.columns
            },
        },
        "styles_prediction": {
            "path": _relative(prediction_csv, root),
            "sha256": compute_sha256(prediction_csv),
            "size_bytes": prediction_csv.stat().st_size,
            "total_rows": len(prediction),
            "total_columns": len(prediction.columns),
            "columns": prediction.columns.tolist(),
            "unique_ids": int(prediction["id"].nunique()),
            "duplicate_ids": int(prediction["id"].duplicated().sum()),
            "missing_counts": {
                column: int(_missing_mask(prediction[column]).sum())
                for column in prediction.columns
            },
        },
    }
    summary_path = output_dir / "csv_summary.json"

    missing_rows = [
        {
            "dataset": "styles_train",
            "column": column,
            "null_count": int(_missing_mask(train[column]).sum()),
            "null_percent": float(_missing_mask(train[column]).mean() * 100),
        }
        for column in train.columns
    ]
    missing_path = output_dir / "missing_values.csv"
    pd.DataFrame(missing_rows).to_csv(missing_path, index=False)

    target_rows: list[dict[str, Any]] = []
    for target in targets:
        for label, count in train[target].value_counts(dropna=False).items():
            target_rows.append(
                {
                    "target": target,
                    "class": "<BLANK>"
                    if pd.isna(label) or str(label).strip() == ""
                    else str(label),
                    "count": int(count),
                    "percentage": float(count / len(train) * 100),
                }
            )
    target_counts_path = output_dir / "target_class_counts.csv"
    pd.DataFrame(target_rows).to_csv(target_counts_path, index=False)

    image_audit = _audit_images(train_image_dir, prediction_image_dir, root=root, workers=workers)
    image_audit_path = output_dir / "image_audit.csv"
    image_audit.to_csv(image_audit_path, index=False)

    decoded = image_audit[image_audit["decode_ok"].astype(bool)].copy()
    decoded["id"] = decoded["id"].astype(int)
    issues: list[dict[str, Any]] = []
    train_csv_ids = set(train["id"].astype(int))
    train_image_ids = set(decoded.loc[decoded["role"].eq("train"), "id"])
    prediction_csv_ids = set(prediction["id"].astype(int))
    prediction_image_ids = set(decoded.loc[decoded["role"].eq("prediction"), "id"])
    missing_train_image_ids = sorted(train_csv_ids - train_image_ids)
    image_backed_train = train[train["id"].astype(int).isin(train_image_ids)]
    taxonomy_changes: dict[str, Any] = {}
    for target in targets:
        raw_valid = train.loc[~_missing_mask(train[target]), target].astype(str).str.strip()
        backed_valid = (
            image_backed_train.loc[~_missing_mask(image_backed_train[target]), target]
            .astype(str)
            .str.strip()
        )
        removed_classes = sorted(set(raw_valid) - set(backed_valid))
        taxonomy_changes[target] = {
            "raw_valid_classes": int(raw_valid.nunique()),
            "image_backed_valid_classes": int(backed_valid.nunique()),
            "removed_classes": [
                {
                    "class": label,
                    "raw_count": int(raw_valid.eq(label).sum()),
                    "missing_image_ids": sorted(
                        train.loc[
                            train["id"].astype(int).isin(missing_train_image_ids)
                            & train[target].astype(str).str.strip().eq(label),
                            "id",
                        ].astype(int)
                    ),
                }
                for label in removed_classes
            ],
        }
    summary["styles_train"]["image_reconciliation"] = {
        "image_backed_rows": int(len(image_backed_train)),
        "metadata_rows_without_valid_image": len(missing_train_image_ids),
        "missing_valid_image_ids": missing_train_image_ids,
        "target_taxonomy_changes": taxonomy_changes,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for item_id in missing_train_image_ids:
        issues.append(
            {
                "role": "train",
                "id": item_id,
                "field": "image",
                "issue_code": "missing_image",
                "severity": "error",
                "action": "excluded_from_manifest",
                "details": "metadata row has no valid training image",
            }
        )
    for item_id in sorted(train_image_ids - train_csv_ids):
        issues.append(
            {
                "role": "train",
                "id": item_id,
                "field": "csv_row",
                "issue_code": "extra_image_without_csv_row",
                "severity": "error",
                "action": "excluded_from_manifest",
                "details": "training image has no metadata row",
            }
        )
    for item_id in sorted(prediction_csv_ids - prediction_image_ids):
        issues.append(
            {
                "role": "prediction",
                "id": item_id,
                "field": "image",
                "issue_code": "missing_prediction_image",
                "severity": "critical",
                "action": "flagged",
                "details": "official prediction row has no valid image",
            }
        )
    for row in image_audit.loc[~image_audit["decode_ok"].astype(bool)].itertuples():
        issues.append(
            {
                "role": row.role,
                "id": row.id if pd.notna(row.id) else -1,
                "field": "file",
                "issue_code": "non_image_file"
                if row.error == "non_image_extension"
                else "corrupt_image",
                "severity": "warning",
                "action": "ignored",
                "details": str(row.error),
            }
        )
    for item_id in summary["styles_train"]["spill_ids"]:
        issues.append(
            {
                "role": "train",
                "id": item_id,
                "field": "productDisplayName",
                "issue_code": "csv_spill",
                "severity": "warning",
                "action": "repaired_in_manifest",
                "details": "name segments are joined without changing the raw CSV",
            }
        )
    for target in targets:
        invalid = _missing_mask(train[target])
        for item_id in train.loc[invalid, "id"].astype(int):
            issues.append(
                {
                    "role": "train",
                    "id": item_id,
                    "field": target,
                    "issue_code": "missing_target_label",
                    "severity": "warning",
                    "action": "masked",
                    "details": f"has_{target}_label is false; the target is not imputed",
                }
            )
    issues_frame = pd.DataFrame(issues, columns=ISSUE_COLUMNS)
    if not issues_frame.empty:
        issues_frame.sort_values(["role", "id", "issue_code"], inplace=True)
    issues_path = output_dir / "issues.csv"
    issues_frame.to_csv(issues_path, index=False)

    duplicate_rows: list[dict[str, Any]] = []
    for index, (sha256, group) in enumerate(
        decoded[decoded["sha256"].notna()].groupby("sha256"), start=1
    ):
        if len(group) < 2:
            continue
        roles = set(group["role"])
        duplicate_rows.append(
            {
                "duplicate_group": f"dup_{index:05d}",
                "sha256": sha256,
                "total_count": len(group),
                "train_count": int(group["role"].eq("train").sum()),
                "test_count": int(group["role"].eq("prediction").sum()),
                "is_cross_role": roles == {"train", "prediction"},
                "member_ids": ",".join(map(str, sorted(group["id"].astype(int)))),
                "roles": ",".join(sorted(roles)),
            }
        )
    duplicates_path = output_dir / "exact_duplicate_groups.csv"
    pd.DataFrame(duplicate_rows, columns=DUPLICATE_COLUMNS).to_csv(duplicates_path, index=False)

    return {
        "csv_summary": summary_path,
        "missing_values": missing_path,
        "target_class_counts": target_counts_path,
        "image_audit": image_audit_path,
        "issues": issues_path,
        "exact_duplicate_groups": duplicates_path,
    }
