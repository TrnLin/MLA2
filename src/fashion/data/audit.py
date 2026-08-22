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
from fashion.data.hashing import (
    compute_sha256,
    csv_header_and_id_fingerprint,
    write_deterministic_csv,
)

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
MISSING_VALUE_COLUMNS = (
    "dataset",
    "partition",
    "column",
    "null_count",
    "null_percent",
    "scope",
)
TARGET_COUNT_COLUMNS = ("target", "class", "partition", "count", "percentage", "scope")


def _missing_mask(series: pd.Series) -> pd.Series:
    """Identify true CSV blanks without treating literal labels such as ``NA`` as missing."""
    return series.isna() | series.astype(str).str.strip().eq("")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


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


def refresh_structural_csv_summary(
    train_csv: str | Path,
    prediction_csv: str | Path,
    image_audit_csv: str | Path,
    output_path: str | Path,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Write all-row CSV evidence using headers, IDs, and image facts only."""
    train_csv = Path(train_csv)
    prediction_csv = Path(prediction_csv)
    root = Path(root)
    train_header = pd.read_csv(train_csv, nrows=0, keep_default_na=False).columns.tolist()
    prediction_header = pd.read_csv(prediction_csv, nrows=0, keep_default_na=False).columns.tolist()
    train_ids = pd.read_csv(train_csv, usecols=["id"], keep_default_na=False)["id"].astype(int)
    prediction_ids = pd.read_csv(prediction_csv, usecols=["id"], keep_default_na=False)[
        "id"
    ].astype(int)
    image_audit = pd.read_csv(image_audit_csv, keep_default_na=False)
    decoded = image_audit[_as_bool(image_audit["decode_ok"])].copy()
    decoded["id"] = pd.to_numeric(decoded["id"], errors="raise").astype(int)
    train_image_ids = set(decoded.loc[decoded["role"].eq("train"), "id"])
    prediction_image_ids = set(decoded.loc[decoded["role"].eq("prediction"), "id"])
    train_id_set = set(train_ids)
    prediction_id_set = set(prediction_ids)
    summary = {
        "schema_version": "2.0.0",
        "scope": "all-row structure only; target evidence is development-only below",
        "protected_target_values_hashed": 0,
        "styles_train": {
            "path": _relative(train_csv, root),
            **csv_header_and_id_fingerprint(train_csv),
            "total_columns": len(train_header),
            "columns": train_header,
            "unique_ids": int(train_ids.nunique()),
            "duplicate_ids": int(train_ids.duplicated().sum()),
            "image_reconciliation": {
                "image_backed_rows": len(train_id_set & train_image_ids),
                "metadata_rows_without_valid_image": len(train_id_set - train_image_ids),
                "missing_valid_image_ids": sorted(train_id_set - train_image_ids),
                "extra_valid_image_ids": sorted(train_image_ids - train_id_set),
            },
        },
        "styles_prediction": {
            "path": _relative(prediction_csv, root),
            **csv_header_and_id_fingerprint(prediction_csv),
            "total_columns": len(prediction_header),
            "columns": prediction_header,
            "unique_ids": int(prediction_ids.nunique()),
            "duplicate_ids": int(prediction_ids.duplicated().sum()),
            "image_reconciliation": {
                "image_backed_rows": len(prediction_id_set & prediction_image_ids),
                "metadata_rows_without_valid_image": len(prediction_id_set - prediction_image_ids),
                "missing_valid_image_ids": sorted(prediction_id_set - prediction_image_ids),
                "extra_valid_image_ids": sorted(prediction_image_ids - prediction_id_set),
            },
        },
    }
    output = Path(output_path)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def write_development_target_audit(
    splits_csv: str | Path,
    summary_path: str | Path,
    missing_values_path: str | Path,
    target_counts_path: str | Path,
    issues_path: str | Path,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> None:
    """Add target evidence from train and validation only; protected values stay unused."""
    frame = pd.read_csv(splits_csv, keep_default_na=False)
    development = frame[frame["partition"].isin({"train", "val"})].copy()
    summary_file = Path(summary_path)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    missing_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    missing_issues: list[dict[str, Any]] = []
    audit_summary: dict[str, Any] = {
        "scope": "train_and_validation_only",
        "included_partitions": ["train", "val"],
        "excluded_partitions": ["holdout", "quarantine"],
        "protected_target_values_hashed": 0,
        "rows": int(len(development)),
        "targets": {},
    }
    for target in targets:
        audit_summary["targets"][target] = {}
        mask_column = f"has_{target}_label"
        for partition in ("train", "val"):
            rows = development[development["partition"].eq(partition)]
            valid = _as_bool(rows[mask_column])
            missing = ~valid
            audit_summary["targets"][target][partition] = {
                "rows": int(len(rows)),
                "valid_count": int(valid.sum()),
                "missing_count": int(missing.sum()),
                "class_count": int(rows.loc[valid, target].astype(str).nunique()),
            }
            missing_rows.append(
                {
                    "dataset": "styles_train_image_backed",
                    "partition": partition,
                    "column": target,
                    "null_count": int(missing.sum()),
                    "null_percent": float(missing.mean() * 100) if len(rows) else 0.0,
                    "scope": "development_targets_only",
                }
            )
            counts = rows.loc[valid, target].astype(str).value_counts().sort_index()
            for label, count in counts.items():
                target_rows.append(
                    {
                        "target": target,
                        "class": label,
                        "partition": partition,
                        "count": int(count),
                        "percentage": float(count / max(int(valid.sum()), 1) * 100),
                        "scope": "development_targets_only",
                    }
                )
            for item_id in rows.loc[missing, "id"].astype(int):
                missing_issues.append(
                    {
                        "role": "train",
                        "id": item_id,
                        "field": target,
                        "issue_code": "missing_target_label",
                        "severity": "warning",
                        "action": "masked",
                        "details": (
                            f"{partition} has_{target}_label is false; target is not imputed"
                        ),
                    }
                )
    summary["development_target_audit"] = audit_summary
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(missing_rows, columns=MISSING_VALUE_COLUMNS).to_csv(
        missing_values_path, index=False
    )
    pd.DataFrame(target_rows, columns=TARGET_COUNT_COLUMNS).to_csv(target_counts_path, index=False)
    issues_file = Path(issues_path)
    existing = pd.read_csv(issues_file, keep_default_na=False)
    existing = existing[~existing["issue_code"].eq("missing_target_label")]
    combined = pd.concat(
        [existing, pd.DataFrame(missing_issues, columns=ISSUE_COLUMNS)], ignore_index=True
    )
    combined.sort_values(["role", "id", "issue_code"], inplace=True)
    combined.to_csv(issues_file, index=False)


def audit_raw_data(
    train_csv: str | Path = TEACHER_TRAIN_CSV,
    prediction_csv: str | Path = TEST_CSV,
    train_image_dir: str | Path = TEACHER_TRAIN_IMAGE_DIR,
    prediction_image_dir: str | Path = TEST_IMAGE_DIR,
    output_dir: str | Path = AUDIT_DIR,
    root: str | Path = ROOT,
    workers: int | None = None,
) -> dict[str, Path]:
    """Audit raw structure and images without publishing any target-value aggregate."""
    train_csv = Path(train_csv)
    prediction_csv = Path(prediction_csv)
    train_image_dir = Path(train_image_dir)
    prediction_image_dir = Path(prediction_image_dir)
    output_dir = Path(output_dir)
    root = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_audit = _audit_images(train_image_dir, prediction_image_dir, root=root, workers=workers)
    image_audit_path = output_dir / "image_audit.csv.gz"
    write_deterministic_csv(
        image_audit,
        image_audit_path,
        index=False,
    )
    summary_path = output_dir / "csv_summary.json"
    summary = refresh_structural_csv_summary(
        train_csv, prediction_csv, image_audit_path, summary_path, root
    )
    missing_path = output_dir / "missing_values.csv"
    target_counts_path = output_dir / "target_class_counts.csv"
    pd.DataFrame(columns=MISSING_VALUE_COLUMNS).to_csv(missing_path, index=False)
    pd.DataFrame(columns=TARGET_COUNT_COLUMNS).to_csv(target_counts_path, index=False)

    decoded = image_audit[_as_bool(image_audit["decode_ok"])].copy()
    decoded["id"] = pd.to_numeric(decoded["id"], errors="raise").astype(int)
    train_csv_ids = set(
        pd.read_csv(train_csv, usecols=["id"], keep_default_na=False)["id"].astype(int)
    )
    prediction_csv_ids = set(
        pd.read_csv(prediction_csv, usecols=["id"], keep_default_na=False)["id"].astype(int)
    )
    train_image_ids = set(decoded.loc[decoded["role"].eq("train"), "id"])
    prediction_image_ids = set(decoded.loc[decoded["role"].eq("prediction"), "id"])
    issues: list[dict[str, Any]] = []
    for item_id in summary["styles_train"]["image_reconciliation"]["missing_valid_image_ids"]:
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
    for row in image_audit.loc[~_as_bool(image_audit["decode_ok"])].itertuples():
        issues.append(
            {
                "role": row.role,
                "id": row.id if pd.notna(row.id) else -1,
                "field": "file",
                "issue_code": (
                    "non_image_file" if row.error == "non_image_extension" else "corrupt_image"
                ),
                "severity": "warning",
                "action": "ignored",
                "details": str(row.error),
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
