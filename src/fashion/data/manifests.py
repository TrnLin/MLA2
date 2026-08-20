"""Build image-backed labelled and official-prediction manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fashion.config import (
    AUDIT_DIR,
    LABEL_MAPS_JSON,
    PREDICTION_MANIFEST_CSV,
    TARGET_COLUMNS,
    TEACHER_TRAIN_CSV,
    TEST_CSV,
    TRAIN_MANIFEST_CSV,
)
from fashion.data.metadata import build_label_maps, has_valid_label, repair_product_name

IMAGE_COLUMNS = (
    "id",
    "path",
    "width",
    "height",
    "aspect_ratio",
    "mode",
    "format",
    "file_size_bytes",
    "sha256",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _clean_metadata(train: pd.DataFrame, targets: tuple[str, ...]) -> pd.DataFrame:
    spill_columns = [column for column in train if column.startswith("Unnamed:")]
    records: list[dict[str, Any]] = []
    metadata_columns = (
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "usage",
    )
    for row in train.to_dict("records"):
        segments = [row.get("productDisplayName"), *(row.get(column) for column in spill_columns)]
        has_spill = any(has_valid_label(row.get(column)) for column in spill_columns)
        record: dict[str, Any] = {"id": int(row["id"])}
        for column in metadata_columns:
            value = row.get(column)
            record[column] = str(value).strip() if has_valid_label(value) else ""
        year = row.get("year")
        record["year"] = int(float(str(year).strip())) if has_valid_label(year) else None
        repaired_name = repair_product_name(segments)
        record["productDisplayName"] = repaired_name or None
        record["product_name_repaired"] = has_spill
        for target in targets:
            record[f"has_{target}_label"] = has_valid_label(record[target])
        records.append(record)
    return pd.DataFrame(records)


def build_manifests(
    train_csv: str | Path = TEACHER_TRAIN_CSV,
    prediction_csv: str | Path = TEST_CSV,
    image_audit_csv: str | Path = AUDIT_DIR / "image_audit.csv",
    train_output: str | Path = TRAIN_MANIFEST_CSV,
    prediction_output: str | Path = PREDICTION_MANIFEST_CSV,
    label_maps_output: str | Path = LABEL_MAPS_JSON,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> dict[str, Path]:
    """Repair metadata and join only valid, decoded images."""
    train_output = Path(train_output)
    prediction_output = Path(prediction_output)
    label_maps_output = Path(label_maps_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    label_maps_output.parent.mkdir(parents=True, exist_ok=True)

    raw_train = pd.read_csv(train_csv, keep_default_na=False)
    raw_prediction = pd.read_csv(prediction_csv, keep_default_na=False)
    expected_prediction_columns = ["id", "gender", "articleType", "season", "usage"]
    if raw_prediction.columns.tolist() != expected_prediction_columns:
        raise ValueError(
            "official prediction CSV columns must remain " + ",".join(expected_prediction_columns)
        )
    image_audit = pd.read_csv(image_audit_csv)
    valid = image_audit[_as_bool(image_audit["decode_ok"])].copy()
    valid["id"] = valid["id"].astype(int)

    cleaned = _clean_metadata(raw_train, targets)
    train_images = valid.loc[valid["role"].eq("train"), IMAGE_COLUMNS]
    train_manifest = cleaned.merge(train_images, on="id", how="inner", validate="one_to_one")
    if train_manifest["id"].duplicated().any():
        raise ValueError("labelled manifest IDs must be unique")
    train_manifest.sort_values("id", inplace=True)
    train_manifest.to_csv(train_output, index=False)

    prediction_images = valid.loc[valid["role"].eq("prediction"), IMAGE_COLUMNS]
    prediction_manifest = raw_prediction[["id"]].merge(
        prediction_images, on="id", how="left", sort=False, validate="one_to_one"
    )
    if prediction_manifest["path"].isna().any():
        missing_ids = prediction_manifest.loc[prediction_manifest["path"].isna(), "id"].tolist()
        raise ValueError(f"official prediction images are missing for IDs: {missing_ids[:5]}")
    prediction_manifest.to_csv(prediction_output, index=False)

    maps = build_label_maps(train_manifest, targets)
    label_maps_output.write_text(json.dumps(maps, indent=2), encoding="utf-8")
    return {
        "train_manifest": train_output,
        "prediction_manifest": prediction_output,
        "label_maps": label_maps_output,
    }
