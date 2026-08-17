"""Shared CSV-audit, hierarchy, and image-hash helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class CsvAudit:
    """Record the physical CSV schema and repaired metadata quality signals."""

    row_count: int
    physical_columns: tuple[str, ...]
    phantom_columns: tuple[str, ...]
    phantom_nonempty_counts: dict[str, int]
    duplicate_ids: tuple[str, ...]
    blank_counts: dict[str, int]
    literal_na_usage_count: int


# Existing callers named this audit after its schema purpose.
SchemaAudit = CsvAudit


def audit_csv(path: Path) -> tuple[pd.DataFrame, CsvAudit]:
    """Read fashion metadata without treating its literal ``NA`` label as missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Training metadata not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Metadata CSV is empty: {path}") from error
        maximum_width = max((len(row) for row in reader), default=len(raw_header))

    header = [
        column if column else f"Unnamed: {index}"
        for index, column in enumerate(raw_header)
    ]
    overflow = [
        f"Overflow: {index}"
        for index in range(len(header), maximum_width)
    ]
    frame = pd.read_csv(
        path,
        header=None,
        names=[*header, *overflow],
        skiprows=1,
        keep_default_na=False,
        dtype={"id": "string"},
        engine="python",
    )
    required = {
        "id", "gender", "masterCategory", "subCategory", "articleType",
        "baseColour", "season", "year", "usage", "productDisplayName",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    phantom = tuple(
        column
        for column in frame.columns
        if column.startswith(("Unnamed:", "Overflow:"))
    )
    nonempty_phantoms = {
        column: int(
            frame[column].fillna("").astype("string").str.strip().ne("").sum()
        )
        for column in phantom
    }
    display_columns = ["productDisplayName", *phantom]
    frame["productDisplayName"] = frame[display_columns].apply(
        lambda row: ",".join(
            str(value) for value in row if pd.notna(value) and str(value).strip()
        ),
        axis=1,
    )

    duplicate_ids = tuple(
        sorted(frame.loc[frame["id"].duplicated(keep=False), "id"].astype(str).unique())
    )
    named = frame.drop(columns=list(phantom))
    id_text = named["id"].astype("string")
    invalid_ids = ~id_text.str.fullmatch(r"[+-]?\d+")
    if invalid_ids.any():
        invalid_values = ", ".join(
            repr(value) for value in id_text.loc[invalid_ids].unique()[:5]
        )
        raise ValueError(
            f"Metadata contains invalid integer ID values: {invalid_values}"
        )
    named["id"] = id_text.astype("int64")
    blank_counts = {
        column: int(named[column].astype("string").str.strip().eq("").sum())
        for column in named.columns
    }
    audit = CsvAudit(
        row_count=len(frame),
        physical_columns=tuple(frame.columns),
        phantom_columns=phantom,
        phantom_nonempty_counts=nonempty_phantoms,
        duplicate_ids=duplicate_ids,
        blank_counts=blank_counts,
        literal_na_usage_count=int(named["usage"].eq("NA").sum()),
    )
    return named, audit


def _display_labels(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("<BLANK>").replace("", "<BLANK>")


def hierarchy_conflicts(frame: pd.DataFrame) -> pd.DataFrame:
    """Return article types mapped to multiple hierarchy categories and their IDs."""
    grouped = (
        frame.groupby("articleType", dropna=False)
        .agg(
            master_categories=(
                "masterCategory",
                lambda values: tuple(sorted(set(_display_labels(values)))),
            ),
            subcategories=(
                "subCategory",
                lambda values: tuple(sorted(set(_display_labels(values)))),
            ),
            rows=("id", "size"),
            ids=("id", lambda values: tuple(sorted(set(map(str, values))))),
        )
        .reset_index()
    )
    grouped["master_category_count"] = grouped["master_categories"].str.len()
    grouped["subcategory_count"] = grouped["subcategories"].str.len()
    return grouped.loc[
        grouped["master_category_count"].gt(1) | grouped["subcategory_count"].gt(1)
    ].sort_values("articleType", ignore_index=True)


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Return a fixed-width hexadecimal difference hash for an image."""
    if hash_size < 1:
        raise ValueError("hash_size must be at least 1")
    grayscale = image.convert("L").resize(
        (hash_size + 1, hash_size),
        Image.Resampling.LANCZOS,
    )
    pixels = np.asarray(grayscale, dtype=np.uint8)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


# Dataset comparison keeps this temporary private import during the migration.
_dhash = dhash


def hamming_distance(left: str, right: str) -> int:
    """Count distinct bits between hexadecimal dHash values."""
    return (int(left, 16) ^ int(right, 16)).bit_count()
