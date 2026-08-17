"""Metadata analysis for the official, image-paired fashion population."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fashion.config import (
    EDA_OUTPUT_DIR,
    EDA_SAMPLE_SIZE,
    ORIGINAL_CSV,
    ORIGINAL_IMAGE_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_COLUMNS,
    TEACHER_TRAIN_CSV,
    TEACHER_TRAIN_IMAGE_DIR,
    TEST_CSV,
)
from fashion.data_audit import CsvAudit, audit_csv, hierarchy_conflicts
from fashion.eda_images import (
    IMAGE_METRICS,
    MEASUREMENT_COLUMNS,
    exact_duplicate_groups,
    measure_images,
    near_duplicate_candidates,
    paired_image_comparison,
    stratified_sample,
)
from fashion.eda_plots import (
    build_report_summary,
    plot_article_type_support,
    plot_association_matrix,
    plot_drift,
    plot_duplicate_summary,
    plot_image_profiles,
    plot_relationship_heatmap,
    plot_review_grid,
    plot_target_distributions,
)


@dataclass(frozen=True)
class EdaPaths:
    """Locations needed to construct the selected EDA population."""

    teacher_train_csv: Path = TEACHER_TRAIN_CSV
    original_csv: Path = ORIGINAL_CSV
    test_csv: Path = TEST_CSV
    original_image_dir: Path = ORIGINAL_IMAGE_DIR
    lowres_image_dir: Path = TEACHER_TRAIN_IMAGE_DIR


@dataclass(frozen=True)
class PopulationAudit:
    """Immutable reconciliation of metadata and image identifiers."""

    source_train_ids: int
    usable_products: int
    quarantined_test_ids: tuple[int, ...]
    teacher_duplicate_ids: tuple[int, ...]
    test_duplicate_ids: tuple[int, ...]
    original_duplicate_ids: tuple[int, ...]
    missing_original_metadata_ids: tuple[int, ...]
    missing_original_image_ids: tuple[int, ...]
    missing_lowres_image_ids: tuple[int, ...]
    missing_both_image_ids: tuple[int, ...]
    unmatched_original_image_ids: tuple[int, ...]
    unmatched_lowres_image_ids: tuple[int, ...]
    invalid_original_image_filenames: tuple[str, ...]
    invalid_lowres_image_filenames: tuple[str, ...]
    duplicate_original_image_ids: tuple[int, ...]
    duplicate_lowres_image_ids: tuple[int, ...]
    teacher_csv_audit: CsvAudit
    original_csv_audit: CsvAudit


@dataclass(frozen=True)
class EdaResult:
    """Immutable locations and cache decisions from one EDA run."""

    output_dir: Path
    summary_path: Path
    manifest: tuple[str, ...]
    cache_status: Mapping[str, str]


def _integer_ids(values: pd.Series, source: str) -> pd.Series:
    text = values.astype("string")
    invalid = ~text.str.fullmatch(r"[+-]?\d+")
    if invalid.any():
        examples = ", ".join(repr(value) for value in text[invalid].unique()[:5])
        raise ValueError(f"{source} contains invalid integer ID values: {examples}")
    return text.astype("int64")


def _read_test_ids(path: Path) -> tuple[pd.Series, tuple[int, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"Test metadata not found: {path}")
    frame = pd.read_csv(
        path,
        usecols=["id"],
        keep_default_na=False,
        dtype={"id": "string"},
    )
    if "id" not in frame:
        raise ValueError(f"Test metadata is missing required column: id ({path})")
    ids = _integer_ids(frame["id"], "Test metadata")
    duplicates = tuple(sorted(ids.loc[ids.duplicated(keep=False)].unique().tolist()))
    return ids, duplicates


def _image_inventory(
    directory: Path,
) -> tuple[dict[int, Path], tuple[str, ...], tuple[int, ...]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    paths: dict[int, Path] = {}
    invalid: list[str] = []
    duplicate_ids: set[int] = set()
    for path in sorted((item for item in directory.iterdir() if item.is_file()), key=lambda item: item.name):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            product_id = int(path.stem)
        except ValueError:
            invalid.append(path.name)
            continue
        if product_id in paths:
            duplicate_ids.add(product_id)
        else:
            paths[product_id] = path
    return paths, tuple(invalid), tuple(sorted(duplicate_ids))


def build_population(paths: EdaPaths = EdaPaths()) -> tuple[pd.DataFrame, PopulationAudit]:
    """Return sorted products with original labels and both required image views."""
    teacher, teacher_csv_audit = audit_csv(paths.teacher_train_csv)
    original, original_csv_audit = audit_csv(paths.original_csv)
    test_ids, test_duplicates = _read_test_ids(paths.test_csv)

    train_ids = pd.Index(teacher["id"].drop_duplicates().sort_values())
    quarantined_ids = pd.Index(test_ids.drop_duplicates().sort_values())
    overlap = train_ids.intersection(quarantined_ids)
    if not overlap.empty:
        values = ", ".join(map(str, overlap.tolist()[:10]))
        raise ValueError(
            "IDs appear in both teacher training and test metadata: "
            f"{values}"
        )

    selected = original.loc[original["id"].isin(train_ids)].copy()
    original_duplicates = tuple(
        sorted(selected.loc[selected["id"].duplicated(keep=False), "id"].unique().tolist())
    )
    selected = selected.drop_duplicates("id", keep="first")
    selected_ids = pd.Index(selected["id"])
    missing_metadata = tuple(sorted(train_ids.difference(selected_ids).tolist()))

    original_images, invalid_original_images, duplicate_original_images = _image_inventory(
        paths.original_image_dir
    )
    lowres_images, invalid_lowres_images, duplicate_lowres_images = _image_inventory(
        paths.lowres_image_dir
    )
    missing_original = tuple(sorted(selected_ids.difference(original_images).tolist()))
    missing_lowres = tuple(sorted(selected_ids.difference(lowres_images).tolist()))
    missing_both = tuple(
        sorted(set(missing_original).intersection(missing_lowres))
    )
    unmatched_original = tuple(sorted(set(original_images).difference(selected_ids)))
    unmatched_lowres = tuple(sorted(set(lowres_images).difference(selected_ids)))

    selected["original_image_path"] = selected["id"].map(original_images)
    selected["lowres_image_path"] = selected["id"].map(lowres_images)
    population = (
        selected.dropna(subset=["original_image_path", "lowres_image_path"])
        .sort_values("id", ignore_index=True)
    )
    audit = PopulationAudit(
        source_train_ids=len(train_ids),
        usable_products=len(population),
        quarantined_test_ids=tuple(quarantined_ids.tolist()),
        teacher_duplicate_ids=tuple(map(int, teacher_csv_audit.duplicate_ids)),
        test_duplicate_ids=test_duplicates,
        original_duplicate_ids=original_duplicates,
        missing_original_metadata_ids=missing_metadata,
        missing_original_image_ids=missing_original,
        missing_lowres_image_ids=missing_lowres,
        missing_both_image_ids=missing_both,
        unmatched_original_image_ids=unmatched_original,
        unmatched_lowres_image_ids=unmatched_lowres,
        invalid_original_image_filenames=invalid_original_images,
        invalid_lowres_image_filenames=invalid_lowres_images,
        duplicate_original_image_ids=duplicate_original_images,
        duplicate_lowres_image_ids=duplicate_lowres_images,
        teacher_csv_audit=teacher_csv_audit,
        original_csv_audit=original_csv_audit,
    )
    return population, audit


def _labels(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("")


def distribution_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return complete category counts and shares, preserving blank and literal NA."""
    if column not in frame:
        raise ValueError(f"Column not found: {column}")
    if frame.empty:
        return pd.DataFrame(
            {
                "label": pd.Series(dtype="string"),
                "count": pd.Series(dtype="int64"),
                "share": pd.Series(dtype="float64"),
                "is_blank": pd.Series(dtype="bool"),
            }
        )
    counts = _labels(frame[column]).value_counts(dropna=False)
    result = counts.rename_axis("label").reset_index(name="count")
    result["label"] = result["label"].astype(str)
    result = result.sort_values(
        ["count", "label"], ascending=[False, True], ignore_index=True
    )
    result["share"] = result["count"] / len(frame)
    result["is_blank"] = result["label"].eq("")
    return result


def skew_table(
    frame: pd.DataFrame, columns: tuple[str, ...] = TARGET_COLUMNS
) -> pd.DataFrame:
    """Summarize target imbalance using descriptive, finite class metrics."""
    records: list[dict[str, float | int | str]] = []
    for column in columns:
        distribution = distribution_table(frame, column)
        learned = distribution.loc[~distribution["is_blank"], "count"]
        counts = learned.to_numpy(dtype=float)
        total = int(counts.sum())
        classes = len(counts)
        probabilities = counts / total if total else np.array([], dtype=float)
        entropy = float(-(probabilities * np.log2(probabilities)).sum()) if total else 0.0
        normalized_entropy = entropy / np.log2(classes) if classes > 1 else 0.0
        records.append(
            {
                "column": column,
                "total": total,
                "classes": classes,
                "blank_count": int(
                    distribution.loc[distribution["is_blank"], "count"].sum()
                ),
                "literal_na_count": int(
                    distribution.loc[distribution["label"].eq("NA"), "count"].sum()
                ),
                "majority_share": float(probabilities.max()) if total else 0.0,
                "imbalance_ratio": float(counts.max() / counts.min()) if total else 0.0,
                "normalized_entropy": float(normalized_entropy),
                "effective_class_count": float(2**entropy) if total else 0.0,
                "gini_impurity": float(1 - np.square(probabilities).sum()) if total else 0.0,
                "top_1_share": float(probabilities[:1].sum()) if total else 0.0,
                "top_5_share": float(probabilities[:5].sum()) if total else 0.0,
            }
        )
    return pd.DataFrame.from_records(records).set_index("column")


def support_band_table(
    frame: pd.DataFrame, column: str = "articleType"
) -> pd.DataFrame:
    """Count label classes by fixed, report-ready training-support bands."""
    distribution = distribution_table(frame, column)
    counts = distribution.loc[~distribution["is_blank"], "count"]
    bands = (("1", 1, 1), ("2", 2, 2), ("3–4", 3, 4), ("5–9", 5, 9), ("10+", 10, None))
    total = int(counts.sum())
    records = []
    for name, lower, upper in bands:
        mask = counts.ge(lower) if upper is None else counts.between(lower, upper)
        records.append(
            {
                "band": name,
                "class_count": int(mask.sum()),
                "product_count": int(counts[mask].sum()),
                "product_share": float(counts[mask].sum() / total) if total else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    """Return bias-corrected Cramér's V for two categorical series."""
    paired = pd.DataFrame({"left": _labels(left), "right": _labels(right)}).dropna()
    if paired.empty or len(paired) <= 1:
        return 0.0
    table = pd.crosstab(paired["left"], paired["right"])
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    valid = expected > 0
    chi_square = float((np.square(observed - expected)[valid] / expected[valid]).sum())
    rows, columns = observed.shape
    phi_squared = chi_square / total
    correction = ((columns - 1) * (rows - 1)) / (total - 1)
    corrected_phi = max(0.0, phi_squared - correction)
    corrected_rows = rows - ((rows - 1) ** 2 / (total - 1))
    corrected_columns = columns - ((columns - 1) ** 2 / (total - 1))
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return float(np.sqrt(corrected_phi / denominator)) if denominator > 0 else 0.0


def association_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Return a symmetric, bias-corrected Cramér's V matrix."""
    matrix = pd.DataFrame(0.0, index=columns, columns=columns)
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            value = cramers_v(frame[left], frame[right])
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
    if not frame.empty:
        for column in columns:
            matrix.loc[column, column] = cramers_v(frame[column], frame[column])
    return matrix


def row_normalized_cooccurrence(
    frame: pd.DataFrame, row_column: str, column: str
) -> pd.DataFrame:
    """Return conditional category shares for each row category."""
    table = pd.crosstab(_labels(frame[row_column]), _labels(frame[column]))
    return table.div(table.sum(axis=1), axis=0).fillna(0.0)


def deterministic_id_bins(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Add deterministic equal-width catalogue-ID range labels."""
    if bins < 1:
        raise ValueError("bins must be at least 1")
    result = frame.copy()
    if result.empty:
        result["id_bin"] = pd.Series(dtype="string")
        return result
    ids = _integer_ids(result["id"], "Population")
    minimum, maximum = int(ids.min()), int(ids.max())
    width = max(1, int(np.ceil((maximum - minimum + 1) / bins)))
    bin_index = ((ids - minimum) // width).clip(upper=bins - 1)
    lower = minimum + bin_index * width
    upper = np.minimum(lower + width - 1, maximum)
    result["id_bin"] = lower.astype(str) + "–" + pd.Series(upper, index=result.index).astype(str)
    return result


def drift_table(
    frame: pd.DataFrame, group_column: str, category_column: str
) -> pd.DataFrame:
    """Measure each group distribution's total variation from all products."""
    if frame.empty:
        return pd.DataFrame(
            {
                group_column: pd.Series(dtype="string"),
                "total": pd.Series(dtype="int64"),
                "total_variation": pd.Series(dtype="float64"),
            }
        )
    categories = _labels(frame[category_column])
    overall = categories.value_counts(normalize=True)
    grouped = pd.DataFrame(
        {"group": _labels(frame[group_column]), "category": categories}
    )
    records: list[dict[str, object]] = []
    for group, values in grouped.groupby("group", sort=True):
        shares = values["category"].value_counts(normalize=True)
        variation = 0.5 * sum(
            abs(float(shares.get(category, 0.0)) - float(overall[category]))
            for category in overall.index
        )
        records.append(
            {group_column: str(group), "total": len(values), "total_variation": variation}
        )
    return pd.DataFrame.from_records(records)


def product_name_audit(frame: pd.DataFrame) -> dict[str, object]:
    """Return deterministic basic evidence from product names, not language inference."""
    names = _labels(frame["productDisplayName"])
    words = Counter(
        word
        for name in names
        for word in re.findall(r"[a-z0-9]+", name.lower())
    )
    common_words = [
        {"word": word, "count": count}
        for word, count in sorted(words.items(), key=lambda item: (-item[1], item[0]))[:20]
    ]
    candidates: list[dict[str, object]] = []
    gender_tokens = (
        (re.compile(r"\b(?:men|male)\b"), "Men"),
        (re.compile(r"\b(?:women|female)\b"), "Women"),
        (re.compile(r"\b(?:boy|boys)\b"), "Boys"),
        (re.compile(r"\b(?:girl|girls)\b"), "Girls"),
    )
    for row in frame.loc[:, ["id", "gender", "productDisplayName"]].itertuples(index=False):
        name = str(row.productDisplayName).lower()
        gender = str(row.gender)
        expected_labels = {
            expected_label
            for pattern, expected_label in gender_tokens
            if pattern.search(name)
        }
        if expected_labels and gender not in expected_labels:
            candidates.append(
                {
                    "id": int(row.id),
                    "gender": gender,
                    "productDisplayName": str(row.productDisplayName),
                }
            )
    lengths = names.str.len()
    hierarchy_columns = {"id", "articleType", "masterCategory", "subCategory"}
    return {
        "total_products": len(frame),
        "missing_name_count": int(names.eq("").sum()),
        "name_length": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "max": int(lengths.max()) if len(lengths) else 0,
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
        },
        "common_words": common_words,
        "gender_contradiction_candidates": sorted(candidates, key=lambda item: item["id"]),
        "hierarchy_conflicts": (
            hierarchy_conflicts(frame).to_dict("records")
            if hierarchy_columns.issubset(frame.columns)
            else []
        ),
    }


METRIC_VERSION = "eda-image-metrics-v1"
MAX_REVIEW_EXAMPLES = 24
EDA_REVIEW_GRIDS = (
    "review-common.png",
    "review-rare.png",
    "review-unusual.png",
    "review-grayscale.png",
    "review-exact-duplicate.png",
    "review-near-duplicate.png",
)
GENERATED_EDA_OUTPUTS = frozenset(
    {
        "summary.json",
        "data-audit.json",
        "product-name-audit.json",
        "target-distributions.csv",
        "metadata-distributions.csv",
        "target-skew.csv",
        "article-type-support.csv",
        "associations.csv",
        "gender-usage.csv",
        "strongest-relationship.csv",
        "year-drift.csv",
        "id-range-drift.csv",
        "hierarchy-conflicts.csv",
        "lowres-measurements.csv",
        "highres-measurements.csv",
        "near-lowres-measurements.csv",
        "paired-image-comparison.csv",
        "exact-duplicates.csv",
        "near-duplicates.csv",
        "lowres-measurements.provenance.json",
        "highres-measurements.provenance.json",
        "near-lowres-measurements.provenance.json",
        "target-distributions.png",
        "article-type-support.png",
        "associations.png",
        "gender-usage.png",
        "strongest-relationship.png",
        "year-drift.png",
        "id-range-drift.png",
        "image-profiles.png",
        "duplicate-summary.png",
        "eda-report-summary.png",
        *EDA_REVIEW_GRIDS,
    }
)
_CACHE_OUTPUTS = frozenset(
    {
        "lowres-measurements.csv",
        "highres-measurements.csv",
        "near-lowres-measurements.csv",
        "lowres-measurements.provenance.json",
        "highres-measurements.provenance.json",
        "near-lowres-measurements.provenance.json",
    }
)
METADATA_COLUMNS = (
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
    "productDisplayName",
)
USEFUL_CATEGORICAL_COLUMNS = (
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "usage",
)


def _directory_provenance(
    frame: pd.DataFrame, path_column: str, seed: int, sample_limit: int
) -> dict[str, object]:
    selected = frame.loc[:, ["id", path_column]].copy()
    selected["id"] = selected["id"].astype(int)
    selected["path"] = selected[path_column].map(lambda path: str(Path(path)))
    selected = selected.sort_values(["id", "path"], kind="stable")
    inventory: list[dict[str, object]] = []
    for path_text in selected["path"].drop_duplicates():
        path = Path(path_text)
        try:
            stats = path.stat()
            inventory.append(
                {
                    "path": path_text,
                    "size": stats.st_size,
                    "mtime_ns": stats.st_mtime_ns,
                    "ctime_ns": stats.st_ctime_ns,
                    "content_sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
        except OSError:
            inventory.append({"path": path_text, "missing": True})
    inventory = sorted(inventory, key=lambda item: str(item["path"]))
    selected_rows = [
        {"id": int(row.id), "path": row.path}
        for row in selected.loc[:, ["id", "path"]].itertuples(index=False)
    ]
    inventory_hash = sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    selected_hash = sha256(
        json.dumps(selected_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    input_directory = str(Path(selected_rows[0]["path"]).parent) if selected_rows else ""
    return {
        "metric_version": METRIC_VERSION,
        "input_directory": input_directory,
        "file_count": len(inventory),
        "file_inventory": inventory,
        "file_inventory_hash": inventory_hash,
        "seed": int(seed),
        "sample_limit": int(sample_limit),
        "selected_id_path_hash": selected_hash,
    }


def _json_safe(value: object) -> object:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


_CACHE_STRING_COLUMNS = (
    "path",
    "mode",
    "file_sha256",
    "byte_sha256",
    "pixel_sha256",
    "dhash",
    "error",
)


def _valid_cached_measurements(
    cached: pd.DataFrame,
    selected: pd.DataFrame,
    path_column: str,
) -> pd.DataFrame | None:
    required = {"id", *MEASUREMENT_COLUMNS}
    if not required.issubset(cached.columns):
        return None
    try:
        cached = cached.copy()
        cached["id"] = _integer_ids(cached["id"], "Cached measurements")
        expected_rows = sorted(
            (int(row.id), str(getattr(row, path_column)))
            for row in selected.loc[:, ["id", path_column]].itertuples(index=False)
        )
        actual_rows = sorted(
            (int(row.id), str(row.path))
            for row in cached.loc[:, ["id", "path"]].itertuples(index=False)
        )
    except (TypeError, ValueError):
        return None
    if actual_rows != expected_rows:
        return None

    for column, width in (
        ("file_sha256", 64),
        ("byte_sha256", 64),
        ("pixel_sha256", 64),
        ("dhash", 16),
    ):
        values = cached[column].dropna()
        if not values.astype("string").str.fullmatch(
            rf"[0-9a-fA-F]{{{width}}}"
        ).all():
            return None

    numeric_columns = (
        "width",
        "height",
        "aspect_ratio",
        "file_bytes",
        *IMAGE_METRICS,
    )
    for column in numeric_columns:
        numeric = pd.to_numeric(cached[column], errors="coerce")
        if (cached[column].notna() & numeric.isna()).any():
            return None
        cached[column] = numeric
    return cached


def _read_cache(
    csv_path: Path,
    provenance_path: Path,
    expected: dict[str, object],
    refresh: bool,
    selected: pd.DataFrame,
    path_column: str,
) -> pd.DataFrame | None:
    if refresh or not csv_path.is_file() or not provenance_path.is_file():
        return None
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if provenance != _json_safe(expected):
        return None
    try:
        cached = pd.read_csv(
            csv_path,
            dtype={column: "string" for column in _CACHE_STRING_COLUMNS},
        )
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    return _valid_cached_measurements(cached, selected, path_column)


def _measure_with_cache(
    frame: pd.DataFrame,
    path_column: str,
    cache_stem: str,
    output_dir: Path,
    *,
    seed: int,
    sample_limit: int,
    refresh: bool,
) -> tuple[pd.DataFrame, dict[str, object], str]:
    selected = frame.loc[:, ["id", path_column]].copy()
    expected = _directory_provenance(selected, path_column, seed, sample_limit)
    csv_path = output_dir / f"{cache_stem}-measurements.csv"
    provenance_path = output_dir / f"{cache_stem}-measurements.provenance.json"
    cached = _read_cache(
        csv_path,
        provenance_path,
        expected,
        refresh,
        selected,
        path_column,
    )
    if cached is not None:
        return cached, expected, "reused"
    measured = measure_images(selected, path_column)
    measured.to_csv(csv_path, index=False)
    _write_json(provenance_path, expected)
    return measured, expected, "recomputed"


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _table_records(table: pd.DataFrame) -> list[dict[str, object]]:
    return _json_safe(table.to_dict("records"))  # type: ignore[return-value]


def _split_provenance() -> dict[str, object]:
    path = PROCESSED_DATA_DIR / "splits.csv"
    if not path.is_file():
        return {"status": "absent", "path": str(path)}
    return {
        "status": "present",
        "path": str(path),
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _open_examples(
    frame: pd.DataFrame, ids: list[int], title_column: str = "articleType"
) -> list[tuple[int, object, str]]:
    from PIL import Image

    indexed = frame.set_index("id")
    examples: list[tuple[int, object, str]] = []
    for product_id in ids:
        if product_id not in indexed.index:
            continue
        row = indexed.loc[product_id]
        try:
            with Image.open(row["lowres_image_path"]) as image:
                examples.append((int(product_id), image.convert("RGB").copy(), str(row[title_column])))
        except OSError:
            continue
    return examples


def _unusual_review_ids(lowres: pd.DataFrame) -> list[int]:
    """Select deterministic measurement extremes and uncommon image modes."""
    columns = ("width", "height", *IMAGE_METRICS)
    candidates: list[int] = []
    for column in columns:
        if column not in lowres:
            continue
        values = lowres.loc[:, ["id", column]].copy()
        values[column] = pd.to_numeric(values[column], errors="coerce")
        values = values.dropna().sort_values([column, "id"], kind="stable")
        if not values.empty:
            candidates.extend((int(values.iloc[0]["id"]), int(values.iloc[-1]["id"])))
    if "mode" in lowres:
        modes = lowres["mode"].dropna().astype(str)
        if not modes.empty:
            counts = modes.value_counts()
            uncommon_modes = counts.loc[counts.lt(counts.max())].index
            candidates.extend(
                lowres.loc[lowres["mode"].astype(str).isin(uncommon_modes), "id"]
                .astype(int)
                .sort_values(kind="stable")
                .tolist()
            )
    return list(dict.fromkeys(candidates))[:MAX_REVIEW_EXAMPLES]


def _write_review_grids(
    frame: pd.DataFrame,
    lowres: pd.DataFrame,
    exact: pd.DataFrame,
    near: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    counts = distribution_table(frame, "articleType").set_index("label")["count"]
    labelled = frame.assign(_support=frame["articleType"].map(counts))
    groups: dict[str, list[int]] = {
        "common": labelled.sort_values(["_support", "id"], ascending=[False, True])["id"].head(MAX_REVIEW_EXAMPLES).tolist(),
        "rare": labelled.sort_values(["_support", "id"])["id"].head(MAX_REVIEW_EXAMPLES).tolist(),
        "unusual": _unusual_review_ids(lowres),
        "grayscale": lowres.loc[lowres["mode"].eq("L"), "id"].head(MAX_REVIEW_EXAMPLES).tolist(),
        "exact-duplicate": [int(item) for ids in exact.get("ids", pd.Series(dtype=object)) for item in ids],
        "near-duplicate": near.loc[:, ["id_left", "id_right"]].to_numpy().ravel().tolist()
        if not near.empty
        else [],
    }
    manifest: list[str] = []
    for name, ids in groups.items():
        ranked_ids = list(dict.fromkeys(map(int, ids)))[:MAX_REVIEW_EXAMPLES]
        examples = _open_examples(frame, ranked_ids)
        filename = f"review-{name}.png"
        _save_figure(plot_review_grid(examples, f"{name.title()} image review"), output_dir / filename)
        manifest.append(filename)
    return sorted(manifest)


def _remove_stale_known_outputs(output_dir: Path) -> None:
    """Clear generated evidence while retaining valid image cache candidates."""
    for name in GENERATED_EDA_OUTPUTS.difference(_CACHE_OUTPUTS):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def _strongest_pair(frame: pd.DataFrame, associations: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    pairs = [
        (float(associations.loc[left, right]), left, right)
        for index, left in enumerate(associations.index)
        for right in associations.columns[index + 1 :]
        if left in frame
        and right in frame
        and _labels(frame[left]).nunique(dropna=False) <= 20
        and _labels(frame[right]).nunique(dropna=False) <= 20
    ]
    if pairs:
        _, left, right = max(pairs, key=lambda item: item[0])
    else:
        left, right = "gender", "usage"
    return left, right, row_normalized_cooccurrence(frame, left, right)


def run_eda(
    paths: EdaPaths | None = None,
    output_dir: Path = EDA_OUTPUT_DIR,
    refresh: bool = False,
) -> EdaResult:
    """Build reproducible report evidence without reading test labels or making a split."""
    split_before = _split_provenance()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_known_outputs(output_dir)
    population, population_audit = build_population(paths or EdaPaths())
    distributions = {column: distribution_table(population, column) for column in METADATA_COLUMNS}
    target_distributions = {column: distributions[column] for column in TARGET_COLUMNS}
    skew = skew_table(population)
    support = support_band_table(population)
    categorical = tuple(column for column in USEFUL_CATEGORICAL_COLUMNS if column in population)
    associations = association_matrix(population, categorical)
    gender_usage = row_normalized_cooccurrence(population, "gender", "usage")
    strongest_left, strongest_right, strongest = _strongest_pair(population, associations)
    year_drift = drift_table(population, "year", "articleType")
    id_population = deterministic_id_bins(population)
    id_drift = drift_table(id_population, "id_bin", "articleType")

    high_sample = stratified_sample(population, "articleType", EDA_SAMPLE_SIZE, RANDOM_SEED)
    lowres, low_provenance, low_status = _measure_with_cache(
        population, "lowres_image_path", "lowres", output_dir,
        seed=RANDOM_SEED, sample_limit=len(population), refresh=refresh,
    )
    highres, high_provenance, high_status = _measure_with_cache(
        high_sample, "original_image_path", "highres", output_dir,
        seed=RANDOM_SEED, sample_limit=EDA_SAMPLE_SIZE, refresh=refresh,
    )
    near_sample = stratified_sample(population, "articleType", EDA_SAMPLE_SIZE, RANDOM_SEED)
    near_measurements, near_provenance, near_status = _measure_with_cache(
        near_sample, "lowres_image_path", "near-lowres", output_dir,
        seed=RANDOM_SEED, sample_limit=EDA_SAMPLE_SIZE, refresh=refresh,
    )
    exact = exact_duplicate_groups(lowres)
    near = near_duplicate_candidates(near_measurements)
    paired = paired_image_comparison(
        lowres.loc[lowres["id"].isin(high_sample["id"])],
        highres,
    )

    detailed: dict[str, pd.DataFrame] = {
        "target-distributions.csv": pd.concat(
            [table.assign(target=target) for target, table in target_distributions.items()],
            ignore_index=True,
        ),
        "metadata-distributions.csv": pd.concat(
            [table.assign(column=column) for column, table in distributions.items()],
            ignore_index=True,
        ),
        "target-skew.csv": skew.reset_index(),
        "article-type-support.csv": support,
        "associations.csv": associations.reset_index(names="column"),
        "gender-usage.csv": gender_usage.reset_index(names="gender"),
        "strongest-relationship.csv": strongest.reset_index(names=strongest_left),
        "year-drift.csv": year_drift,
        "id-range-drift.csv": id_drift,
        "hierarchy-conflicts.csv": hierarchy_conflicts(population),
        "exact-duplicates.csv": exact,
        "near-duplicates.csv": near,
        "paired-image-comparison.csv": paired,
    }
    for filename, table in detailed.items():
        table.to_csv(output_dir / filename, index=False)
    name_audit = product_name_audit(population)
    data_audit = _json_safe(population_audit)
    _write_json(output_dir / "product-name-audit.json", name_audit)
    _write_json(output_dir / "data-audit.json", data_audit)

    _save_figure(plot_target_distributions(target_distributions), output_dir / "target-distributions.png")
    _save_figure(plot_article_type_support(target_distributions["articleType"], support), output_dir / "article-type-support.png")
    _save_figure(plot_association_matrix(associations), output_dir / "associations.png")
    _save_figure(plot_relationship_heatmap(gender_usage, "gender", "usage"), output_dir / "gender-usage.png")
    _save_figure(plot_relationship_heatmap(strongest, strongest_left, strongest_right), output_dir / "strongest-relationship.png")
    _save_figure(plot_drift(year_drift, "year"), output_dir / "year-drift.png")
    _save_figure(plot_drift(id_drift, "id_bin"), output_dir / "id-range-drift.png")
    _save_figure(plot_image_profiles(paired), output_dir / "image-profiles.png")
    duplicate_summary = pd.DataFrame({"kind": ["exact", "near"], "count": [len(exact), len(near)]})
    _save_figure(plot_duplicate_summary(duplicate_summary), output_dir / "duplicate-summary.png")
    _save_figure(
        build_report_summary(target_distributions, associations, paired),
        output_dir / "eda-report-summary.png",
    )
    review_manifest = _write_review_grids(population, lowres, exact, near, output_dir)
    split_after = _split_provenance()
    split_unchanged = split_before == split_after
    if not split_unchanged:
        raise RuntimeError(
            "Processed split was created or changed during the EDA run"
        )

    test_ids = population_audit.quarantined_test_ids
    manifest = tuple(sorted(GENERATED_EDA_OUTPUTS))
    summary = {
        "population": {
            "source_train_ids": population_audit.source_train_ids,
            "usable_products": population_audit.usable_products,
            "mismatch_ids": {
                "missing_original_metadata": population_audit.missing_original_metadata_ids,
                "missing_original_images": population_audit.missing_original_image_ids,
                "missing_lowres_images": population_audit.missing_lowres_image_ids,
                "missing_both_images": population_audit.missing_both_image_ids,
            },
        },
        "test_quarantine": {
            "test_id_count": len(test_ids),
            "test_ids_hash": sha256(",".join(map(str, test_ids)).encode("utf-8")).hexdigest(),
            "overlap_count": 0,
        },
        "metadata": {
            "distributions": {column: _table_records(table) for column, table in distributions.items()},
            "skew": _table_records(skew.reset_index()),
            "article_type_support": _table_records(support),
        },
        "associations": {
            "cramers_v": associations.to_dict(),
            "gender_usage": gender_usage.to_dict(),
            "strongest_pair": {"left": strongest_left, "right": strongest_right},
        },
        "drift": {"year": _table_records(year_drift), "id_range": _table_records(id_drift)},
        "data_audit": data_audit,
        "names_and_hierarchy": name_audit,
        "images": {
            "lowres_summary": _table_records(lowres.describe(include="all").reset_index()),
            "highres_summary": _table_records(highres.describe(include="all").reset_index()),
            "paired_summary": _table_records(paired.describe(include="all").reset_index()),
        },
        "duplicates": {
            "exact_groups": len(exact),
            "near_candidates": len(near),
            "exact_group_count": len(exact),
            "exact_product_count": int(exact["count"].sum()) if not exact.empty else 0,
            "near_candidate_pair_count": len(near),
            "near_sample_product_count": len(near_measurements),
            "near_values_are": "sampled candidate pairs, not confirmed duplicates",
        },
        "cache_provenance": {
            "lowres": low_provenance,
            "highres": high_provenance,
            "near_lowres": near_provenance,
        },
        "cache_status": {"lowres": low_status, "highres": high_status, "near_lowres": near_status},
        "split_provenance": {
            "before": split_before,
            "after": split_after,
            "unchanged": split_unchanged,
        },
        "output_manifest": {"files": manifest, "review_grids": review_manifest},
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    return EdaResult(
        output_dir=output_dir,
        summary_path=summary_path,
        manifest=manifest,
        cache_status=MappingProxyType(
            {"lowres": low_status, "highres": high_status, "near_lowres": near_status}
        ),
    )


def load_eda_summary(output_dir: Path = EDA_OUTPUT_DIR) -> dict[str, object]:
    """Load the JSON-safe summary generated by :func:`run_eda`."""
    return json.loads((Path(output_dir) / "summary.json").read_text(encoding="utf-8"))
