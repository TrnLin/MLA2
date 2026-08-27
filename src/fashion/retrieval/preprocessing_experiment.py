"""Development-only orchestration for the Task 4 preprocessing comparison."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from fashion.config import ROOT
from fashion.retrieval.preprocessing import (
    PreprocessingContract,
    load_preprocessed_image,
)
from fashion.retrieval.probe import (
    PROBE_VERSION,
    extract_spatial_probe,
    rank_probe_embeddings,
)
from fashion.retrieval.protocol import (
    RetrievalViews,
    build_development_views,
    evaluate_family_rankings,
    evaluate_primary_rankings,
)

SourceName = Literal["teacher", "v1"]


@dataclass(frozen=True)
class FeatureIndex:
    """ID-aligned fixed descriptors and their measured extraction cost."""

    source: str
    contract: PreprocessingContract
    ids: np.ndarray
    features: np.ndarray
    transform_seconds: float
    source_bytes: int


@dataclass(frozen=True)
class PairEvaluation:
    """Rankings, per-query values, and summaries for one source direction."""

    summary: pd.DataFrame
    primary_rankings: pd.DataFrame
    family_rankings: pd.DataFrame
    primary_per_query: pd.DataFrame
    family_per_query: pd.DataFrame


@dataclass(frozen=True)
class CachedFeatureIndex:
    """A validated local feature cache and its opened index."""

    cache_dir: Path
    index: FeatureIndex
    manifest: dict[str, object]


@dataclass(frozen=True)
class PreprocessingExperiment:
    """Complete comparison outputs plus reusable feature indexes."""

    comparison: pd.DataFrame
    selection: pd.DataFrame
    stability: pd.DataFrame
    stability_summary: pd.DataFrame
    top_sizes: tuple[str, ...]
    feature_indexes: dict[tuple[str, str], FeatureIndex]


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    if missing := required.difference(frame.columns):
        raise ValueError(f"experiment frame is missing columns: {sorted(missing)}")


def _development_frame(frame: pd.DataFrame, path_column: str) -> pd.DataFrame:
    _require_columns(frame, {"id", "partition", path_column})
    if frame.empty or not frame["partition"].eq("development").all():
        raise ValueError("preprocessing experiments require development rows only")
    working = frame.copy()
    numeric_ids = pd.to_numeric(working["id"], errors="coerce")
    if numeric_ids.isna().any() or not numeric_ids.mod(1).eq(0).all():
        raise ValueError("feature IDs must be integer-compatible")
    working["id"] = numeric_ids.astype(np.int64)
    if working["id"].duplicated().any():
        raise ValueError("feature IDs must be unique")
    return working.sort_values("id").reset_index(drop=True)


def _resolve_path(value: object, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def extract_feature_index(
    frame: pd.DataFrame,
    *,
    path_column: str,
    source: SourceName,
    contract: PreprocessingContract,
    root: str | Path = ROOT,
    workers: int | None = None,
) -> FeatureIndex:
    """Transform and describe one complete development image source."""
    working = _development_frame(frame, path_column)
    root_path = Path(root)
    paths = [_resolve_path(value, root_path) for value in working[path_column]]
    worker_count = workers or min(8, os.cpu_count() or 4)
    if isinstance(worker_count, bool) or worker_count <= 0:
        raise ValueError("workers must be positive")

    def transform(path: Path) -> np.ndarray:
        image = load_preprocessed_image(path, contract)
        return extract_spatial_probe(image.pixels, image.content_mask)

    started = time.perf_counter()
    if worker_count == 1:
        features = [transform(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            features = list(executor.map(transform, paths))
    elapsed = time.perf_counter() - started
    return FeatureIndex(
        source=source,
        contract=contract,
        ids=working["id"].to_numpy(dtype=np.int64),
        features=np.stack(features).astype(np.float32, copy=False),
        transform_seconds=elapsed,
        source_bytes=sum(path.stat().st_size for path in paths),
    )


def _feature_source_fingerprint(
    frame: pd.DataFrame,
    *,
    path_column: str,
    sha_column: str | None,
    root: Path,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    for row in frame.to_dict("records"):
        path = _resolve_path(row[path_column], root)
        file_size = path.stat().st_size
        total_bytes += file_size
        source_sha = str(row[sha_column]) if sha_column is not None else ""
        digest.update(
            f"{int(row['id'])}\t{row[path_column]}\t{source_sha}\t{file_size}\n".encode()
        )
    return digest.hexdigest(), total_bytes


def _load_cached_feature_index(
    cache_dir: Path,
    contract: PreprocessingContract,
) -> CachedFeatureIndex:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = np.load(cache_dir / "ids.npy", mmap_mode="r")
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    if ids.dtype != np.int64 or ids.shape != (int(manifest["rows"]),):
        raise ValueError("cached feature IDs do not match their manifest")
    if features.dtype != np.float32 or list(features.shape) != manifest["feature_shape"]:
        raise ValueError("cached features do not match their manifest")
    if not np.array_equal(ids, np.sort(ids)) or len(np.unique(ids)) != len(ids):
        raise ValueError("cached feature IDs must be sorted and unique")
    index = FeatureIndex(
        source=str(manifest["source"]),
        contract=contract,
        ids=ids,
        features=features,
        transform_seconds=float(manifest["transform_seconds"]),
        source_bytes=int(manifest["source_bytes"]),
    )
    return CachedFeatureIndex(cache_dir=cache_dir, index=index, manifest=manifest)


def ensure_feature_index(
    frame: pd.DataFrame,
    *,
    path_column: str,
    source: SourceName,
    contract: PreprocessingContract,
    cache_root: str | Path,
    root: str | Path = ROOT,
    sha_column: str | None = None,
    workers: int | None = None,
) -> CachedFeatureIndex:
    """Reuse or extract one exact local probe-feature index."""
    working = _development_frame(frame, path_column)
    if sha_column is not None:
        _require_columns(working, {sha_column})
    root_path = Path(root)
    fingerprint, source_bytes = _feature_source_fingerprint(
        working,
        path_column=path_column,
        sha_column=sha_column,
        root=root_path,
    )
    expected: dict[str, object] = {
        "schema_version": "1.0.0",
        "scope": "development",
        "probe": PROBE_VERSION,
        "source": source,
        "path_column": path_column,
        "sha_column": sha_column,
        "rows": len(working),
        "source_fingerprint": fingerprint,
        "source_bytes": source_bytes,
        "contract": contract.to_dict(),
    }
    cache_dir = Path(cache_root) / contract.key / source
    if cache_dir.is_dir():
        try:
            cached = _load_cached_feature_index(cache_dir, contract)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            cached = None
        if cached is not None and all(
            cached.manifest.get(key) == value for key, value in expected.items()
        ):
            return cached

    index = extract_feature_index(
        working,
        path_column=path_column,
        source=source,
        contract=contract,
        root=root_path,
        workers=workers,
    )
    manifest = {
        **expected,
        "transform_seconds": index.transform_seconds,
        "feature_shape": list(index.features.shape),
        "feature_dtype": "float32",
    }
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{cache_dir.name}-", dir=cache_dir.parent)
    )
    try:
        np.save(temporary / "ids.npy", index.ids, allow_pickle=False)
        np.save(temporary / "features.npy", index.features, allow_pickle=False)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(temporary, cache_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _load_cached_feature_index(cache_dir, contract)


def source_directions() -> tuple[tuple[SourceName, SourceName], ...]:
    """Return the approved same-source and cross-source matrix order."""
    return (
        ("teacher", "teacher"),
        ("v1", "v1"),
        ("teacher", "v1"),
        ("v1", "teacher"),
    )


def _features_for_ids(
    index: FeatureIndex,
    ids: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    requested = pd.to_numeric(ids, errors="coerce")
    if requested.isna().any() or not requested.mod(1).eq(0).all():
        raise ValueError("retrieval view IDs must be integer-compatible")
    numeric = requested.to_numpy(dtype=np.int64)
    positions = {int(product_id): position for position, product_id in enumerate(index.ids)}
    missing = sorted(set(numeric).difference(positions))
    if missing:
        raise ValueError(f"feature index is missing retrieval IDs: {missing[:10]}")
    selected = np.fromiter(
        (positions[int(product_id)] for product_id in numeric),
        dtype=np.int64,
        count=len(numeric),
    )
    return numeric, index.features[selected]


def _label_summary(
    summary: pd.DataFrame,
    *,
    protocol: str,
    query_index: FeatureIndex,
    gallery_index: FeatureIndex,
    fold: int,
) -> pd.DataFrame:
    contract = query_index.contract
    labelled = summary.copy()
    metadata: dict[str, object] = {
        "fold": int(fold),
        "size": f"{contract.width}x{contract.height}",
        "width": contract.width,
        "height": contract.height,
        "query_source": query_index.source,
        "gallery_source": gallery_index.source,
        "protocol": protocol,
        "query_transform_seconds": query_index.transform_seconds,
        "gallery_transform_seconds": gallery_index.transform_seconds,
        "query_source_bytes": query_index.source_bytes,
        "gallery_source_bytes": gallery_index.source_bytes,
        "feature_bytes_per_image": query_index.features.shape[1]
        * query_index.features.dtype.itemsize,
        "uint8_tensor_bytes_per_image": contract.pixel_count * 3,
        "float32_tensor_bytes_per_image": contract.pixel_count * 3 * 4,
    }
    for position, (column, value) in enumerate(metadata.items()):
        labelled.insert(position, column, value)
    return labelled


def evaluate_source_pair(
    query_index: FeatureIndex,
    gallery_index: FeatureIndex,
    *,
    primary_views: RetrievalViews,
    family_views: RetrievalViews,
    fold: int,
    k_values: tuple[int, ...] = (5, 10, 20),
    family_k: int = 10,
    chunk_size: int = 128,
) -> PairEvaluation:
    """Evaluate one teacher/V1 query-gallery direction under both protocols."""
    if query_index.contract != gallery_index.contract:
        raise ValueError("query and gallery feature indexes must share one contract")
    primary_query_ids, primary_query_features = _features_for_ids(
        query_index, primary_views.queries["id"]
    )
    primary_gallery_ids, primary_gallery_features = _features_for_ids(
        gallery_index, primary_views.gallery["id"]
    )
    primary_rankings = rank_probe_embeddings(
        query_ids=primary_query_ids,
        query_features=primary_query_features,
        gallery_ids=primary_gallery_ids,
        gallery_features=primary_gallery_features,
        views=primary_views,
        protocol="primary",
        max_k=max(k_values),
        chunk_size=chunk_size,
    )
    primary_per_query, primary_summary = evaluate_primary_rankings(
        primary_rankings,
        primary_views,
        k_values=k_values,
    )

    family_query_ids, family_query_features = _features_for_ids(
        query_index, family_views.queries["id"]
    )
    family_gallery_ids, family_gallery_features = _features_for_ids(
        gallery_index, family_views.gallery["id"]
    )
    family_rankings = rank_probe_embeddings(
        query_ids=family_query_ids,
        query_features=family_query_features,
        gallery_ids=family_gallery_ids,
        gallery_features=family_gallery_features,
        views=family_views,
        protocol="family",
        max_k=family_k,
        chunk_size=chunk_size,
    )
    family_per_query, family_summary = evaluate_family_rankings(
        family_rankings,
        family_views,
        k=family_k,
    )
    labelled_primary = _label_summary(
        primary_summary,
        protocol="primary",
        query_index=query_index,
        gallery_index=gallery_index,
        fold=fold,
    )
    labelled_family = _label_summary(
        family_summary,
        protocol="family",
        query_index=query_index,
        gallery_index=gallery_index,
        fold=fold,
    )
    return PairEvaluation(
        summary=pd.concat([labelled_primary, labelled_family], ignore_index=True),
        primary_rankings=primary_rankings,
        family_rankings=family_rankings,
        primary_per_query=primary_per_query,
        family_per_query=family_per_query,
    )


def _primary_ndcg_rows(results: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fold",
        "size",
        "query_source",
        "gallery_source",
        "protocol",
        "metric",
        "k",
        "aggregation",
        "value",
    }
    _require_columns(results, required)
    return results.loc[
        results["protocol"].eq("primary")
        & results["metric"].eq("ndcg")
        & results["k"].eq(10)
        & results["aggregation"].eq("query_mean")
    ].copy()


def build_size_selection(results: pd.DataFrame) -> pd.DataFrame:
    """Select one shared size from the equal teacher/V1 same-source mean."""
    _require_columns(results, {"width", "height"})
    primary = _primary_ndcg_rows(results)
    primary = primary.loc[
        primary["fold"].eq(1)
        & primary["query_source"].eq(primary["gallery_source"])
        & primary["query_source"].isin(("teacher", "v1"))
    ]

    records: list[dict[str, object]] = []
    for size, size_rows in primary.groupby("size", sort=False):
        if len(size_rows) != 2 or set(size_rows["query_source"]) != {"teacher", "v1"}:
            raise ValueError(f"size {size} needs one teacher and one V1 same-source score")
        widths = size_rows["width"].unique()
        heights = size_rows["height"].unique()
        if len(widths) != 1 or len(heights) != 1:
            raise ValueError(f"size {size} has inconsistent geometry")
        by_source = size_rows.set_index("query_source")["value"].astype(float)
        width = int(widths[0])
        height = int(heights[0])
        records.append(
            {
                "size": size,
                "width": width,
                "height": height,
                "pixels": width * height,
                "teacher_ndcg_at_10": float(by_source["teacher"]),
                "v1_ndcg_at_10": float(by_source["v1"]),
                "selection_ndcg_at_10": float(by_source.mean()),
            }
        )
    if not records:
        raise ValueError("no fold-1 same-source preprocessing scores were found")
    selection = pd.DataFrame(records).sort_values(
        ["selection_ndcg_at_10", "pixels"],
        ascending=[False, True],
        kind="mergesort",
    )
    selection.insert(0, "selection_rank", np.arange(1, len(selection) + 1))
    return selection.reset_index(drop=True)


def select_top_sizes(selection: pd.DataFrame, count: int = 2) -> tuple[str, ...]:
    """Return ranked size names after validating the requested count."""
    _require_columns(selection, {"selection_rank", "size"})
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    ordered = selection.sort_values("selection_rank")
    if len(ordered) < count:
        raise ValueError("selection has fewer sizes than requested")
    return tuple(ordered["size"].astype(str).iloc[:count])


def summarize_stability(
    results: pd.DataFrame,
    *,
    top_sizes: tuple[str, ...],
) -> pd.DataFrame:
    """Aggregate equal-source size scores across all five held-out folds."""
    if len(top_sizes) != 2 or len(set(top_sizes)) != 2:
        raise ValueError("stability requires exactly two unique top sizes")
    primary = _primary_ndcg_rows(results)
    primary = primary.loc[
        primary["size"].isin(top_sizes)
        & primary["query_source"].eq(primary["gallery_source"])
        & primary["query_source"].isin(("teacher", "v1"))
    ]
    fold_records: list[dict[str, object]] = []
    for (size, fold), fold_rows in primary.groupby(["size", "fold"], sort=False):
        if len(fold_rows) != 2 or set(fold_rows["query_source"]) != {"teacher", "v1"}:
            raise ValueError(f"size {size} fold {fold} needs both same-source scores")
        fold_records.append(
            {
                "size": size,
                "fold": int(fold),
                "selection_ndcg_at_10": float(fold_rows["value"].astype(float).mean()),
            }
        )
    fold_scores = pd.DataFrame(fold_records)
    records: list[dict[str, object]] = []
    for size in top_sizes:
        values = (
            fold_scores.loc[fold_scores["size"].eq(size)]
            .sort_values("fold")
            .reset_index(drop=True)
        )
        if values["fold"].tolist() != list(range(5)):
            raise ValueError(f"size {size} must contain folds 0 through 4 exactly once")
        records.append(
            {
                "size": size,
                "fold_count": len(values),
                "mean_selection_ndcg_at_10": float(values["selection_ndcg_at_10"].mean()),
                "std_selection_ndcg_at_10": float(
                    values["selection_ndcg_at_10"].std(ddof=1)
                ),
            }
        )
    return pd.DataFrame(records)


def _stability_detail(
    comparison: pd.DataFrame,
    *,
    top_sizes: tuple[str, ...],
    stability_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = _primary_ndcg_rows(comparison)
    rows = rows.loc[
        rows["size"].isin(top_sizes)
        & rows["query_source"].eq(rows["gallery_source"])
        & rows["query_source"].isin(("teacher", "v1"))
    ]
    detail = (
        rows.pivot_table(
            index=["size", "fold"],
            columns="query_source",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "teacher": "teacher_ndcg_at_10",
                "v1": "v1_ndcg_at_10",
            }
        )
    )
    detail["selection_ndcg_at_10"] = detail[
        ["teacher_ndcg_at_10", "v1_ndcg_at_10"]
    ].mean(axis=1)
    detail = detail.merge(stability_summary, on="size", validate="many_to_one")
    detail.insert(0, "scope", "development")
    return detail.sort_values(["size", "fold"]).reset_index(drop=True)


def run_preprocessing_experiment(
    splits: pd.DataFrame,
    variant_index: pd.DataFrame,
    *,
    contracts: tuple[PreprocessingContract, ...],
    feature_cache_root: str | Path,
    root: str | Path = ROOT,
    workers: int | None = None,
    k_values: tuple[int, ...] = (5, 10, 20),
    family_k: int = 10,
    chunk_size: int = 256,
) -> PreprocessingExperiment:
    """Run the complete size/source matrix and top-two five-fold check."""
    if 10 not in k_values:
        raise ValueError("the preprocessing experiment requires Protocol A nDCG@10")
    if len(contracts) < 2 or len({contract.key for contract in contracts}) != len(
        contracts
    ):
        raise ValueError("the preprocessing experiment requires unique candidate contracts")
    _require_columns(
        variant_index,
        {
            "id",
            "partition",
            "teacher_path",
            "teacher_sha256",
            "external_path",
            "external_sha256",
        },
    )
    development = variant_index.loc[
        variant_index["partition"].eq("development")
    ].copy()
    if len(development) != len(variant_index):
        raise ValueError("the preprocessing runner accepts development variants only")
    expected_ids = set(
        splits.loc[splits["partition"].eq("development"), "id"].astype(int)
    )
    if set(development["id"].astype(int)) != expected_ids:
        raise ValueError("variant IDs must exactly match canonical development IDs")

    source_specs: dict[SourceName, tuple[str, str]] = {
        "teacher": ("teacher_path", "teacher_sha256"),
        "v1": ("external_path", "external_sha256"),
    }
    feature_indexes: dict[tuple[str, str], FeatureIndex] = {}
    for contract in contracts:
        size = f"{contract.width}x{contract.height}"
        for source, (path_column, sha_column) in source_specs.items():
            feature_indexes[(size, source)] = ensure_feature_index(
                development,
                path_column=path_column,
                sha_column=sha_column,
                source=source,
                contract=contract,
                cache_root=feature_cache_root,
                root=root,
                workers=workers,
            ).index

    summaries: list[pd.DataFrame] = []
    primary, family = build_development_views(splits, validation_fold=1)
    for contract in contracts:
        size = f"{contract.width}x{contract.height}"
        for query_source, gallery_source in source_directions():
            evaluation = evaluate_source_pair(
                feature_indexes[(size, query_source)],
                feature_indexes[(size, gallery_source)],
                primary_views=primary,
                family_views=family,
                fold=1,
                k_values=k_values,
                family_k=family_k,
                chunk_size=chunk_size,
            )
            summaries.append(evaluation.summary)

    fold_one = pd.concat(summaries, ignore_index=True)
    selection = build_size_selection(fold_one)
    selection.insert(0, "scope", "development")
    top_sizes = select_top_sizes(selection, count=2)
    for size in top_sizes:
        for fold in range(5):
            if fold == 1:
                continue
            primary_fold, family_fold = build_development_views(
                splits, validation_fold=fold
            )
            for source in ("teacher", "v1"):
                evaluation = evaluate_source_pair(
                    feature_indexes[(size, source)],
                    feature_indexes[(size, source)],
                    primary_views=primary_fold,
                    family_views=family_fold,
                    fold=fold,
                    k_values=k_values,
                    family_k=family_k,
                    chunk_size=chunk_size,
                )
                summaries.append(evaluation.summary)

    comparison = pd.concat(summaries, ignore_index=True)
    comparison.insert(0, "scope", "development")
    stability_summary = summarize_stability(comparison, top_sizes=top_sizes)
    stability = _stability_detail(
        comparison,
        top_sizes=top_sizes,
        stability_summary=stability_summary,
    )
    return PreprocessingExperiment(
        comparison=comparison,
        selection=selection,
        stability=stability,
        stability_summary=stability_summary,
        top_sizes=top_sizes,
        feature_indexes=feature_indexes,
    )


def build_odd_aspect_canvas(
    image: Image.Image,
    orientation: Literal["wide", "tall"],
) -> Image.Image:
    """Place an image unchanged on a deterministic white 2:1 or 1:2 canvas."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if orientation == "wide":
        size = (max(rgb.width, 2 * rgb.height), rgb.height)
    elif orientation == "tall":
        size = (rgb.width, max(rgb.height, 2 * rgb.width))
    else:
        raise ValueError("orientation must be 'wide' or 'tall'")
    canvas = Image.new("RGB", size, (255, 255, 255))
    offset = ((size[0] - rgb.width) // 2, (size[1] - rgb.height) // 2)
    canvas.paste(rgb, offset)
    return canvas
