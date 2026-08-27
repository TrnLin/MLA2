"""Batch-one timing and memory evidence for the frozen Task 4 baseline."""

from __future__ import annotations

import os
import platform
import resource
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.task4.preprocessing import (
    PreprocessedImage,
    PreprocessingContract,
    load_preprocessed_image,
)
from fashion.task4.preprocessing_experiment import (
    FeatureIndex,
    SourceName,
    extract_feature_index,
)
from fashion.task4.probe import (
    PROBE_VERSION,
    extract_spatial_probe,
    rank_probe_embeddings,
)
from fashion.task4.protocol import FIXED_VALIDATION_FOLD, RetrievalViews

__all__ = (
    "IndexCost",
    "TimingPolicy",
    "benchmark_source_direction",
    "build_cost_record",
    "build_protocol_a_search",
    "build_query_encoder",
    "measure_index_build",
    "summarize_timings",
)

_BASELINE_CONTRACT = PreprocessingContract(width=240, height=320)
_INDEX_LIMIT_BYTES = 2**30
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_TIMING_COLUMNS = (
    ("encoding", "encoding_seconds"),
    ("search", "search_seconds"),
    ("end_to_end", "end_to_end_seconds"),
)
_DIRECTIONS = (
    ("teacher", "teacher"),
    ("v1", "v1"),
    ("teacher", "v1"),
    ("v1", "teacher"),
)
_PERCENTILES = ("p50", "p95")


@dataclass(frozen=True)
class TimingPolicy:
    """Fixed CPU timing policy for one-query retrieval."""

    warmup_queries: int = 100
    thread_count: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.warmup_queries, bool)
            or not isinstance(self.warmup_queries, int)
            or self.warmup_queries < 0
        ):
            raise ValueError("warmup queries must be a non-negative integer")
        if self.thread_count != 1:
            raise ValueError("Task 4 timing must use one thread")


@dataclass(frozen=True)
class IndexCost:
    """Measured build and storage cost for one source gallery."""

    source: SourceName
    contract: PreprocessingContract
    rows: int
    dimension: int
    payload_bytes: int
    index_bytes: int
    build_seconds: float
    peak_rss_bytes: int


def _ordered_query_rows(query_rows: pd.DataFrame) -> pd.DataFrame:
    if "id" not in query_rows:
        raise ValueError("query rows are missing id")
    ordered = query_rows.copy()
    numeric_ids = pd.to_numeric(ordered["id"], errors="coerce")
    valid = numeric_ids.notna() & np.isfinite(numeric_ids) & numeric_ids.mod(1).eq(0)
    if not valid.all():
        raise ValueError("query IDs must be integer-compatible")
    ordered["_numeric_id"] = numeric_ids.astype(np.int64)
    if ordered["_numeric_id"].duplicated().any():
        raise ValueError("query IDs must be unique")
    return ordered.sort_values("_numeric_id", kind="mergesort")


def _validate_source(source: str) -> SourceName:
    if source not in {"teacher", "v1"}:
        raise ValueError("source must be 'teacher' or 'v1'")
    return source  # type: ignore[return-value]


def benchmark_source_direction(
    query_rows: pd.DataFrame,
    *,
    query_source: SourceName,
    gallery_source: SourceName,
    encode: Callable[[pd.Series], np.ndarray],
    search: Callable[[int, np.ndarray], pd.DataFrame],
    policy: TimingPolicy = TimingPolicy(),
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> pd.DataFrame:
    """Warm up, then retain one batch-one timing sample per sorted query."""
    query_source = _validate_source(query_source)
    gallery_source = _validate_source(gallery_source)
    ordered = _ordered_query_rows(query_rows)
    for _, row in ordered.iloc[: policy.warmup_queries].iterrows():
        search(int(row["_numeric_id"]), encode(row))

    records: list[dict[str, object]] = []
    for _, row in ordered.iterrows():
        started = clock_ns()
        feature = encode(row)
        encoded = clock_ns()
        search(int(row["_numeric_id"]), feature)
        searched = clock_ns()
        records.append(
            {
                "query_id": int(row["_numeric_id"]),
                "query_source": query_source,
                "gallery_source": gallery_source,
                "encoding_seconds": (encoded - started) / 1e9,
                "search_seconds": (searched - encoded) / 1e9,
                "end_to_end_seconds": (searched - started) / 1e9,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "query_id",
            "query_source",
            "gallery_source",
            "encoding_seconds",
            "search_seconds",
            "end_to_end_seconds",
        ],
    )


def summarize_timings(samples: pd.DataFrame) -> pd.DataFrame:
    """Report linear p50 and p95 while retaining the full sample count."""
    required = {
        "query_source",
        "gallery_source",
        *(column for _, column in _TIMING_COLUMNS),
    }
    if missing := required.difference(samples.columns):
        raise ValueError(f"timing samples are missing columns: {sorted(missing)}")
    if samples.empty:
        raise ValueError("timing samples must not be empty")
    directions = samples.loc[:, ["query_source", "gallery_source"]].drop_duplicates()
    if len(directions) != 1:
        raise ValueError("timing samples must describe exactly one source direction")
    query_source = _validate_source(str(directions["query_source"].iloc[0]))
    gallery_source = _validate_source(str(directions["gallery_source"].iloc[0]))
    numeric_columns = [column for _, column in _TIMING_COLUMNS]
    numeric = samples.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("timing samples must contain finite numbers")
    if numeric.lt(0).any().any():
        raise ValueError("timing samples must be non-negative")
    records = [
        {
            "query_source": query_source,
            "gallery_source": gallery_source,
            "metric": metric,
            "percentile": percentile,
            "value_seconds": float(np.quantile(numeric[column], quantile)),
            "timed_queries": len(samples),
        }
        for metric, column in _TIMING_COLUMNS
        for percentile, quantile in (("p50", 0.50), ("p95", 0.95))
    ]
    return pd.DataFrame.from_records(records)


def _resolve_path(value: object, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def build_query_encoder(
    *,
    path_column: str,
    contract: PreprocessingContract,
    root: str | Path = ROOT,
    load_image: Callable[
        [str | Path, PreprocessingContract], PreprocessedImage
    ] = load_preprocessed_image,
    extract: Callable[[np.ndarray, np.ndarray], np.ndarray] = extract_spatial_probe,
) -> Callable[[pd.Series], np.ndarray]:
    """Build an encoder whose interval starts at local file loading."""
    if contract != _BASELINE_CONTRACT:
        raise ValueError("query timing requires the frozen 240x320 contract")
    root_path = Path(root)

    def encode(row: pd.Series) -> np.ndarray:
        if path_column not in row:
            raise ValueError(f"query row is missing {path_column}")
        transformed = load_image(_resolve_path(row[path_column], root_path), contract)
        return extract(transformed.pixels, transformed.content_mask)

    return encode


def build_protocol_a_search(
    *,
    views: RetrievalViews,
    gallery_index: FeatureIndex,
) -> Callable[[int, np.ndarray], pd.DataFrame]:
    """Build the complete Protocol A cosine-to-Top-20 search adapter."""
    if gallery_index.contract != _BASELINE_CONTRACT:
        raise ValueError("search timing requires the frozen 240x320 contract")
    ordered_queries = _ordered_query_rows(views.queries).set_index("_numeric_id", drop=False)
    gallery_views = views.gallery.copy()

    def search(query_id: int, feature: np.ndarray) -> pd.DataFrame:
        if query_id not in ordered_queries.index:
            raise ValueError(f"query ID {query_id} is outside the retrieval view")
        one_query = ordered_queries.loc[[query_id]].drop(columns="_numeric_id")
        return rank_probe_embeddings(
            query_ids=np.asarray([query_id], dtype=np.int64),
            query_features=np.asarray(feature, dtype=np.float32).reshape(1, -1),
            gallery_ids=gallery_index.ids,
            gallery_features=gallery_index.features,
            views=RetrievalViews(queries=one_query, gallery=gallery_views),
            protocol="primary",
            max_k=20,
            chunk_size=1,
        )

    return search


def _index_build_worker(
    connection: Connection,
    gallery_rows: pd.DataFrame,
    source: SourceName,
    path_column: str,
    contract: PreprocessingContract,
    root: str | Path,
) -> None:
    try:
        started = time.perf_counter()
        index = extract_feature_index(
            gallery_rows,
            path_column=path_column,
            source=source,
            contract=contract,
            root=root,
            workers=1,
        )
        build_seconds = time.perf_counter() - started
        peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        connection.send(
            (
                "ok",
                IndexCost(
                    source=source,
                    contract=contract,
                    rows=int(index.features.shape[0]),
                    dimension=int(index.features.shape[1]),
                    payload_bytes=int(index.features.nbytes),
                    index_bytes=int(index.features.nbytes + index.ids.nbytes),
                    build_seconds=float(build_seconds),
                    peak_rss_bytes=int(peak_rss_bytes),
                ),
            )
        )
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def measure_index_build(
    gallery_rows: pd.DataFrame,
    *,
    source: SourceName,
    path_column: str,
    contract: PreprocessingContract,
    root: str | Path = ROOT,
) -> IndexCost:
    """Build one gallery index in a spawned child and return its measured cost."""
    from multiprocessing import get_context

    source = _validate_source(source)
    if contract != _BASELINE_CONTRACT:
        raise ValueError("index measurement requires the frozen 240x320 contract")
    context = get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(
        target=_index_build_worker,
        args=(sending, gallery_rows, source, path_column, contract, root),
    )
    process.start()
    sending.close()
    process.join()
    if process.exitcode != 0:
        receiving.close()
        raise RuntimeError(f"index build child exited with code {process.exitcode}")
    if not receiving.poll():
        receiving.close()
        raise RuntimeError("index build child returned no result")
    status, payload = receiving.recv()
    receiving.close()
    if status == "error":
        raise RuntimeError(f"index build child failed:\n{payload}")
    if not isinstance(payload, IndexCost):
        raise RuntimeError("index build child returned an invalid result")
    return payload


def _cpu_text() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _thread_environment() -> dict[str, str]:
    environment = {variable: os.environ.get(variable) for variable in _THREAD_VARIABLES}
    if any(value != "1" for value in environment.values()):
        raise ValueError("thread environment variables must all equal '1'")
    return {variable: str(value) for variable, value in environment.items()}


def _hardware_record(
    policy: TimingPolicy,
    thread_environment: Mapping[str, str],
) -> dict[str, object]:
    return {
        "cpu": _cpu_text(),
        "logical_cores": os.cpu_count(),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "thread_count": policy.thread_count,
        "thread_environment": dict(thread_environment),
    }


def _json_value(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Real) and not isinstance(value, Integral):
        if not np.isfinite(value):
            raise ValueError("cost evidence contains a non-finite numeric value")
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _validated_timing_summary(timing_summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "query_source",
        "gallery_source",
        "metric",
        "percentile",
        "value_seconds",
        "timed_queries",
    }
    if missing := required.difference(timing_summary.columns):
        raise ValueError(f"timing summary is missing columns: {sorted(missing)}")
    key_columns = ["query_source", "gallery_source", "metric", "percentile"]
    if timing_summary.duplicated(key_columns).any():
        raise ValueError("timing summary requires exactly one row per combination")
    expected = {
        (query_source, gallery_source, metric, percentile)
        for query_source, gallery_source in _DIRECTIONS
        for metric, _ in _TIMING_COLUMNS
        for percentile in _PERCENTILES
    }
    observed = set(timing_summary.loc[:, key_columns].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("timing summary must contain the complete four-direction grid")

    validated = timing_summary.copy()
    values = pd.to_numeric(validated["value_seconds"], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or values.lt(0).any():
        raise ValueError("timing summary values must be finite and non-negative")
    counts = pd.to_numeric(validated["timed_queries"], errors="coerce")
    valid_counts = (
        counts.notna()
        & np.isfinite(counts)
        & counts.mod(1).eq(0)
        & counts.gt(0)
    )
    if not valid_counts.all() or counts.nunique() != 1:
        raise ValueError("timing summary must contain one positive timed-query count")
    validated["value_seconds"] = values.astype(float)
    validated["timed_queries"] = counts.astype(int)
    return validated


def _validate_index_cost(cost: IndexCost, source: str) -> None:
    if cost.source != source:
        raise ValueError("index cost source does not match its key")
    if cost.contract != _BASELINE_CONTRACT:
        raise ValueError("index costs must use the frozen 240x320 contract")
    integer_values = (
        cost.rows,
        cost.dimension,
        cost.payload_bytes,
        cost.index_bytes,
        cost.peak_rss_bytes,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < 0
        for value in integer_values
    ):
        raise ValueError("index cost counts and byte values must be non-negative integers")
    if cost.rows <= 0 or cost.dimension <= 0:
        raise ValueError("index cost rows and dimension must be positive")
    if (
        isinstance(cost.build_seconds, bool)
        or not isinstance(cost.build_seconds, Real)
        or not np.isfinite(cost.build_seconds)
        or cost.build_seconds < 0
    ):
        raise ValueError("index build seconds must be a finite non-negative number")


def build_cost_record(
    timing_summary: pd.DataFrame,
    index_costs: Mapping[SourceName, IndexCost],
    *,
    policy: TimingPolicy,
) -> dict[str, object]:
    """Build JSON-safe baseline cost, hardware, and practical-threshold evidence."""
    validated_timings = _validated_timing_summary(timing_summary)
    timed_queries = int(validated_timings["timed_queries"].iloc[0])
    p95_values = validated_timings.loc[
        validated_timings["metric"].eq("end_to_end")
        & validated_timings["percentile"].eq("p95"),
        "value_seconds",
    ]
    if set(index_costs) != {"teacher", "v1"}:
        raise ValueError("index costs must contain exactly teacher and v1")
    costs: dict[str, dict[str, object]] = {}
    for source in ("teacher", "v1"):
        cost = index_costs[source]
        _validate_index_cost(cost, source)
        costs[source] = asdict(cost)
    thread_environment = _thread_environment()

    return _json_value(
        {
            "schema_version": 1,
            "scope": "development",
            "fold": FIXED_VALIDATION_FOLD,
            "contract": _BASELINE_CONTRACT.to_dict(),
            "probe_version": PROBE_VERSION,
            "parameters": 0,
            "checkpoint_bytes": 0,
            "hardware": _hardware_record(policy, thread_environment),
            "warmup_queries": policy.warmup_queries,
            "timed_queries": timed_queries,
            "timing_summary": validated_timings.to_dict("records"),
            "per_source_index_cost": costs,
            "p95_end_to_end_under_one_second": bool((p95_values < 1.0).all()),
            "index_under_one_gibibyte": bool(
                all(cost.index_bytes < _INDEX_LIMIT_BYTES for cost in index_costs.values())
            ),
        }
    )
