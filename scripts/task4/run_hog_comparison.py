import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mla2-task4-hog/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mla2-task4-hog/xdg")

import argparse
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fashion.config import ROOT
from fashion.data.dataset import load_splits
from fashion.data.hashing import write_deterministic_csv
from fashion.data.splits import cv_assignment_digest
from fashion.task4.analysis import (
    CanvasStressEvaluation,
    build_query_support,
    evaluate_canvas_stress,
    mark_failure_slices,
    summarize_failure_slices,
)
from fashion.task4.baseline import BaselineEvaluation, evaluate_baseline
from fashion.task4.baseline_evidence import render_baseline_examples
from fashion.task4.benchmark import (
    TimingPolicy,
    benchmark_source_direction,
    build_cost_record,
    build_protocol_a_search,
    build_query_encoder,
    measure_index_build,
    summarize_timings,
)
from fashion.task4.cache import (
    DevelopmentImageCache,
    load_development_image_cache,
)
from fashion.task4.hog import (
    HOG_CONFIG,
    HOG_CONFIG_FINGERPRINT,
    HOG_METHOD,
    extract_hog,
)
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import (
    FeatureIndex,
    extract_canvas_feature_index,
    source_directions,
)
from fashion.task4.protocol import build_development_views

hog_module = __import__("fashion.task4.hog", fromlist=["ensure_hog_feature_index"])
_BASELINE_RUNNER_PATH = ROOT / "scripts/task4/run_baseline.py"
_BASELINE_SPEC = importlib.util.spec_from_file_location(
    "_task4_baseline_runner",
    _BASELINE_RUNNER_PATH,
)
if _BASELINE_SPEC is None or _BASELINE_SPEC.loader is None:
    raise RuntimeError("could not load the Task 4 baseline runner")
baseline_runner = importlib.util.module_from_spec(_BASELINE_SPEC)
sys.modules[_BASELINE_SPEC.name] = baseline_runner
_BASELINE_SPEC.loader.exec_module(baseline_runner)

CONTRACT = PreprocessingContract(width=240, height=320)
INDEX_LIMIT_BYTES = 2**30
P95_LIMIT_SECONDS = 1.0
SOURCE_SPECS = {
    "teacher": ("teacher_path", "teacher_sha256"),
    "v1": ("external_path", "external_sha256"),
}
VARIANT_COLUMNS = (
    "id",
    "partition",
    "teacher_path",
    "external_path",
    "external_sha256",
)


def _progress(stage: str, event: str) -> None:
    print(f"task4-hog stage={stage} event={event}", flush=True)


@contextmanager
def _thread_limit(count: int):
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("thread limit must be a positive integer")
    with threadpool_limits(limits=count):
        yield


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded development-only Task 4 HOG comparison."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="CPU feature workers; timed queries remain one-threaded",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _select_development_sources(
    splits: pd.DataFrame,
    variants: pd.DataFrame,
) -> pd.DataFrame:
    required_splits = {"id", "partition", "sha256"}
    required_variants = set(VARIANT_COLUMNS)
    if missing := required_splits.difference(splits.columns):
        raise ValueError(f"canonical splits are missing columns: {sorted(missing)}")
    if missing := required_variants.difference(variants.columns):
        raise ValueError(f"variant index is missing columns: {sorted(missing)}")
    development_splits = splits.loc[
        splits["partition"].eq("development"), ["id", "sha256"]
    ].copy()
    development = variants.loc[variants["partition"].eq("development")].copy()
    development["id"] = pd.to_numeric(development["id"], errors="raise").astype(
        np.int64
    )
    development_splits["id"] = pd.to_numeric(
        development_splits["id"], errors="raise"
    ).astype(np.int64)
    if development.empty or development["id"].duplicated().any():
        raise ValueError("HOG variants require unique development IDs")
    if set(development["id"]) != set(development_splits["id"]):
        raise ValueError(
            "HOG variants must exactly match canonical development split IDs"
        )
    hashes = development_splits.set_index("id")["sha256"]
    development["teacher_sha256"] = development["id"].map(hashes)
    for column in (
        "teacher_path",
        "external_path",
        "teacher_sha256",
        "external_sha256",
    ):
        if development[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"HOG development sources have blank {column}")
    return development.sort_values("id").reset_index(drop=True)


def _load_development_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = load_splits()
    variants = pd.read_csv(
        ROOT / "data/processed/task4/external_variant_index.csv.gz",
        usecols=list(VARIANT_COLUMNS),
        keep_default_na=False,
    )
    return splits, _select_development_sources(splits, variants)


def _load_image_caches(
    development: pd.DataFrame,
) -> dict[str, DevelopmentImageCache]:
    cache_root = (
        ROOT
        / "data/processed/task4/preprocessing/images"
        / CONTRACT.key
    )
    caches = {
        source: load_development_image_cache(cache_root / source)
        for source in ("teacher", "v1")
    }
    expected_ids = development["id"].to_numpy(dtype=np.int64)
    for source, cache in caches.items():
        if cache.manifest.get("source") != source:
            raise ValueError("HOG image-cache source identity is malformed")
        if cache.manifest.get("contract") != CONTRACT.to_dict():
            raise ValueError("HOG image caches must use frozen 240x320 preprocessing")
        if not np.array_equal(cache.ids, expected_ids):
            raise ValueError("HOG image-cache IDs do not match development inputs")
    return caches


def _open_hog_indexes(
    caches: dict[str, DevelopmentImageCache],
    *,
    workers: int,
    cache_root: str | Path,
) -> dict[str, FeatureIndex]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if set(caches) != {"teacher", "v1"}:
        raise ValueError("HOG requires teacher and V1 image caches")
    per_source_workers = max(1, workers // 2)

    def build(source: str):
        return hog_module.ensure_hog_feature_index(
            caches[source],
            cache_root=cache_root,
            workers=per_source_workers,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        opened = dict(zip(("teacher", "v1"), executor.map(build, ("teacher", "v1"))))
    return {source: opened[source].index for source in ("teacher", "v1")}


def _run_canvas_analysis(
    splits: pd.DataFrame,
    development: pd.DataFrame,
    quality: BaselineEvaluation,
    indexes: dict[str, FeatureIndex],
    *,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, CanvasStressEvaluation]:
    primary, family = build_development_views(splits, validation_fold=1)
    query_rows = (
        primary.queries.loc[:, ["id"]]
        .merge(
            development.loc[:, ["id", "partition", "external_path"]],
            on="id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("id")
        .reset_index(drop=True)
    )
    canvas_workers = max(1, min(workers, 8))
    canvas_indexes = {
        orientation: extract_canvas_feature_index(
            query_rows,
            source="v1",
            path_column="external_path",
            orientation=orientation,
            contract=CONTRACT,
            root=ROOT,
            workers=canvas_workers,
            extract=extract_hog,
            method=HOG_METHOD,
            descriptor_fingerprint=HOG_CONFIG_FINGERPRINT,
            fold=1,
        )
        for orientation in ("wide", "tall")
    }
    canvas = evaluate_canvas_stress(
        quality.pair_evaluations[("v1", "v1")],
        canvas_indexes,
        indexes["v1"],
        primary,
        fold=1,
    )
    membership = mark_failure_slices(build_query_support(primary, family))
    ordinary = summarize_failure_slices(quality.query_metrics, membership)
    slices = pd.concat(
        [
            ordinary,
            baseline_runner._canvas_failure_summary(
                canvas.summary,
                canvas.per_query,
            ),
        ],
        ignore_index=True,
    )
    return membership, slices, canvas


def _benchmark(
    splits: pd.DataFrame,
    development: pd.DataFrame,
    indexes: dict[str, FeatureIndex],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary, _ = build_development_views(splits, validation_fold=1)
    query_rows = (
        primary.queries.loc[:, ["id"]]
        .merge(
            development.loc[:, ["id", "teacher_path", "external_path"]],
            on="id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("id")
        .reset_index(drop=True)
    )
    policy = TimingPolicy()
    samples: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for query_source, gallery_source in source_directions():
        path_column = SOURCE_SPECS[query_source][0]
        direction_samples = benchmark_source_direction(
            query_rows,
            query_source=query_source,
            gallery_source=gallery_source,
            encode=build_query_encoder(
                path_column=path_column,
                contract=CONTRACT,
                root=ROOT,
                extract=extract_hog,
            ),
            search=build_protocol_a_search(
                views=primary,
                gallery_index=indexes[gallery_source],
            ),
            policy=policy,
            fold=1,
        )
        samples.append(direction_samples)
        summaries.append(summarize_timings(direction_samples))
    return pd.concat(samples, ignore_index=True), pd.concat(
        summaries,
        ignore_index=True,
    )


def _measure_cost(
    splits: pd.DataFrame,
    development: pd.DataFrame,
    timing_summary: pd.DataFrame,
    *,
    workers: int,
) -> dict[str, object]:
    primary, _ = build_development_views(splits, validation_fold=1)
    gallery_rows = (
        primary.gallery.loc[:, ["id"]]
        .merge(
            development.loc[
                :,
                [
                    "id",
                    "partition",
                    "teacher_path",
                    "external_path",
                ],
            ],
            on="id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("id")
        .reset_index(drop=True)
    )
    index_costs = {
        source: measure_index_build(
            gallery_rows,
            source=source,
            path_column=path_column,
            contract=CONTRACT,
            root=ROOT,
            extract=extract_hog,
            method=HOG_METHOD,
            descriptor_fingerprint=HOG_CONFIG_FINGERPRINT,
            fold=1,
            workers=workers,
        )
        for source, (path_column, _) in SOURCE_SPECS.items()
    }
    cost = build_cost_record(
        timing_summary,
        index_costs,
        policy=TimingPolicy(),
    )
    cost.pop("probe_version", None)
    cost["method"] = HOG_METHOD
    cost["descriptor_config"] = HOG_CONFIG.to_dict()
    cost["descriptor_fingerprint"] = HOG_CONFIG_FINGERPRINT
    cost.update(_gate_verdict(cost))
    return cost


def _gate_verdict(cost: dict[str, object]) -> dict[str, bool]:
    timing = pd.DataFrame.from_records(cost.get("timing_summary", []))
    required = {
        "query_source",
        "gallery_source",
        "metric",
        "percentile",
        "value_seconds",
    }
    if missing := required.difference(timing.columns):
        raise ValueError(f"HOG timing summary is missing columns: {sorted(missing)}")
    p95 = timing.loc[
        timing["metric"].eq("end_to_end")
        & timing["percentile"].eq("p95")
    ]
    if set(
        p95.loc[:, ["query_source", "gallery_source"]].itertuples(
            index=False,
            name=None,
        )
    ) != set(source_directions()):
        raise ValueError("HOG cost requires four direction p95 values")
    source_costs = cost.get("per_source_index_cost")
    if not isinstance(source_costs, dict) or set(source_costs) != {
        "teacher",
        "v1",
    }:
        raise ValueError("HOG cost requires teacher and V1 indexes")
    return {
        "p95_end_to_end_under_one_second": bool(
            pd.to_numeric(p95["value_seconds"], errors="raise").lt(
                P95_LIMIT_SECONDS
            ).all()
        ),
        "index_under_one_gibibyte": bool(
            all(
                int(source_costs[source]["index_bytes"]) < INDEX_LIMIT_BYTES
                for source in ("teacher", "v1")
            )
        ),
    }


def _add_identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "scope" not in result:
        result.insert(0, "scope", "development")
    if "fold" not in result:
        result.insert(1, "fold", 1)
    if "method" not in result:
        result.insert(2, "method", HOG_METHOD)
    if "descriptor_fingerprint" not in result:
        result.insert(3, "descriptor_fingerprint", HOG_CONFIG_FINGERPRINT)
    return result


def _validate_semantic_outputs(
    *,
    summary: pd.DataFrame,
    timings: pd.DataFrame,
    method: str,
    descriptor_fingerprint: str,
    query_count: int,
) -> None:
    directions = set(source_directions())
    summary_directions = set(
        summary.loc[
            summary["query_source"].isin(("teacher", "v1"))
            & summary["gallery_source"].isin(("teacher", "v1")),
            ["query_source", "gallery_source"],
        ].itertuples(index=False, name=None)
    )
    if summary_directions != directions:
        raise ValueError("HOG quality evidence must contain all four directions")
    if set(summary["scope"].astype(str)) != {"development"} or set(
        pd.to_numeric(summary["fold"], errors="coerce")
    ) != {1}:
        raise ValueError("HOG quality evidence must be development fold 1")
    directional = summary.loc[
        summary["query_source"].isin(("teacher", "v1"))
        & summary["gallery_source"].isin(("teacher", "v1"))
    ]
    if set(directional["method"].astype(str)) != {method} or set(
        directional["descriptor_fingerprint"].astype(str)
    ) != {descriptor_fingerprint}:
        raise ValueError("HOG quality descriptor identity is malformed")
    timing_directions = set(
        timings.loc[
            :, ["query_source", "gallery_source"]
        ].itertuples(index=False, name=None)
    )
    if timing_directions != directions:
        raise ValueError("HOG timings must contain all four directions")
    counts = timings.groupby(["query_source", "gallery_source"])["query_id"].agg(
        ["size", "nunique"]
    )
    if not counts.eq(query_count).all().all():
        raise ValueError("HOG timings must contain every fold-1 query exactly once")


@contextmanager
def _unchanged_file(path: str | Path):
    target = Path(path)
    before_exists = target.exists()
    before = target.read_bytes() if before_exists else None
    try:
        yield
    finally:
        after_exists = target.exists()
        after = target.read_bytes() if after_exists else None
        if (before_exists, before) != (after_exists, after):
            raise RuntimeError("untrained HOG comparison must not modify results/runs.csv")


def _build_manifest(
    *,
    split_fingerprint: str,
    source_manifests: dict[str, dict[str, object]],
    artifacts: dict[str, Path],
    cost: dict[str, object],
) -> dict[str, object]:
    if set(source_manifests) != {"teacher", "v1"}:
        raise ValueError("HOG manifest requires teacher and V1 source manifests")
    return {
        "schema_version": "1.0.0",
        "scope": "development",
        "fold": 1,
        "method": HOG_METHOD,
        "descriptor_config": HOG_CONFIG.to_dict(),
        "descriptor_fingerprint": HOG_CONFIG_FINGERPRINT,
        "checkpoint_sha256": None,
        "split_fingerprint": split_fingerprint,
        "preprocessing_contract": CONTRACT.to_dict(),
        "source_identity": {
            source: {
                "source_fingerprint": source_manifests[source][
                    "source_fingerprint"
                ],
                "image_cache_manifest_sha256": hashlib.sha256(
                    json.dumps(
                        source_manifests[source],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
            for source in ("teacher", "v1")
        },
        "gates": _gate_verdict(cost),
        "registry_appended": False,
        "holdout_opened": False,
        "quarantine_opened": False,
        "artifacts": {
            name: {
                "path": str(path.relative_to(ROOT))
                if path.is_relative_to(ROOT)
                else str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sorted(artifacts.items())
        },
    }


def _write_hog_artifacts(
    *,
    quality: BaselineEvaluation,
    slices: pd.DataFrame,
    canvas: CanvasStressEvaluation,
    timings: pd.DataFrame,
    timing_summary: pd.DataFrame,
    cost: dict[str, object],
    examples: pd.DataFrame,
    image_rows: pd.DataFrame,
    example_rankings: dict[str, pd.DataFrame],
    split_fingerprint: str,
    source_manifests: dict[str, dict[str, object]],
) -> Path:
    evidence_dir = ROOT / "results/evidence/task4/hog"
    figure_dir = ROOT / "results/figures/task4/hog"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "quality_summary": _add_identity(quality.summary),
        "query_metrics": _add_identity(quality.query_metrics),
        "failure_slices": _add_identity(slices),
        "canvas_summary": _add_identity(canvas.summary),
        "canvas_per_query": _add_identity(canvas.per_query),
        "timing_samples": _add_identity(timings),
        "timing_summary": _add_identity(timing_summary),
        "examples": _add_identity(examples),
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = evidence_dir / f"{name}.csv"
        write_deterministic_csv(
            frame,
            path,
            index=False,
            float_format="%.8f",
        )
        paths[name] = path
    cost_path = evidence_dir / "cost.json"
    cost_path.write_text(
        json.dumps(cost, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["cost"] = cost_path
    render_baseline_examples(
        examples,
        image_rows,
        example_rankings,
        figure_dir / "hog_examples.png",
        title="Task 4 HOG — V1 query to V1 Top-5",
    )
    paths["examples_figure"] = figure_dir / "hog_examples.png"
    manifest = _build_manifest(
        split_fingerprint=split_fingerprint,
        source_manifests=source_manifests,
        artifacts=paths,
        cost=cost,
    )
    manifest_path = evidence_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    reopened_summary = pd.read_csv(paths["quality_summary"])
    reopened_timings = pd.read_csv(paths["timing_samples"])
    query_count = int(
        reopened_timings.groupby(["query_source", "gallery_source"])[
            "query_id"
        ].nunique().min()
    )
    _validate_semantic_outputs(
        summary=reopened_summary,
        timings=reopened_timings,
        method=HOG_METHOD,
        descriptor_fingerprint=HOG_CONFIG_FINGERPRINT,
        query_count=query_count,
    )
    reopened_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if reopened_manifest != manifest:
        raise ValueError("HOG manifest changed after writing")
    return manifest_path


def run_hog_comparison(*, workers: int) -> Path:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    with _unchanged_file(ROOT / "results/runs.csv"):
        _progress("inputs", "start")
        splits, development = _load_development_sources()
        caches = _load_image_caches(development)
        _progress("inputs", "complete")
        _progress("feature-indexes", "start")
        indexes = _open_hog_indexes(
            caches,
            workers=workers,
            cache_root=ROOT / "results/cache/task4/hog/features",
        )
        _progress("feature-indexes", "complete")
        _progress("quality", "start")
        with _thread_limit(workers):
            quality = evaluate_baseline(
                splits,
                indexes,
                fold=1,
                method=HOG_METHOD,
            )
        _progress("quality", "complete")
        _progress("canvas-and-slices", "start")
        with _thread_limit(workers):
            membership, slices, canvas = _run_canvas_analysis(
                splits,
                development,
                quality,
                indexes,
                workers=workers,
            )
        _progress("canvas-and-slices", "complete")
        _progress("timing", "start")
        with _thread_limit(1):
            timings, timing_summary = _benchmark(
                splits,
                development,
                indexes,
            )
        _progress("timing", "complete")
        _progress("cost", "start")
        cost = _measure_cost(
            splits,
            development,
            timing_summary,
            workers=workers,
        )
        _progress("cost", "complete")
        examples, example_rankings = baseline_runner._build_examples(
            quality,
            membership,
            canvas,
        )
        primary, _ = build_development_views(splits, validation_fold=1)
        baseline_runner._validate_run_outputs(
            quality,
            slices,
            timings,
            query_count=len(primary.queries),
        )
        _validate_semantic_outputs(
            summary=_add_identity(quality.summary),
            timings=_add_identity(timings),
            method=HOG_METHOD,
            descriptor_fingerprint=HOG_CONFIG_FINGERPRINT,
            query_count=len(primary.queries),
        )
        _progress("evidence", "start")
        manifest = _write_hog_artifacts(
            quality=quality,
            slices=slices,
            canvas=canvas,
            timings=timings,
            timing_summary=timing_summary,
            cost=cost,
            examples=examples,
            image_rows=development,
            example_rankings=example_rankings,
            split_fingerprint=cv_assignment_digest(splits),
            source_manifests={
                source: caches[source].manifest for source in ("teacher", "v1")
            },
        )
        _progress("evidence", "complete")
        return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_hog_comparison(workers=args.workers)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
