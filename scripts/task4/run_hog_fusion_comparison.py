import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mla2-task4-hog-fusion/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mla2-task4-hog-fusion/xdg")

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

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
from fashion.task4.hog_fusion import (
    HOG_FUSION_CONFIG,
    HOG_FUSION_CONFIG_FINGERPRINT,
    HOG_FUSION_METHOD,
    ParentFeatureCache,
    ensure_fused_feature_index,
    extract_hog_fusion,
    load_parent_feature_cache,
)
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import (
    CachedFeatureIndex,
    FeatureIndex,
    extract_canvas_feature_index,
    source_directions,
)
from fashion.task4.protocol import build_development_views

_BASELINE_RUNNER_PATH = ROOT / "scripts/task4/run_baseline.py"
_BASELINE_SPEC = importlib.util.spec_from_file_location(
    "_task4_hog_fusion_baseline_runner",
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
    print(f"task4-hog-fusion stage={stage} event={event}", flush=True)


@contextmanager
def _thread_limit(count: int):
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("thread limit must be a positive integer")
    with threadpool_limits(limits=count):
        yield


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded development-only Task 4 HOG fusion comparison."
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
    canonical = splits.loc[splits["partition"].eq("development"), ["id", "sha256"]].copy()
    development = variants.loc[variants["partition"].eq("development")].copy()
    canonical["id"] = pd.to_numeric(canonical["id"], errors="raise").astype(np.int64)
    development["id"] = pd.to_numeric(development["id"], errors="raise").astype(np.int64)
    if development["id"].duplicated().any():
        raise ValueError("fusion variants require unique development IDs")
    if set(development["id"]) != set(canonical["id"]):
        raise ValueError("fusion variants must exactly match canonical development split IDs")
    development["teacher_sha256"] = development["id"].map(canonical.set_index("id")["sha256"])
    for column in (
        "teacher_path",
        "external_path",
        "teacher_sha256",
        "external_sha256",
    ):
        if development[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"fusion development sources have blank {column}")
    return development.sort_values("id").reset_index(drop=True)


def _load_development_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = load_splits()
    variants = pd.read_csv(
        ROOT / "data/processed/task4/external_variant_index.csv.gz",
        usecols=list(VARIANT_COLUMNS),
        keep_default_na=False,
    )
    return splits, _select_development_sources(splits, variants)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _find_hog_parent(source: str) -> ParentFeatureCache:
    root = ROOT / "results/cache/task4/hog/features" / source
    candidates: list[ParentFeatureCache] = []
    for directory in sorted(path for path in root.glob("*") if path.is_dir()):
        try:
            cache = load_parent_feature_cache(directory, component="hog")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if (
            cache.index.source == source
            and cache.manifest.get("scope") == "development"
            and cache.manifest.get("fold") == 1
            and cache.manifest.get("method") == HOG_FUSION_CONFIG.hog_method
            and cache.manifest.get("descriptor_fingerprint")
            == HOG_FUSION_CONFIG.hog_descriptor_fingerprint
        ):
            candidates.append(cache)
    if len(candidates) != 1:
        raise ValueError(
            f"fusion requires exactly one canonical HOG cache for {source}; found {len(candidates)}"
        )
    return candidates[0]


def _open_parent_and_fused_indexes(
    development: pd.DataFrame,
) -> tuple[
    dict[str, FeatureIndex],
    dict[str, dict[str, ParentFeatureCache]],
    dict[str, CachedFeatureIndex],
]:
    expected_ids = development["id"].to_numpy(dtype=np.int64)
    spatial_root = ROOT / "data/processed/task4/preprocessing/features" / CONTRACT.key
    parents: dict[str, dict[str, ParentFeatureCache]] = {}
    fused: dict[str, CachedFeatureIndex] = {}
    for source in ("teacher", "v1"):
        hog_cache = _find_hog_parent(source)
        spatial_cache = load_parent_feature_cache(
            spatial_root / source,
            component="spatial_hsv_edge",
        )
        if not np.array_equal(hog_cache.index.ids, expected_ids):
            raise ValueError("HOG parent IDs do not match exact development coverage")
        if not np.array_equal(spatial_cache.index.ids, expected_ids):
            raise ValueError("HSV-edge parent IDs do not match exact development coverage")
        parents[source] = {
            "hog": hog_cache,
            "spatial_hsv_edge": spatial_cache,
        }
        fused[source] = ensure_fused_feature_index(
            hog_cache,
            spatial_cache,
            cache_root=ROOT / "results/cache/task4/hog_fusion/features",
        )
        if not np.array_equal(fused[source].index.ids, expected_ids):
            raise ValueError("fused cache IDs do not match exact development coverage")
    return (
        {source: fused[source].index for source in ("teacher", "v1")},
        parents,
        fused,
    )


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
    canvas_workers = max(1, min(workers, 16))
    canvas_indexes = {
        orientation: extract_canvas_feature_index(
            query_rows,
            source="v1",
            path_column="external_path",
            orientation=orientation,
            contract=CONTRACT,
            root=ROOT,
            workers=canvas_workers,
            extract=extract_hog_fusion,
            method=HOG_FUSION_METHOD,
            descriptor_fingerprint=HOG_FUSION_CONFIG_FINGERPRINT,
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
                extract=extract_hog_fusion,
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
            extract=extract_hog_fusion,
            method=HOG_FUSION_METHOD,
            descriptor_fingerprint=HOG_FUSION_CONFIG_FINGERPRINT,
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
    cost["method"] = HOG_FUSION_METHOD
    cost["descriptor_config"] = HOG_FUSION_CONFIG.to_dict()
    cost["descriptor_fingerprint"] = HOG_FUSION_CONFIG_FINGERPRINT
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
        raise ValueError(f"fusion timing summary is missing columns: {sorted(missing)}")
    p95 = timing.loc[timing["metric"].eq("end_to_end") & timing["percentile"].eq("p95")]
    if set(
        p95.loc[:, ["query_source", "gallery_source"]].itertuples(
            index=False,
            name=None,
        )
    ) != set(source_directions()):
        raise ValueError("fusion cost requires four direction p95 values")
    source_costs = cost.get("per_source_index_cost")
    if not isinstance(source_costs, dict) or set(source_costs) != {
        "teacher",
        "v1",
    }:
        raise ValueError("fusion cost requires teacher and V1 indexes")
    return {
        "p95_end_to_end_under_one_second": bool(
            pd.to_numeric(p95["value_seconds"], errors="raise").lt(P95_LIMIT_SECONDS).all()
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
        result.insert(2, "method", HOG_FUSION_METHOD)
    if "descriptor_fingerprint" not in result:
        result.insert(3, "descriptor_fingerprint", HOG_FUSION_CONFIG_FINGERPRINT)
    return result


def _validate_semantic_outputs(
    *,
    summary: pd.DataFrame,
    timings: pd.DataFrame,
    method: str,
    descriptor_fingerprint: str,
    expected_query_ids: tuple[int, ...] | list[int] | np.ndarray,
    query_metrics: pd.DataFrame | None = None,
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
        raise ValueError("fusion quality evidence must contain all four directions")
    if set(summary["scope"].astype(str)) != {"development"} or set(
        pd.to_numeric(summary["fold"], errors="coerce")
    ) != {1}:
        raise ValueError("fusion quality evidence must be development fold 1")
    directional = summary.loc[
        summary["query_source"].isin(("teacher", "v1"))
        & summary["gallery_source"].isin(("teacher", "v1"))
    ]
    if set(directional["method"].astype(str)) != {method} or set(
        directional["descriptor_fingerprint"].astype(str)
    ) != {descriptor_fingerprint}:
        raise ValueError("fusion quality descriptor identity is malformed")
    timing_directions = set(
        timings.loc[:, ["query_source", "gallery_source"]].itertuples(
            index=False,
            name=None,
        )
    )
    if timing_directions != directions:
        raise ValueError("fusion timings must contain all four directions")
    expected = tuple(int(value) for value in expected_query_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("expected query IDs must be unique")
    expected_set = set(expected)
    for direction in source_directions():
        observed = timings.loc[
            timings["query_source"].eq(direction[0]) & timings["gallery_source"].eq(direction[1]),
            "query_id",
        ]
        numeric = pd.to_numeric(observed, errors="coerce")
        if (
            numeric.isna().any()
            or len(numeric) != len(expected)
            or numeric.nunique() != len(expected)
            or set(numeric.astype(int)) != expected_set
        ):
            raise ValueError("fusion timing query coverage is incomplete or duplicated")
    if query_metrics is not None:
        for direction in source_directions():
            for protocol in ("primary", "family"):
                observed = query_metrics.loc[
                    query_metrics["query_source"].eq(direction[0])
                    & query_metrics["gallery_source"].eq(direction[1])
                    & query_metrics["protocol"].eq(protocol),
                    "query_id",
                ]
                numeric = pd.to_numeric(observed, errors="coerce")
                if (
                    numeric.isna().any()
                    or len(numeric) != len(expected)
                    or numeric.nunique() != len(expected)
                    or set(numeric.astype(int)) != expected_set
                ):
                    raise ValueError("fusion quality query coverage is incomplete or duplicated")


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
            raise RuntimeError("untrained HOG fusion comparison must not modify results/runs.csv")


def _build_manifest(
    *,
    split_fingerprint: str,
    fused_cache_manifests: dict[str, dict[str, object]],
    parent_manifests: dict[str, dict[str, object]],
    artifacts: dict[str, Path],
    cost: dict[str, object],
    artifact_public_paths: dict[str, str] | None = None,
) -> dict[str, object]:
    if set(fused_cache_manifests) != {"teacher", "v1"}:
        raise ValueError("fusion manifest requires teacher and V1 fused caches")
    if set(parent_manifests) != {"teacher", "v1"}:
        raise ValueError("fusion manifest requires teacher and V1 parents")
    for source in ("teacher", "v1"):
        if set(parent_manifests[source]) != {"hog", "spatial_hsv_edge"}:
            raise ValueError("fusion manifest requires both component parents")
    public = artifact_public_paths or {name: _relative(path) for name, path in artifacts.items()}
    if set(public) != set(artifacts):
        raise ValueError("fusion public artifact paths do not match artifacts")
    return {
        "schema_version": "1.0.0",
        "scope": "development",
        "fold": 1,
        "method": HOG_FUSION_METHOD,
        "descriptor_config": HOG_FUSION_CONFIG.to_dict(),
        "descriptor_fingerprint": HOG_FUSION_CONFIG_FINGERPRINT,
        "checkpoint_sha256": None,
        "config_fingerprint": None,
        "split_fingerprint": split_fingerprint,
        "preprocessing_contract": CONTRACT.to_dict(),
        "parents": parent_manifests,
        "fused_caches": fused_cache_manifests,
        "gates": _gate_verdict(cost),
        "registry_appended": False,
        "holdout_opened": False,
        "quarantine_opened": False,
        "artifacts": {
            name: {
                "path": public[name],
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sorted(artifacts.items())
        },
    }


def _publish_package_atomically(
    destination: Path,
    build: Callable[[Path], None],
) -> None:
    """Build and validate off-path, then replace one complete package."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-staging-", dir=destination.parent))
    backup = destination.with_name(f".{destination.name}-backup-{uuid.uuid4().hex}")
    moved_old = False
    try:
        build(staging)
        if not (staging / "manifest.json").is_file():
            raise ValueError("staged package has no validated manifest")
        if destination.exists():
            os.replace(destination, backup)
            moved_old = True
        os.replace(staging, destination)
        if moved_old:
            shutil.rmtree(backup)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if moved_old and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        else:
            shutil.rmtree(backup, ignore_errors=True)
        raise


def _parent_records(
    parents: dict[str, dict[str, ParentFeatureCache]],
) -> dict[str, dict[str, object]]:
    return {
        source: {
            component: {
                "path": _relative(cache.cache_dir / "manifest.json"),
                "sha256": cache.manifest_sha256,
                "ids_file_sha256": cache.ids_file_sha256,
                "features_file_sha256": cache.features_file_sha256,
                "manifest": cache.manifest,
            }
            for component, cache in parents[source].items()
        }
        for source in ("teacher", "v1")
    }


def _fused_cache_records(
    fused: dict[str, CachedFeatureIndex],
) -> dict[str, dict[str, object]]:
    return {
        source: {
            "path": _relative(fused[source].cache_dir / "manifest.json"),
            "sha256": _sha256_file(fused[source].cache_dir / "manifest.json"),
            "manifest": fused[source].manifest,
        }
        for source in ("teacher", "v1")
    }


def _write_fusion_artifacts(
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
    parents: dict[str, dict[str, ParentFeatureCache]],
    fused: dict[str, CachedFeatureIndex],
    expected_query_ids: np.ndarray,
) -> Path:
    evidence_dir = ROOT / "results/evidence/task4/hog_fusion"
    figure_dir = ROOT / "results/figures/task4/hog_fusion"
    evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.parent.mkdir(parents=True, exist_ok=True)
    evidence_staging = Path(
        tempfile.mkdtemp(prefix=".hog-fusion-evidence-", dir=evidence_dir.parent)
    )
    figure_staging = Path(tempfile.mkdtemp(prefix=".hog-fusion-figure-", dir=figure_dir.parent))
    evidence_backup = evidence_dir.with_name(f".{evidence_dir.name}-backup-{uuid.uuid4().hex}")
    figure_backup = figure_dir.with_name(f".{figure_dir.name}-backup-{uuid.uuid4().hex}")
    moved_evidence = False
    moved_figure = False
    try:
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
        public_paths: dict[str, str] = {}
        for name, frame in frames.items():
            path = evidence_staging / f"{name}.csv"
            write_deterministic_csv(
                frame,
                path,
                index=False,
                float_format="%.8f",
            )
            paths[name] = path
            public_paths[name] = _relative(evidence_dir / path.name)
        cost_path = evidence_staging / "cost.json"
        cost_path.write_text(
            json.dumps(cost, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths["cost"] = cost_path
        public_paths["cost"] = _relative(evidence_dir / cost_path.name)
        figure_path = figure_staging / "hog_fusion_examples.png"
        render_baseline_examples(
            examples,
            image_rows,
            example_rankings,
            figure_path,
            title="Task 4 equal HOG + HSV-edge fusion — V1 query to V1 Top-5",
        )
        paths["examples_figure"] = figure_path
        public_paths["examples_figure"] = _relative(figure_dir / "hog_fusion_examples.png")
        manifest = _build_manifest(
            split_fingerprint=split_fingerprint,
            fused_cache_manifests=_fused_cache_records(fused),
            parent_manifests=_parent_records(parents),
            artifacts=paths,
            artifact_public_paths=public_paths,
            cost=cost,
        )
        manifest_path = evidence_staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        reopened_summary = pd.read_csv(paths["quality_summary"])
        reopened_query_metrics = pd.read_csv(paths["query_metrics"])
        reopened_timings = pd.read_csv(paths["timing_samples"])
        _validate_semantic_outputs(
            summary=reopened_summary,
            timings=reopened_timings,
            query_metrics=reopened_query_metrics,
            method=HOG_FUSION_METHOD,
            descriptor_fingerprint=HOG_FUSION_CONFIG_FINGERPRINT,
            expected_query_ids=expected_query_ids,
        )
        reopened_cost = json.loads(cost_path.read_text(encoding="utf-8"))
        if reopened_cost != cost:
            raise ValueError("fusion cost artifact changed after writing")
        reopened_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if reopened_manifest != manifest:
            raise ValueError("fusion manifest changed after writing")
        for record in manifest["artifacts"].values():
            public_path = ROOT / str(record["path"])
            staged_path = next(
                path for name, path in paths.items() if public_paths[name] == str(record["path"])
            )
            if (
                record["sha256"] != _sha256_file(staged_path)
                or record["bytes"] != staged_path.stat().st_size
                or not str(public_path).startswith(str(ROOT))
            ):
                raise ValueError("fusion staged artifact identity is malformed")

        if figure_dir.exists():
            os.replace(figure_dir, figure_backup)
            moved_figure = True
        os.replace(figure_staging, figure_dir)
        if evidence_dir.exists():
            os.replace(evidence_dir, evidence_backup)
            moved_evidence = True
        os.replace(evidence_staging, evidence_dir)
        final_manifest = evidence_dir / "manifest.json"
        if json.loads(final_manifest.read_text(encoding="utf-8")) != manifest:
            raise ValueError("published fusion manifest failed semantic reopening")
        for record in manifest["artifacts"].values():
            artifact = ROOT / str(record["path"])
            if (
                not artifact.is_file()
                or artifact.stat().st_size != int(record["bytes"])
                or _sha256_file(artifact) != record["sha256"]
            ):
                raise ValueError("published fusion artifact failed its identity")
        shutil.rmtree(evidence_backup, ignore_errors=True)
        shutil.rmtree(figure_backup, ignore_errors=True)
        return final_manifest
    except BaseException:
        shutil.rmtree(evidence_staging, ignore_errors=True)
        shutil.rmtree(figure_staging, ignore_errors=True)
        if moved_evidence and evidence_backup.exists():
            shutil.rmtree(evidence_dir, ignore_errors=True)
            os.replace(evidence_backup, evidence_dir)
        if moved_figure and figure_backup.exists():
            shutil.rmtree(figure_dir, ignore_errors=True)
            os.replace(figure_backup, figure_dir)
        raise


def run_hog_fusion_comparison(*, workers: int) -> Path:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    with _unchanged_file(ROOT / "results/runs.csv"):
        _progress("inputs", "start")
        splits, development = _load_development_sources()
        _progress("inputs", "complete")
        _progress("parent-and-fused-indexes", "start")
        indexes, parents, fused = _open_parent_and_fused_indexes(development)
        _progress("parent-and-fused-indexes", "complete")
        _progress("quality", "start")
        with _thread_limit(workers):
            quality = evaluate_baseline(
                splits,
                indexes,
                fold=1,
                method=HOG_FUSION_METHOD,
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
        expected_query_ids = (
            pd.to_numeric(primary.queries["id"], errors="raise")
            .astype(np.int64)
            .sort_values()
            .to_numpy()
        )
        baseline_runner._validate_run_outputs(
            quality,
            slices,
            timings,
            query_count=len(expected_query_ids),
        )
        _validate_semantic_outputs(
            summary=_add_identity(quality.summary),
            timings=_add_identity(timings),
            query_metrics=_add_identity(quality.query_metrics),
            method=HOG_FUSION_METHOD,
            descriptor_fingerprint=HOG_FUSION_CONFIG_FINGERPRINT,
            expected_query_ids=expected_query_ids,
        )
        _progress("evidence", "start")
        manifest = _write_fusion_artifacts(
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
            parents=parents,
            fused=fused,
            expected_query_ids=expected_query_ids,
        )
        _progress("evidence", "complete")
        return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_hog_fusion_comparison(workers=args.workers)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
