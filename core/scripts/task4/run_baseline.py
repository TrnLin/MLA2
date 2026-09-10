import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mla2-task4-cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mla2-task4-cache/xdg")

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data.dataset import load_splits
from fashion.task4.analysis import (
    CanvasStressEvaluation,
    build_query_support,
    evaluate_canvas_stress,
    mark_failure_slices,
    select_example_ids,
    summarize_failure_slices,
)
from fashion.task4.baseline import (
    BaselineEvaluation,
    evaluate_baseline,
    verify_preprocessing_reproduction,
)
from fashion.task4.baseline_evidence import (
    EXAMPLE_COLUMNS,
    FAILURE_SLICE_COLUMNS,
    TIMING_COLUMNS,
    write_baseline_artifacts,
)
from fashion.task4.benchmark import (
    TimingPolicy,
    benchmark_source_direction,
    build_cost_record,
    build_protocol_a_search,
    build_query_encoder,
    measure_index_build,
    summarize_timings,
)
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import (
    FeatureIndex,
    ensure_feature_index,
    extract_canvas_feature_index,
    source_directions,
)
from fashion.task4.protocol import build_development_views

SELECTED_CONTRACT = PreprocessingContract(width=240, height=320)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild development-only Task 4 baseline evidence."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="parallel cache workers; timed work remains single-threaded",
    )
    return parser


def _load_development_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = load_splits()
    variant_path = ROOT / "data/processed/task4/external_variant_index.csv.gz"
    variant = pd.read_csv(
        variant_path,
        usecols=list(VARIANT_COLUMNS),
        keep_default_na=False,
    )
    development = variant.loc[variant["partition"].eq("development")].copy()
    if development.empty or not development["partition"].eq("development").all():
        raise ValueError("baseline variants must contain development rows only")
    if development["id"].duplicated().any():
        raise ValueError("baseline variant IDs must be unique")
    development["id"] = pd.to_numeric(
        development["id"], errors="raise"
    ).astype(np.int64)
    development_ids = set(
        splits.loc[splits["partition"].eq("development"), "id"].astype(int)
    )
    if set(development["id"]) != development_ids:
        raise ValueError("V1 variants do not exactly match canonical development IDs")
    hashes = splits.loc[
        splits["partition"].eq("development"), ["id", "sha256"]
    ].set_index("id")["sha256"]
    development["teacher_sha256"] = development["id"].map(hashes)
    for column in (
        "teacher_path",
        "external_path",
        "teacher_sha256",
        "external_sha256",
    ):
        if development[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"baseline development variants have blank {column}")
    return splits, development.sort_values("id").reset_index(drop=True)


def _open_feature_indexes(
    development: pd.DataFrame,
    *,
    workers: int,
) -> dict[str, FeatureIndex]:
    cache_root = ROOT / "data/processed/task4/preprocessing/features"
    return {
        source: ensure_feature_index(
            development,
            path_column=path_column,
            sha_column=sha_column,
            source=source,
            contract=SELECTED_CONTRACT,
            cache_root=cache_root,
            root=ROOT,
            workers=workers,
        ).index
        for source, (path_column, sha_column) in SOURCE_SPECS.items()
    }


def _canvas_failure_summary(
    canvas_summary: pd.DataFrame,
    canvas_per_query: pd.DataFrame,
) -> pd.DataFrame:
    context = [
        "scope",
        "fold",
        "size",
        "query_source",
        "gallery_source",
        "query_variant",
    ]
    required = {*context, "query_id", "canvas_ndcg_at_10"}
    if missing := required.difference(canvas_per_query.columns):
        raise ValueError(f"canvas per-query evidence is missing columns: {sorted(missing)}")
    numeric = pd.to_numeric(canvas_per_query["canvas_ndcg_at_10"], errors="coerce")
    finite = numeric.notna() & np.isfinite(numeric)
    counted = canvas_per_query.assign(_scored=finite).groupby(
        context,
        sort=False,
        observed=True,
        dropna=False,
    ).agg(
        observed_queries=("query_id", "size"),
        unique_queries=("query_id", "nunique"),
        scored_queries=("_scored", "sum"),
    )
    if not counted["observed_queries"].eq(counted["unique_queries"]).all():
        raise ValueError("canvas per-query evidence contains duplicate query IDs")
    counted["excluded_queries"] = (
        counted["observed_queries"] - counted["scored_queries"]
    )
    counted["coverage"] = counted["scored_queries"] / counted["observed_queries"]
    counted = counted.reset_index()

    rows = canvas_summary.rename(
        columns={
            "queries": "summary_queries",
            "ndcg_at_10": "value",
        }
    ).merge(counted, on=context, validate="one_to_one")
    if not rows["summary_queries"].eq(rows["observed_queries"]).all():
        raise ValueError("canvas summary and per-query counts disagree")
    rows = rows.rename(
        columns={
            "query_variant": "_query_variant",
            "observed_queries": "total_queries",
        }
    )
    rows["protocol"] = "primary"
    rows["slice"] = "canvas_" + rows["_query_variant"].astype(str)
    rows["metric"] = "ndcg"
    rows["k"] = 10
    rows["aggregation"] = "query_mean"
    return rows.loc[:, FAILURE_SLICE_COLUMNS]


def _run_canvas_analysis(
    splits: pd.DataFrame,
    development: pd.DataFrame,
    baseline: BaselineEvaluation,
    indexes: dict[str, FeatureIndex],
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
    canvas_indexes = {
        orientation: extract_canvas_feature_index(
            query_rows,
            source="v1",
            path_column="external_path",
            orientation=orientation,
            contract=SELECTED_CONTRACT,
            root=ROOT,
            workers=1,
        )
        for orientation in ("wide", "tall")
    }
    canvas = evaluate_canvas_stress(
        baseline.pair_evaluations[("v1", "v1")],
        canvas_indexes,
        indexes["v1"],
        primary,
        fold=1,
    )
    membership = mark_failure_slices(build_query_support(primary, family))
    ordinary = summarize_failure_slices(baseline.query_metrics, membership)
    slices = pd.concat(
        [ordinary, _canvas_failure_summary(canvas.summary, canvas.per_query)],
        ignore_index=True,
    ).loc[:, FAILURE_SLICE_COLUMNS]
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
            development.loc[
                :, ["id", "teacher_path", "external_path"]
            ],
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
                contract=SELECTED_CONTRACT,
                root=ROOT,
            ),
            search=build_protocol_a_search(
                views=primary,
                gallery_index=indexes[gallery_source],
            ),
            policy=policy,
        )
        samples.append(direction_samples)
        summaries.append(summarize_timings(direction_samples))
    return (
        pd.concat(samples, ignore_index=True),
        pd.concat(summaries, ignore_index=True),
    )


def _measure_cost(
    splits: pd.DataFrame,
    development: pd.DataFrame,
    timing_summary: pd.DataFrame,
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
            contract=SELECTED_CONTRACT,
            root=ROOT,
        )
        for source, (path_column, _) in SOURCE_SPECS.items()
    }
    return build_cost_record(
        timing_summary,
        index_costs,
        policy=TimingPolicy(),
    )


def _score_for_example(
    baseline: BaselineEvaluation,
    canvas: CanvasStressEvaluation,
    *,
    slice_name: str,
    query_id: int,
) -> tuple[str, float, str, pd.DataFrame]:
    if slice_name == "canvas_failure":
        per_query = canvas.per_query.loc[
            canvas.per_query["query_id"].eq(query_id)
            & ~canvas.per_query["query_variant"].eq("clean")
        ].copy()
        variants = pd.Categorical(
            per_query["query_variant"],
            categories=["wide", "tall"],
            ordered=True,
        )
        selected = (
            per_query.assign(_variant_order=variants)
            .sort_values(
                ["ndcg_change_from_clean", "_variant_order"],
                kind="mergesort",
            )
            .iloc[0]
        )
        variant = str(selected["query_variant"])
        return (
            "ndcg_change_from_clean",
            float(selected["ndcg_change_from_clean"]),
            variant,
            canvas.rankings[variant],
        )

    protocol = (
        "family"
        if slice_name in {"family_unavailable", "weak_family"}
        else "primary"
    )
    metric = "recall_at_10" if protocol == "family" else "ndcg_at_10"
    query_metrics = baseline.query_metrics.loc[
        baseline.query_metrics["query_source"].eq("v1")
        & baseline.query_metrics["gallery_source"].eq("v1")
        & baseline.query_metrics["protocol"].eq(protocol)
        & baseline.query_metrics["query_id"].eq(query_id)
    ]
    if len(query_metrics) != 1:
        raise ValueError(f"example {slice_name} has malformed query metrics")
    value = query_metrics[metric].item()
    score = float(value) if pd.notna(value) else float("nan")
    pair = baseline.pair_evaluations[("v1", "v1")]
    rankings = (
        pair.family_rankings if protocol == "family" else pair.primary_rankings
    )
    return metric, score, "clean", rankings


def _build_examples(
    baseline: BaselineEvaluation,
    membership: pd.DataFrame,
    canvas: CanvasStressEvaluation,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    selected_ids = select_example_ids(
        baseline.query_metrics,
        membership,
        canvas.per_query,
    )
    required = {"normal_success", "grayscale", "weak_family", "canvas_failure"}
    if missing := sorted(required.difference(selected_ids)):
        raise ValueError(f"baseline examples are missing required slices: {missing}")

    records: list[dict[str, object]] = []
    example_rankings: dict[str, pd.DataFrame] = {}
    for slice_name, query_id in selected_ids.items():
        metric, value, query_variant, rankings = _score_for_example(
            baseline,
            canvas,
            slice_name=slice_name,
            query_id=query_id,
        )
        top_five = rankings.loc[
            rankings["query_id"].eq(query_id) & rankings["rank"].between(1, 5),
            ["query_id", "candidate_id", "distance", "rank"],
        ].sort_values("rank")
        if top_five["rank"].astype(int).tolist() != [1, 2, 3, 4, 5]:
            raise ValueError(f"baseline example {slice_name} has no complete Top-5")
        example_rankings[slice_name] = top_five.reset_index(drop=True)
        for result in top_five.itertuples(index=False):
            records.append(
                {
                    "scope": "development",
                    "fold": 1,
                    "size": "240x320",
                    "query_source": "v1",
                    "gallery_source": "v1",
                    "slice": slice_name,
                    "query_variant": query_variant,
                    "query_id": int(query_id),
                    "metric": metric,
                    "value": value,
                    "rank": int(result.rank),
                    "candidate_id": int(result.candidate_id),
                    "distance": float(result.distance),
                }
            )
    return pd.DataFrame.from_records(records, columns=EXAMPLE_COLUMNS), example_rankings


def _validate_run_outputs(
    baseline: BaselineEvaluation,
    slices: pd.DataFrame,
    timings: pd.DataFrame,
    *,
    query_count: int,
) -> None:
    if tuple(timings.columns) != TIMING_COLUMNS:
        raise ValueError(f"baseline timing columns must be {TIMING_COLUMNS}")
    if set(timings["scope"].astype(str)) != {"development"}:
        raise ValueError("baseline timing scope must be development")
    if set(pd.to_numeric(timings["fold"], errors="coerce")) != {1}:
        raise ValueError("baseline timings must use frozen fold 1")
    expected_directions = set(source_directions())
    baseline_directions = set(
        baseline.query_metrics.loc[
            :, ["query_source", "gallery_source"]
        ].itertuples(index=False, name=None)
    )
    if baseline_directions != expected_directions:
        raise ValueError("baseline quality does not contain all four directions")
    timing_directions = set(
        timings.loc[
            :, ["query_source", "gallery_source"]
        ].itertuples(index=False, name=None)
    )
    if timing_directions != expected_directions or len(timings) != 4 * query_count:
        raise ValueError("baseline timings do not contain every query in four directions")
    direction_counts = timings.groupby(
        ["query_source", "gallery_source"], sort=False
    )["query_id"].agg(["size", "nunique"])
    if not direction_counts.eq(query_count).all().all():
        raise ValueError("baseline timings have missing or duplicate query IDs")
    expected_slices = {
        "grayscale",
        "rare_article_type",
        "rare_type_colour",
        "unusual_geometry",
        "family_unavailable",
        "weak_family",
        "canvas_clean",
        "canvas_wide",
        "canvas_tall",
    }
    if set(slices["slice"]) != expected_slices:
        raise ValueError("baseline failure evidence is missing ordinary or canvas slices")


def run_baseline_evidence(*, workers: int) -> None:
    """Rebuild and validate all development-only Task 4 baseline artifacts."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be positive")
    splits, development = _load_development_sources()
    indexes = _open_feature_indexes(development, workers=workers)
    baseline = evaluate_baseline(splits, indexes, fold=1)
    comparison = pd.read_csv(
        ROOT / "results/evidence/task4/preprocessing_comparison.csv",
        keep_default_na=False,
    )
    verify_preprocessing_reproduction(baseline.summary, comparison)

    membership, slices, canvas = _run_canvas_analysis(
        splits,
        development,
        baseline,
        indexes,
    )
    timings, timing_summary = _benchmark(splits, development, indexes)
    cost = _measure_cost(splits, development, timing_summary)
    examples, example_rankings = _build_examples(baseline, membership, canvas)
    primary, _ = build_development_views(splits, validation_fold=1)
    _validate_run_outputs(
        baseline,
        slices,
        timings,
        query_count=len(primary.queries),
    )
    write_baseline_artifacts(
        baseline=baseline,
        slice_summary=slices,
        timings=timings,
        cost=cost,
        examples=examples,
        evidence_dir=ROOT / "results/evidence/task4",
        figure_dir=ROOT / "results/figures/task4",
        image_rows=development,
        example_rankings=example_rankings,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    run_baseline_evidence(workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
