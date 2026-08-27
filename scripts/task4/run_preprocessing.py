"""Rebuild Task 4 Milestone 3 development-only evidence and local caches."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mla2-task4-cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mla2-task4-cache/xdg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from fashion.config import ROOT
from fashion.data.dataset import load_splits
from fashion.data.hashing import write_deterministic_csv
from fashion.task4.cache import (
    ensure_development_image_cache,
    fit_cached_fold_rgb_statistics,
)
from fashion.task4.preprocessing import PreprocessingContract, preprocess_image
from fashion.task4.preprocessing_experiment import (
    FeatureIndex,
    build_odd_aspect_canvas,
    ensure_feature_index,
    evaluate_source_pair,
    resolve_size_policy,
    run_preprocessing_experiment,
)
from fashion.task4.probe import PROBE_VERSION, extract_spatial_probe
from fashion.task4.protocol import build_development_views

CANDIDATE_CONTRACTS = (
    PreprocessingContract(width=60, height=80),
    PreprocessingContract(width=96, height=128),
    PreprocessingContract(width=240, height=320),
)
SELECTED_CONTRACT = PreprocessingContract(width=240, height=320)
SOURCE_SPECS = {
    "teacher": ("teacher_path", "teacher_sha256"),
    "v1": ("external_path", "external_sha256"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild development-only Task 4 preprocessing evidence."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="parallel image decode/feature workers (default: up to 16)",
    )
    return parser


def _load_development_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = load_splits()
    variant = pd.read_csv(
        ROOT / "data/processed/task4/external_variant_index.csv.gz",
        keep_default_na=False,
    )
    development = variant.loc[variant["partition"].eq("development")].copy()
    development["teacher_sha256"] = development["id"].map(
        splits.set_index("id")["sha256"]
    )
    expected_ids = set(
        splits.loc[splits["partition"].eq("development"), "id"].astype(int)
    )
    if set(development["id"].astype(int)) != expected_ids:
        raise ValueError("V1 variants do not exactly match canonical development IDs")
    return splits, development


def _odd_query_index(
    query_rows: pd.DataFrame,
    *,
    source: str,
    path_column: str,
    orientation: str,
    contract: PreprocessingContract,
    workers: int,
) -> FeatureIndex:
    records = query_rows.to_dict("records")

    def transform(record: dict[str, object]) -> np.ndarray:
        with Image.open(ROOT / str(record[path_column])) as image:
            canvas = build_odd_aspect_canvas(image, orientation)
            transformed = preprocess_image(canvas, contract)
        return extract_spatial_probe(
            transformed.pixels,
            transformed.content_mask,
        )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        features = list(executor.map(transform, records))
    return FeatureIndex(
        source=source,
        contract=contract,
        ids=query_rows["id"].to_numpy(dtype=np.int64),
        features=np.stack(features).astype(np.float32, copy=False),
        transform_seconds=time.perf_counter() - started,
        source_bytes=sum(
            (ROOT / str(value)).stat().st_size for value in query_rows[path_column]
        ),
    )


def _robustness_evidence(
    splits: pd.DataFrame,
    development: pd.DataFrame,
    *,
    winner_size: str,
    winner_contract: PreprocessingContract,
    feature_indexes: dict[tuple[str, str], FeatureIndex],
    workers: int,
) -> pd.DataFrame:
    primary, family = build_development_views(splits, validation_fold=1)
    query_rows = (
        primary.queries[["id"]]
        .merge(
            development[["id", "teacher_path", "external_path"]],
            on="id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("id")
        .reset_index(drop=True)
    )
    records: list[dict[str, object]] = []
    for source, (path_column, _) in SOURCE_SPECS.items():
        gallery_index = feature_indexes[(winner_size, source)]
        clean = evaluate_source_pair(
            gallery_index,
            gallery_index,
            primary_views=primary,
            family_views=family,
            fold=1,
            chunk_size=256,
        )
        clean_ndcg = clean.summary.query(
            "protocol == 'primary' and metric == 'ndcg' "
            "and k == 10 and aggregation == 'query_mean'"
        )["value"].item()
        clean_sets = (
            clean.primary_rankings.loc[clean.primary_rankings["rank"].le(10)]
            .groupby("query_id")["candidate_id"]
            .agg(lambda values: set(values))
        )
        records.append(
            {
                "scope": "development",
                "source": source,
                "query_variant": "clean",
                "queries": len(query_rows),
                "ndcg_at_10": clean_ndcg,
                "ndcg_change_from_clean": 0.0,
                "mean_top10_overlap": 1.0,
            }
        )
        for orientation in ("wide", "tall"):
            odd_index = _odd_query_index(
                query_rows,
                source=source,
                path_column=path_column,
                orientation=orientation,
                contract=winner_contract,
                workers=workers,
            )
            odd = evaluate_source_pair(
                odd_index,
                gallery_index,
                primary_views=primary,
                family_views=family,
                fold=1,
                chunk_size=256,
            )
            odd_ndcg = odd.summary.query(
                "protocol == 'primary' and metric == 'ndcg' "
                "and k == 10 and aggregation == 'query_mean'"
            )["value"].item()
            odd_sets = (
                odd.primary_rankings.loc[odd.primary_rankings["rank"].le(10)]
                .groupby("query_id")["candidate_id"]
                .agg(lambda values: set(values))
            )
            overlaps = [
                len(clean_sets.loc[query_id] & odd_sets.loc[query_id]) / 10
                for query_id in clean_sets.index
            ]
            records.append(
                {
                    "scope": "development",
                    "source": source,
                    "query_variant": orientation,
                    "queries": len(query_rows),
                    "ndcg_at_10": odd_ndcg,
                    "ndcg_change_from_clean": odd_ndcg - clean_ndcg,
                    "mean_top10_overlap": float(np.mean(overlaps)),
                }
            )
    return pd.DataFrame(records)


def _write_figure(
    comparison: pd.DataFrame,
    selection: pd.DataFrame,
    feature_indexes: dict[tuple[str, str], FeatureIndex],
    *,
    selected_size: str,
    probe_winner_size: str,
    destination: Path,
) -> None:
    fold1 = comparison.loc[
        comparison["fold"].eq(1)
        & comparison["protocol"].eq("primary")
        & comparison["metric"].eq("ndcg")
        & comparison["k"].eq(10)
        & comparison["aggregation"].eq("query_mean")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for source, label, colour in (
        ("teacher", "Teacher → teacher", "#2A9D8F"),
        ("v1", "V1 → V1", "#E76F51"),
    ):
        rows = fold1.loc[
            fold1["query_source"].eq(source)
            & fold1["gallery_source"].eq(source)
        ].sort_values("width")
        axes[0].plot(
            rows["width"] * rows["height"],
            rows["value"],
            marker="o",
            label=label,
            color=colour,
        )
    axes[0].set(
        xlabel="Input pixels",
        ylabel="Mean nDCG@10",
        title="Same-source quality",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for query_source, gallery_source, label, colour in (
        ("teacher", "v1", "Teacher → V1", "#457B9D"),
        ("v1", "teacher", "V1 → teacher", "#F4A261"),
    ):
        rows = fold1.loc[
            fold1["query_source"].eq(query_source)
            & fold1["gallery_source"].eq(gallery_source)
        ].sort_values("width")
        axes[1].plot(
            rows["width"] * rows["height"],
            rows["value"],
            marker="o",
            label=label,
            color=colour,
        )
    axes[1].set(
        xlabel="Input pixels",
        ylabel="Mean nDCG@10",
        title="Cross-source robustness",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    ordered = selection.sort_values("pixels")
    throughput: list[float] = []
    tensor_kib: list[float] = []
    for row in ordered.itertuples(index=False):
        seconds = sum(
            feature_indexes[(row.size, source)].transform_seconds
            for source in ("teacher", "v1")
        )
        image_count = sum(
            len(feature_indexes[(row.size, source)].ids)
            for source in ("teacher", "v1")
        )
        throughput.append(image_count / seconds)
        tensor_kib.append(int(row.width) * int(row.height) * 3 * 4 / 1024)
    axes[2].bar(ordered["size"], throughput, color="#264653")
    axes[2].set(
        xlabel="Width × height",
        ylabel="Images per second",
        title="Measured throughput and tensor cost",
    )
    cost_axis = axes[2].twinx()
    cost_axis.plot(ordered["size"], tensor_kib, color="#E9C46A", marker="o")
    cost_axis.set_ylabel("Float32 RGB KiB per image")
    fig.suptitle(
        "Task 4 preprocessing comparison — "
        f"frozen {selected_size}; probe winner {probe_winner_size}"
    )
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    evidence_dir = ROOT / "results/evidence/task4"
    figure_dir = ROOT / "results/figures/task4"
    local_dir = ROOT / "data/processed/task4/preprocessing"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    splits, development = _load_development_sources()
    feature_cache_root = local_dir / "features"
    for contract in CANDIDATE_CONTRACTS:
        size = f"{contract.width}x{contract.height}"
        for source, (path_column, sha_column) in SOURCE_SPECS.items():
            print(f"Extracting {size} {source} features...", flush=True)
            ensure_feature_index(
                development,
                path_column=path_column,
                sha_column=sha_column,
                source=source,
                contract=contract,
                cache_root=feature_cache_root,
                root=ROOT,
                workers=args.workers,
            )
    experiment = run_preprocessing_experiment(
        splits,
        development,
        contracts=CANDIDATE_CONTRACTS,
        feature_cache_root=feature_cache_root,
        root=ROOT,
        workers=args.workers,
    )
    write_deterministic_csv(
        experiment.comparison,
        evidence_dir / "preprocessing_comparison.csv",
        index=False,
        float_format="%.8f",
    )
    write_deterministic_csv(
        experiment.selection,
        evidence_dir / "preprocessing_size_selection.csv",
        index=False,
        float_format="%.8f",
    )
    write_deterministic_csv(
        experiment.stability,
        evidence_dir / "preprocessing_stability.csv",
        index=False,
        float_format="%.8f",
    )

    selected_size = f"{SELECTED_CONTRACT.width}x{SELECTED_CONTRACT.height}"
    size_policy = resolve_size_policy(
        experiment.selection,
        selected_size=selected_size,
    )
    selected = experiment.selection.loc[
        experiment.selection["size"].eq(selected_size)
    ].iloc[0]
    robustness = _robustness_evidence(
        splits,
        development,
        winner_size=selected_size,
        winner_contract=SELECTED_CONTRACT,
        feature_indexes=experiment.feature_indexes,
        workers=args.workers,
    )
    write_deterministic_csv(
        robustness,
        evidence_dir / "preprocessing_robustness.csv",
        index=False,
        float_format="%.8f",
    )

    normalization: dict[str, object] = {
        "schema_version": "1.0.0",
        "scope": "development training folds only",
        "validation_fold": 1,
        "contract": SELECTED_CONTRACT.to_dict(),
        "sources": {},
    }
    cache_manifests: dict[str, object] = {}
    for source, (path_column, sha_column) in SOURCE_SPECS.items():
        cache = ensure_development_image_cache(
            development,
            path_column=path_column,
            sha_column=sha_column,
            source=source,
            contract=SELECTED_CONTRACT,
            cache_root=local_dir / "images",
            root=ROOT,
        )
        cache_manifests[source] = cache.manifest
        normalization["sources"][source] = fit_cached_fold_rgb_statistics(
            cache,
            development,
            validation_fold=1,
        )
    (evidence_dir / "preprocessing_normalization_fold1.json").write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract_evidence = {
        "schema_version": "1.0.0",
        "scope": "development",
        **size_policy,
        "selected_width": int(selected["width"]),
        "selected_height": int(selected["height"]),
        "candidate_sizes": [
            f"{contract.width}x{contract.height}"
            for contract in CANDIDATE_CONTRACTS
        ],
        "selection_policy": (
            "240x320 is frozen by ADR 0021 to preserve V1 detail for learned "
            "models; this explicitly overrides the fixed probe ranking"
        ),
        "probe_ranking_rule": (
            "equal mean of teacher-to-teacher and v1-to-v1 Protocol A "
            "query-mean linear nDCG@10"
        ),
        "cross_source_role": "supporting robustness only",
        "probe": {
            "version": PROBE_VERSION,
            "name": "fixed spatial HSV and edge probe",
            "grid": "4x4",
            "weights": "equal L2-normalized colour and edge blocks",
            "distance": "exact cosine",
            "trained": False,
        },
        "input_contract": SELECTED_CONTRACT.to_dict(),
        "normalization": (
            "fit RGB mean/std on the current round training folds only; "
            "never refit queries"
        ),
        "cache_scope": "lossless uint8 teacher and V1 development images only",
        "holdout_opened": False,
        "cache_manifests": cache_manifests,
        "limitation": (
            "This fixed probe supports a preprocessing decision but does not "
            "prove learned-model quality."
        ),
    }
    (evidence_dir / "preprocessing_contract.json").write_text(
        json.dumps(contract_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_figure(
        experiment.comparison,
        experiment.selection,
        experiment.feature_indexes,
        selected_size=selected_size,
        probe_winner_size=str(size_policy["probe_winner_size"]),
        destination=figure_dir / "preprocessing_comparison.png",
    )
    print(
        f"Frozen {selected_size}; probe winner "
        f"{size_policy['probe_winner_size']}; development-only evidence rebuilt."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
