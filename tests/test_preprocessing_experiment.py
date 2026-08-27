from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.config import ROOT
from fashion.retrieval.preprocessing import PreprocessingContract
from fashion.retrieval.preprocessing_experiment import (
    FeatureIndex,
    build_odd_aspect_canvas,
    build_size_selection,
    ensure_feature_index,
    evaluate_source_pair,
    extract_feature_index,
    run_preprocessing_experiment,
    select_top_sizes,
    source_directions,
    summarize_stability,
)
from fashion.retrieval.protocol import RetrievalViews


def test_feature_extraction_rejects_non_development_rows_before_file_access(
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        {
            "id": [1],
            "partition": ["holdout"],
            "source_path": ["does-not-exist.jpg"],
        }
    )

    with pytest.raises(ValueError, match="development"):
        extract_feature_index(
            frame,
            path_column="source_path",
            source="teacher",
            contract=PreprocessingContract(width=4, height=4),
            root=tmp_path,
        )


def test_feature_extraction_sorts_ids_and_records_real_cost(tmp_path) -> None:
    Image.new("RGB", (2, 2), (255, 0, 0)).save(tmp_path / "red.png")
    Image.new("RGB", (2, 2), (0, 0, 255)).save(tmp_path / "blue.png")
    frame = pd.DataFrame(
        {
            "id": [2, 1],
            "partition": ["development", "development"],
            "source_path": ["blue.png", "red.png"],
        }
    )

    result = extract_feature_index(
        frame,
        path_column="source_path",
        source="teacher",
        contract=PreprocessingContract(width=4, height=4),
        root=tmp_path,
        workers=1,
    )

    assert result.ids.tolist() == [1, 2]
    assert result.features.shape[0] == 2
    assert np.linalg.norm(result.features, axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert result.source_bytes == sum(
        (tmp_path / name).stat().st_size for name in ("red.png", "blue.png")
    )
    assert result.transform_seconds > 0


def test_feature_cache_reuses_an_exact_source_and_contract(tmp_path) -> None:
    Image.new("RGB", (2, 2), (255, 0, 0)).save(tmp_path / "red.png")
    frame = pd.DataFrame(
        {
            "id": [1],
            "partition": ["development"],
            "source_path": ["red.png"],
            "source_sha256": ["red-sha"],
        }
    )
    arguments = {
        "path_column": "source_path",
        "sha_column": "source_sha256",
        "source": "teacher",
        "contract": PreprocessingContract(width=4, height=4),
        "cache_root": tmp_path / "feature-cache",
        "root": tmp_path,
        "workers": 1,
    }

    first = ensure_feature_index(frame, **arguments)
    features_path = first.cache_dir / "features.npy"
    first_mtime = features_path.stat().st_mtime_ns
    second = ensure_feature_index(frame, **arguments)

    assert second.cache_dir == first.cache_dir
    assert features_path.stat().st_mtime_ns == first_mtime
    assert np.array_equal(second.index.ids, first.index.ids)
    assert np.array_equal(second.index.features, first.index.features)


def test_source_matrix_contains_four_directions() -> None:
    assert source_directions() == (
        ("teacher", "teacher"),
        ("v1", "v1"),
        ("teacher", "v1"),
        ("v1", "teacher"),
    )


def _evaluation_views() -> tuple[RetrievalViews, RetrievalViews]:
    rows = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "articleType": ["Shirts", "Shirts", "Dress", "Dress"],
            "baseColour": ["Blue", "Blue", "Red", "Red"],
            "sha256": ["sha-1", "sha-2", "sha-3", "sha-4"],
            "duplicate_group": ["dup-1", "dup-2", "dup-3", "dup-4"],
            "product_family_group": ["family-q", "family-2", "family-3", "family-q"],
        }
    ).set_index("id", drop=False)
    primary = RetrievalViews(
        queries=rows.loc[[1, 4]].reset_index(drop=True),
        gallery=rows.loc[[2, 3]].reset_index(drop=True),
    )
    family_rows = rows.loc[[1, 4]].reset_index(drop=True)
    return primary, RetrievalViews(queries=family_rows, gallery=family_rows)


def _feature_index(source: str) -> FeatureIndex:
    return FeatureIndex(
        source=source,
        contract=PreprocessingContract(width=4, height=4),
        ids=np.array([1, 2, 3, 4]),
        features=np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        transform_seconds=2.0,
        source_bytes=100,
    )


def test_source_pair_uses_both_frozen_protocols() -> None:
    primary, family = _evaluation_views()

    evaluation = evaluate_source_pair(
        _feature_index("teacher"),
        _feature_index("v1"),
        primary_views=primary,
        family_views=family,
        fold=1,
        k_values=(1,),
        family_k=1,
    )

    assert set(evaluation.summary["protocol"]) == {"primary", "family"}
    assert set(evaluation.summary["query_source"]) == {"teacher"}
    assert set(evaluation.summary["gallery_source"]) == {"v1"}
    ndcg = evaluation.summary.query(
        "protocol == 'primary' and metric == 'ndcg' and aggregation == 'query_mean'"
    )
    assert ndcg["value"].item() == pytest.approx(1.0)
    recall = evaluation.summary.query("protocol == 'family' and metric == 'recall'")
    assert recall["value"].item() == pytest.approx(1.0)
    assert evaluation.primary_rankings.groupby("query_id").size().to_dict() == {1: 1, 4: 1}


def test_full_experiment_runs_matrix_and_top_two_stability(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    variants: list[dict[str, object]] = []
    for fold in range(5):
        for offset in range(11):
            product_id = 100 + fold * 11 + offset
            path = tmp_path / f"{product_id}.png"
            Image.new("RGB", (2, 3), (20 * fold, 30 * offset, 100)).save(path)
            rows.append(
                {
                    "id": product_id,
                    "sha256": f"sha-{product_id}",
                    "duplicate_group": f"duplicate-{product_id}",
                    "product_name_key": f"family-{fold}",
                    "product_family_group": f"family-{fold}",
                    "partition": "development",
                    "cv_fold": fold,
                    "is_cross_role_exact_duplicate": False,
                    "is_cross_role_near_duplicate": False,
                    "has_conflicting_target_labels": False,
                    "conflicting_targets": "",
                    "quarantine_reason": "",
                    "articleType": "Shirts",
                    "baseColour": "Blue",
                    "season": "Summer",
                    "gender": "Unisex",
                    "usage": "Casual",
                    "has_articleType_label": True,
                    "has_season_label": True,
                    "has_gender_label": True,
                    "has_usage_label": True,
                }
            )
            variants.append(
                {
                    "id": product_id,
                    "partition": "development",
                    "teacher_path": path.name,
                    "teacher_sha256": f"sha-{product_id}",
                    "external_path": path.name,
                    "external_sha256": f"sha-{product_id}",
                }
            )

    experiment = run_preprocessing_experiment(
        pd.DataFrame(rows),
        pd.DataFrame(variants),
        contracts=(
            PreprocessingContract(width=2, height=3),
            PreprocessingContract(width=4, height=6),
        ),
        feature_cache_root=tmp_path / "features",
        root=tmp_path,
        workers=1,
    )

    fold_one = experiment.comparison.loc[experiment.comparison["fold"].eq(1)]
    assert set(experiment.comparison["scope"]) == {"development"}
    assert set(zip(fold_one["query_source"], fold_one["gallery_source"], strict=False)) == {
        ("teacher", "teacher"),
        ("v1", "v1"),
        ("teacher", "v1"),
        ("v1", "teacher"),
    }
    assert set(experiment.stability["fold"]) == set(range(5))
    assert len(experiment.top_sizes) == 2


def test_preprocessing_evidence_runner_has_a_help_entrypoint() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_task4_preprocessing.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "development-only Task 4 preprocessing evidence" in completed.stdout


def _selection_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for size, width, height, teacher, v1, cross in (
        ("60x80", 60, 80, 0.60, 0.50, 0.99),
        ("96x128", 96, 128, 0.65, 0.65, 0.01),
        ("240x320", 240, 320, 0.70, 0.50, 1.00),
    ):
        for query_source, gallery_source, value in (
            ("teacher", "teacher", teacher),
            ("v1", "v1", v1),
            ("teacher", "v1", cross),
            ("v1", "teacher", cross),
        ):
            rows.append(
                {
                    "fold": 1,
                    "size": size,
                    "width": width,
                    "height": height,
                    "query_source": query_source,
                    "gallery_source": gallery_source,
                    "protocol": "primary",
                    "metric": "ndcg",
                    "k": 10,
                    "aggregation": "query_mean",
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def test_size_selection_uses_only_equal_same_source_ndcg_mean() -> None:
    selection = build_size_selection(_selection_rows())

    assert selection.iloc[0]["size"] == "96x128"
    assert selection.iloc[0]["selection_ndcg_at_10"] == pytest.approx(0.65)
    assert selection.set_index("size").loc["60x80", "selection_ndcg_at_10"] == pytest.approx(
        0.55
    )


def test_top_size_ties_break_toward_fewer_pixels() -> None:
    rows = _selection_rows()
    rows.loc[
        rows["query_source"].eq(rows["gallery_source"]),
        "value",
    ] = 0.5

    selection = build_size_selection(rows)

    assert select_top_sizes(selection, count=2) == ("60x80", "96x128")


def test_stability_uses_each_fold_once_for_only_top_two_sizes() -> None:
    rows: list[dict[str, object]] = []
    for size in ("60x80", "96x128", "240x320"):
        for fold in range(5):
            for source in ("teacher", "v1"):
                rows.append(
                    {
                        "fold": fold,
                        "size": size,
                        "query_source": source,
                        "gallery_source": source,
                        "protocol": "primary",
                        "metric": "ndcg",
                        "k": 10,
                        "aggregation": "query_mean",
                        "value": 0.5 + 0.01 * fold,
                    }
                )

    stability = summarize_stability(
        pd.DataFrame(rows),
        top_sizes=("60x80", "96x128"),
    )

    assert set(stability["size"]) == {"60x80", "96x128"}
    assert stability["fold_count"].tolist() == [5, 5]
    assert stability["mean_selection_ndcg_at_10"].tolist() == pytest.approx([0.52, 0.52])


@pytest.mark.parametrize(
    ("orientation", "expected_size", "expected_offset"),
    [
        ("wide", (8, 4), (3, 0)),
        ("tall", (2, 4), (0, 0)),
    ],
)
def test_wide_and_tall_canvases_do_not_crop_content(
    orientation: str,
    expected_size: tuple[int, int],
    expected_offset: tuple[int, int],
) -> None:
    image = Image.new("RGB", (2, 4), (255, 0, 0))

    result = build_odd_aspect_canvas(image, orientation)

    assert result.size == expected_size
    left, top = expected_offset
    assert np.all(
        np.asarray(result)[top : top + image.height, left : left + image.width]
        == (255, 0, 0)
    )
