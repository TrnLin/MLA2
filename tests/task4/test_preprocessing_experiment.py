from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import fashion.task4.preprocessing_experiment as preprocessing_experiment
from fashion.config import ROOT
from fashion.data.splits import cv_assignment_digest
from fashion.task4.preprocessing import PreprocessingContract
from fashion.task4.preprocessing_experiment import (
    FeatureIndex,
    build_odd_aspect_canvas,
    build_size_selection,
    ensure_feature_index,
    evaluate_source_pair,
    extract_canvas_feature_index,
    extract_feature_index,
    run_preprocessing_experiment,
    select_top_sizes,
    source_directions,
    summarize_stability,
)
from fashion.task4.protocol import RetrievalViews


def _learned_extract(pixels: np.ndarray, _mask: np.ndarray) -> np.ndarray:
    feature = np.zeros(128, dtype=np.float32)
    feature[int(pixels[0, 0, 0]) % 128] = 1.0
    return feature


def _nan_extract(_pixels: np.ndarray, _mask: np.ndarray) -> np.ndarray:
    return np.array([np.nan, 0.0], dtype=np.float32)


def _nonconvertible_extract(_pixels: np.ndarray, _mask: np.ndarray) -> np.ndarray:
    return np.array(["not-a-number"], dtype=object)


def _wrong_rank_extract(_pixels: np.ndarray, _mask: np.ndarray) -> np.ndarray:
    return np.array([[1.0, 0.0]], dtype=np.float32)


def _zero_extract(_pixels: np.ndarray, _mask: np.ndarray) -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def _bad_norm_extract(_pixels: np.ndarray, _mask: np.ndarray) -> np.ndarray:
    return np.ones(2, dtype=np.float32)


def _inconsistent_extract(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    red = int(pixels[mask][0, 0])
    size = 2 if red == 1 else 3
    feature = np.zeros(size, dtype=np.float32)
    feature[0] = 1.0
    return feature


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


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"id": [1], "source_path": ["does-not-exist.png"]}),
        pd.DataFrame(
            {
                "id": [1],
                "partition": ["holdout"],
                "source_path": ["does-not-exist.png"],
            }
        ),
    ],
)
def test_canvas_extraction_requires_development_before_file_access(
    frame: pd.DataFrame,
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="development|partition"):
        extract_canvas_feature_index(
            frame,
            source="v1",
            path_column="source_path",
            orientation="wide",
            contract=PreprocessingContract(width=4, height=4),
            root=tmp_path,
            workers=1,
        )


def test_canvas_extraction_sorts_numeric_ids(tmp_path) -> None:
    Image.new("RGB", (2, 4), (255, 0, 0)).save(tmp_path / "red.png")
    Image.new("RGB", (4, 2), (0, 0, 255)).save(tmp_path / "blue.png")
    query_rows = pd.DataFrame(
        {
            "id": [2, 1],
            "partition": ["development", "development"],
            "source_path": ["blue.png", "red.png"],
        }
    )

    result = extract_canvas_feature_index(
        query_rows,
        source="v1",
        path_column="source_path",
        orientation="tall",
        contract=PreprocessingContract(width=4, height=4),
        root=tmp_path,
        workers=1,
    )

    assert result.ids.tolist() == [1, 2]
    assert result.features.shape[0] == 2
    assert result.source_bytes == sum(
        (tmp_path / name).stat().st_size for name in ("red.png", "blue.png")
    )


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
    assert second.cache_dir == (
        arguments["cache_root"] / arguments["contract"].key / arguments["source"]
    )
    assert {
        "method",
        "fold",
        "checkpoint_fingerprint",
        "config_fingerprint",
    }.isdisjoint(second.manifest)
    assert features_path.stat().st_mtime_ns == first_mtime
    assert np.array_equal(second.index.ids, first.index.ids)
    assert np.array_equal(second.index.features, first.index.features)


@pytest.mark.parametrize(
    ("changed_argument", "changed_value"),
    [
        ("checkpoint_fingerprint", "checkpoint-b"),
        ("config_fingerprint", "config-b"),
    ],
)
def test_learned_feature_cache_identity_includes_checkpoint_and_config(
    tmp_path,
    changed_argument: str,
    changed_value: str,
) -> None:
    Image.new("RGB", (2, 2), (7, 0, 0)).save(tmp_path / "image.png")
    frame = pd.DataFrame(
        {
            "id": [1],
            "partition": ["development"],
            "source_path": ["image.png"],
        }
    )
    arguments = {
        "path_column": "source_path",
        "source": "teacher",
        "contract": PreprocessingContract(width=4, height=4),
        "cache_root": tmp_path / "feature-cache",
        "root": tmp_path,
        "workers": 1,
        "extract": _learned_extract,
        "method": "r1-vicreg",
        "checkpoint_fingerprint": "checkpoint-a",
        "config_fingerprint": "config-a",
        "fold": 2,
    }

    first = ensure_feature_index(frame, **arguments)
    second = ensure_feature_index(frame, **{**arguments, changed_argument: changed_value})

    assert first.cache_dir != second.cache_dir
    assert first.index.features.shape == (1, 128)
    assert first.manifest["method"] == "r1-vicreg"
    assert first.manifest["fold"] == 2
    assert first.manifest["checkpoint_fingerprint"] == "checkpoint-a"
    assert first.manifest["config_fingerprint"] == "config-a"


@pytest.mark.parametrize(
    ("checkpoint_fingerprint", "config_fingerprint", "message"),
    [
        (None, "config-a", "checkpoint fingerprint is required"),
        ("checkpoint-a", None, "config fingerprint is required"),
        ("", "config-a", "checkpoint fingerprint is required"),
        ("checkpoint-a", " ", "config fingerprint is required"),
    ],
)
def test_nonlegacy_cache_requires_both_fingerprints_before_file_access(
    tmp_path,
    checkpoint_fingerprint: str | None,
    config_fingerprint: str | None,
    message: str,
) -> None:
    frame = pd.DataFrame(
        {
            "id": [1],
            "partition": ["development"],
            "source_path": ["does-not-exist.png"],
        }
    )

    with pytest.raises(ValueError, match=message):
        ensure_feature_index(
            frame,
            path_column="source_path",
            source="teacher",
            contract=PreprocessingContract(width=4, height=4),
            cache_root=tmp_path / "feature-cache",
            root=tmp_path,
            workers=1,
            extract=_learned_extract,
            method="r1-vicreg",
            checkpoint_fingerprint=checkpoint_fingerprint,
            config_fingerprint=config_fingerprint,
            fold=1,
        )


@pytest.mark.parametrize(
    ("extract", "message"),
    [
        (_nonconvertible_extract, "float32-convertible"),
        (_nan_extract, "finite"),
        (_wrong_rank_extract, "one-dimensional"),
        (_zero_extract, "non-zero"),
        (_bad_norm_extract, "unit norm"),
        (_inconsistent_extract, "consistent dimension"),
    ],
)
def test_feature_cache_rejects_invalid_extracted_vectors_before_write(
    tmp_path,
    extract,
    message: str,
) -> None:
    Image.new("RGB", (2, 2), (1, 0, 0)).save(tmp_path / "first.png")
    Image.new("RGB", (2, 2), (2, 0, 0)).save(tmp_path / "second.png")
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "partition": ["development", "development"],
            "source_path": ["first.png", "second.png"],
        }
    )
    cache_root = tmp_path / "feature-cache"

    with pytest.raises(ValueError, match=message):
        ensure_feature_index(
            frame,
            path_column="source_path",
            source="teacher",
            contract=PreprocessingContract(width=4, height=4),
            cache_root=cache_root,
            root=tmp_path,
            workers=1,
            extract=extract,
            method="r1-vicreg",
            checkpoint_fingerprint="checkpoint-a",
            config_fingerprint="config-a",
            fold=1,
        )

    assert not cache_root.exists()


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "r2-vicreg"),
        ("fold", 2),
        ("checkpoint_fingerprint", "checkpoint-b"),
        ("config_fingerprint", "config-b"),
    ],
)
def test_source_pair_rejects_mismatched_feature_provenance(
    field: str,
    value: object,
) -> None:
    primary, family = _evaluation_views()
    common = {
        "method": "r1-vicreg",
        "fold": 1,
        "checkpoint_fingerprint": "checkpoint-a",
        "config_fingerprint": "config-a",
    }
    query_index = replace(
        _feature_index("teacher"),
        **common,
    )
    gallery_index = replace(
        _feature_index("v1"),
        **{**common, field: value},
    )

    with pytest.raises(ValueError, match="provenance"):
        evaluate_source_pair(
            query_index,
            gallery_index,
            primary_views=primary,
            family_views=family,
            fold=1,
            k_values=(1,),
            family_k=1,
        )


def test_source_pair_rejects_free_fold_and_derives_identity_labels() -> None:
    primary, family = _evaluation_views()
    identity = {
        "method": "r1-vicreg",
        "fold": 2,
        "checkpoint_fingerprint": "checkpoint-a",
        "config_fingerprint": "config-a",
    }
    query_index = replace(_feature_index("teacher"), **identity)
    gallery_index = replace(_feature_index("v1"), **identity)

    with pytest.raises(ValueError, match="fold.*provenance"):
        evaluate_source_pair(
            query_index,
            gallery_index,
            primary_views=primary,
            family_views=family,
            fold=1,
            k_values=(1,),
            family_k=1,
        )

    evaluation = evaluate_source_pair(
        query_index,
        gallery_index,
        primary_views=primary,
        family_views=family,
        fold=2,
        k_values=(1,),
        family_k=1,
    )
    assert set(evaluation.summary["method"]) == {"r1-vicreg"}
    assert set(evaluation.summary["fold"]) == {2}
    assert set(evaluation.summary["checkpoint_fingerprint"]) == {"checkpoint-a"}
    assert set(evaluation.summary["config_fingerprint"]) == {"config-a"}


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


def _preprocessing_runner_module():
    spec = importlib.util.spec_from_file_location(
        "task4_run_preprocessing_for_tests",
        ROOT / "scripts/task4/run_preprocessing.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner_development_frame() -> pd.DataFrame:
    ids = [1, 2, 3]
    return pd.DataFrame(
        {
            "id": ids,
            "partition": ["development"] * 3,
            "cv_fold": [0, 2, 1],
            "teacher_path": [f"teacher/{value}.png" for value in ids],
            "teacher_sha256": [f"teacher-{value}" for value in ids],
            "external_path": [f"v1/{value}.png" for value in ids],
            "external_sha256": [f"v1-{value}" for value in ids],
        }
    )


def _runner_splits() -> pd.DataFrame:
    ids = [1, 2, 3]
    return pd.DataFrame(
        {
            "id": ids,
            "path": [f"teacher/{value}.png" for value in ids],
            "sha256": [f"teacher-{value}" for value in ids],
            "duplicate_group": [f"duplicate-{value}" for value in ids],
            "product_name_key": [f"name-{value}" for value in ids],
            "product_family_group": [f"family-{value}" for value in ids],
            "partition": ["development"] * 3,
            "cv_fold": [0, 2, 1],
            "is_cross_role_exact_duplicate": [False] * 3,
            "is_cross_role_near_duplicate": [False] * 3,
            "has_conflicting_target_labels": [False] * 3,
            "conflicting_targets": [""] * 3,
            "quarantine_reason": [""] * 3,
        }
    )


def test_preprocessing_runner_normalization_binds_the_canonical_split_digest(
    tmp_path,
) -> None:
    module = _preprocessing_runner_module()
    development = _runner_development_frame()
    splits = _runner_splits()
    colours = {1: (0, 0, 0), 2: (255, 255, 255), 3: (255, 0, 0)}
    for source in ("teacher", "v1"):
        (tmp_path / source).mkdir()
        for product_id, colour in colours.items():
            Image.new("RGB", (24, 32), colour).save(
                tmp_path / source / f"{product_id}.png"
            )

    normalization, cache_manifests = module.build_normalization_evidence(
        splits,
        development,
        cache_root=tmp_path / "cache",
        root=tmp_path,
    )

    digest = cv_assignment_digest(splits)
    assert normalization["split_fingerprint"] == digest
    assert normalization["validation_fold"] == 1
    assert set(normalization["sources"]) == {"teacher", "v1"}
    assert set(cache_manifests) == {"teacher", "v1"}
    for source in ("teacher", "v1"):
        assert normalization["sources"][source]["split_fingerprint"] == digest
        assert normalization["sources"][source]["validation_fold"] == 1


def test_committed_normalization_artifact_carries_the_canonical_split_digest() -> None:
    artifact = json.loads(
        (ROOT / "results/evidence/task4/preprocessing_normalization_fold1.json").read_text(
            encoding="utf-8"
        )
    )
    splits = pd.read_csv(ROOT / "data/processed/splits.csv", keep_default_na=False)
    digest = cv_assignment_digest(splits)

    assert artifact["split_fingerprint"] == digest
    assert set(artifact["sources"]) == {"teacher", "v1"}
    for source in ("teacher", "v1"):
        assert artifact["sources"][source]["split_fingerprint"] == digest


def test_preprocessing_evidence_runner_has_a_help_entrypoint() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/task4/run_preprocessing.py", "--help"],
        cwd=ROOT,
        check=True,
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


def test_size_policy_keeps_probe_winner_visible_when_detail_choice_overrides_it() -> None:
    selection = build_size_selection(_selection_rows())

    decision = preprocessing_experiment.resolve_size_policy(
        selection, selected_size="240x320"
    )

    assert decision == {
        "selected_size": "240x320",
        "selected_probe_rank": 2,
        "selected_probe_ndcg_at_10": pytest.approx(0.6),
        "probe_winner_size": "96x128",
        "probe_winner_ndcg_at_10": pytest.approx(0.65),
    }


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
