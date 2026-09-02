from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.train.task3_clean_slate import (
    CLEAN_SLATE_SCREEN_FOLDS,
    _canonical_inner_splits,
    _expand_probabilities,
    _screen_folds,
    build_fixed_feature_cache,
    check_clean_slate_screen_setup,
    fixed_feature_vector,
    smoothed_type_usage_mapping,
)


def test_fixed_features_are_finite_and_keep_separate_views() -> None:
    image = np.full((80, 60, 3), 250, dtype=np.uint8)
    image[20:65, 18:44] = np.array([20, 80, 180], dtype=np.uint8)

    full = fixed_feature_vector(image, view="full")
    masked = fixed_feature_vector(image, view="foreground_masked")

    assert full.dtype == np.float32
    assert full.shape == masked.shape
    assert full.ndim == 1
    assert len(full) > 1900
    assert np.isfinite(full).all()
    assert np.isfinite(masked).all()
    assert not np.array_equal(full, masked)


def test_type_usage_mapping_is_smoothed_and_complete() -> None:
    mapping = smoothed_type_usage_mapping(
        ["Top", "Top", "Shoe"],
        ["Casual", "Sports", "Sports"],
        article_classes=["Top", "Shoe", "Bag"],
        usage_classes=["Casual", "Home", "Sports"],
        strength=5.0,
        global_pseudocount=0.5,
    )

    assert mapping.shape == (3, 3)
    assert np.allclose(mapping.sum(axis=1), 1.0)
    assert (mapping > 0).all()
    assert mapping[2, 2] > mapping[2, 1]


def test_probability_expansion_obeys_fixed_class_order() -> None:
    expanded = _expand_probabilities(
        np.array([[0.7, 0.3]]),
        observed_classes=["B", "A"],
        fixed_classes=["A", "B", "C"],
    )

    assert np.allclose(expanded, [[0.3, 0.7, 0.0]])
    assert expanded.sum(axis=1).tolist() == pytest.approx([1.0])


def test_first_screen_rejects_a_five_fold_run() -> None:
    assert _screen_folds(CLEAN_SLATE_SCREEN_FOLDS) == (0, 4)
    with pytest.raises(ValueError, match="folds 0 and 4"):
        _screen_folds(range(5))


def test_inner_selection_reuses_only_saved_fold_assignments() -> None:
    frame = pd.DataFrame(
        {
            "cv_fold": [1, 1, 2, 2, 3, 3, 4, 4],
            "product_family_group": [f"family-{index}" for index in range(8)],
        }
    )

    splits = _canonical_inner_splits(frame, outer_fold=0)

    assert len(splits) == 4
    assert [sorted(frame.iloc[validation]["cv_fold"].unique()) for _, validation in splits] == [
        [1],
        [2],
        [3],
        [4],
    ]


def test_preflight_is_zero_fit_and_human_review_is_non_blocking(prepared_project) -> None:
    check = check_clean_slate_screen_setup(root=prepared_project.root)

    assert check["training_screen_ready"] is True
    assert check["training_blockers"] == []
    assert check["human_observability_review_status"] == "deferred_non_blocking"
    assert check["folds"] == [0, 4]
    assert check["model_fits"] == 0
    assert check["optimizer_steps"] == 0
    assert check["gender_model"] != check["usage_model"]


def test_teacher_feature_cache_is_built_locally_then_verified_before_reuse(
    prepared_project, tmp_path, monkeypatch
) -> None:
    from fashion.data import load_splits

    splits = load_splits(prepared_project.splits)
    durable_dir = tmp_path / "drive" / "features"
    local_work_dir = tmp_path / "content" / "feature-work"
    opened_paths: list[Path] = []
    original_open_memmap = np.lib.format.open_memmap

    def record_open_memmap(filename, *args, **kwargs):
        opened_paths.append(Path(filename))
        return original_open_memmap(filename, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "open_memmap", record_open_memmap)
    first = build_fixed_feature_cache(
        splits,
        view="full",
        audit_contract_hash="fixed-test-audit",
        output_dir=durable_dir,
        root=prepared_project.root,
        workers=2,
        local_work_dir=local_work_dir,
    )
    build_paths = list(opened_paths)
    second = build_fixed_feature_cache(
        splits,
        view="full",
        audit_contract_hash="fixed-test-audit",
        output_dir=durable_dir,
        root=prepared_project.root,
        workers=2,
        local_work_dir=local_work_dir,
    )

    assert build_paths
    assert all(path.parent == local_work_dir for path in build_paths)
    assert Path(first["matrix_path"]).parent == durable_dir
    assert not list(local_work_dir.glob("*.npy"))
    assert not list(durable_dir.glob("*.tmp.npy"))
    assert first["reused"] is False
    assert second["reused"] is True
    assert first["rows"] == second["rows"]
    assert first["columns"] == second["columns"]


def test_clean_slate_screen_notebook_keeps_models_and_folds_separate() -> None:
    notebook = nbformat.read(ROOT / "notebooks/04l_task3_clean_slate_screen_1.ipynb", as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert notebook.metadata["title"] == "Task 3 — Clean-Slate Screen 1"
    assert source.count("folds=(0, 4)") == 3
    assert code.count("run_clean_slate_gender_screen(") == 1
    assert code.count("run_clean_slate_usage_screen(") == 1
    assert "run_task3_baseline_cv" not in code
    assert "run_task3_child_cv" not in code
    assert "pretrained=True" not in source
    assert "load_splits_for_final_evaluation" not in source
    assert "human observability review is deferred" in source.lower()
    assert "reuse_completed=True" in code
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
