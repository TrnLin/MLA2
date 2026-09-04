from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.train.task3_dataset_v2 import (
    GENDER_G2_ALL_FOLDS,
    GENDER_G2_CONFIRMATION_FOLDS,
    SCREEN_FOLDS,
    VISUAL_COMPONENT_STRATEGY,
    add_visual_component_weights,
    build_visual_component_mapping,
    dataset_v2_spec,
)

ROOT = Path(__file__).resolve().parents[2]


def _parents(target: str) -> tuple[str, str, str, str, str]:
    return tuple(f"{target}-parent-{fold}" for fold in range(5))  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("name", "target", "factor", "view", "augmentation", "weighting", "loss"),
    [
        (
            "gender_v2_foreground_mask",
            "gender",
            "same_canvas_foreground_mask",
            "foreground_masked",
            "none",
            "none",
            "cross_entropy",
        ),
        (
            "gender_v2_translation",
            "gender",
            "training_translation",
            "full",
            "translation_uniform_2px_p05",
            "none",
            "cross_entropy",
        ),
        (
            "gender_v2_component_weight",
            "gender",
            "visual_component_weight",
            "full",
            "none",
            VISUAL_COMPONENT_STRATEGY,
            "visual_component_cross_entropy",
        ),
        (
            "usage_v2_component_weight",
            "usage",
            "visual_component_weight",
            "full",
            "none",
            VISUAL_COMPONENT_STRATEGY,
            "effective_number_visual_component_cross_entropy",
        ),
    ],
)
def test_dataset_v2_specs_lock_one_data_factor(
    name: str,
    target: str,
    factor: str,
    view: str,
    augmentation: str,
    weighting: str,
    loss: str,
) -> None:
    parents = _parents(target)
    spec = dataset_v2_spec(name, parents)  # type: ignore[arg-type]

    assert spec.target == target
    assert spec.changed_factor == factor
    assert spec.input_view == view
    assert spec.training_augmentation == augmentation
    assert spec.sample_weight_strategy == weighting
    assert spec.loss_name == loss
    assert spec.parent_run_ids == parents
    assert spec.checkpoint_policy == "final_epoch"

    with pytest.raises(ValueError, match="more than its predeclared factor"):
        replace(spec, classifier_dropout=0.2)


def test_dataset_v2_keeps_exact_causal_parents() -> None:
    gender = dataset_v2_spec("gender_v2_translation", _parents("gender"))
    usage = dataset_v2_spec("usage_v2_component_weight", _parents("usage"))

    assert gender.parent_artifact_dir == "experiments/t3_gender_e6_gem_p3"
    assert gender.model_family == "task3_small_cnn_gem_p3"
    assert gender.class_weight_beta is None
    assert usage.parent_artifact_dir == "experiments/t3_usage_e2_class_balanced_ce"
    assert usage.model_family == "task3_small_cnn"
    assert usage.class_weight_beta == pytest.approx(0.999)
    assert usage.class_weight_cap == pytest.approx(5.0)
    assert SCREEN_FOLDS == (0, 4)
    assert GENDER_G2_CONFIRMATION_FOLDS == (1, 2, 3)
    assert GENDER_G2_ALL_FOLDS == (0, 1, 2, 3, 4)


def test_foreground_mask_keeps_the_canvas_and_removes_connected_white_background() -> None:
    pytest.importorskip("torch")
    from fashion.train.data import apply_task3_input_view

    pixels = np.full((8, 6, 3), 250, dtype=np.uint8)
    pixels[2:6, 2:4] = (20, 40, 60)
    image = Image.fromarray(pixels)

    masked = np.asarray(apply_task3_input_view(image, "foreground_masked"))

    assert masked.shape == pixels.shape
    assert np.array_equal(masked[3, 2], np.array([20, 40, 60], dtype=np.uint8))
    assert np.array_equal(masked[0, 0], np.array([255, 255, 255], dtype=np.uint8))
    with pytest.raises(ValueError, match="unknown Task 3 input view"):
        apply_task3_input_view(image, "crop")


def test_visual_components_join_exact_and_accepted_near_duplicates(tmp_path) -> None:
    splits = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "partition": ["development"] * 5,
            "cv_fold": [0, 0, 4, 4, 2],
            "sha256": ["same", "same", "three", "four", "five"],
            "gender": ["Boys", "Boys", "Girls", "Girls", "Men"],
        }
    )
    candidates = pd.DataFrame(
        {
            "id_1": [3, 1],
            "id_2": [4, 5],
            "role_1": ["labelled", "labelled"],
            "role_2": ["labelled", "prediction"],
            "accepted_near_duplicate": [True, True],
        }
    )
    path = tmp_path / "near.csv.gz"
    candidates.to_csv(path, index=False)

    mapping, contract = build_visual_component_mapping(splits, candidates_path=path)
    component = mapping.set_index("id")["visual_component_id"]

    assert component[1] == component[2]
    assert component[3] == component[4]
    assert component[1] != component[5]
    assert contract["components"] == 3
    assert contract["multirow_components"] == 2
    assert contract["multirow_component_rows"] == 4
    assert contract["fold_crossings"] == 0

    weighted, weight_contract = add_visual_component_weights(
        splits,
        splits,
        target="gender",
        candidates_path=path,
    )
    weights = weighted.set_index("id")["visual_component_weight"]
    assert weights[1] == pytest.approx(weights[2])
    assert weights[3] == pytest.approx(weights[4])
    assert weights[5] == pytest.approx(2 * weights[1])
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weight_contract["one_row_per_epoch"] is True
    assert weight_contract["weighted_sampler"] is False


def test_visual_component_mapping_rejects_fold_crossing(tmp_path) -> None:
    splits = pd.DataFrame(
        {
            "id": [1, 2],
            "partition": ["development", "development"],
            "cv_fold": [0, 4],
            "sha256": ["same", "same"],
        }
    )
    path = tmp_path / "near.csv.gz"
    pd.DataFrame(
        columns=("id_1", "id_2", "role_1", "role_2", "accepted_near_duplicate")
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="crosses canonical folds"):
        build_visual_component_mapping(splits, candidates_path=path)


@pytest.mark.parametrize(
    ("filename", "screen_name", "parent_lookup"),
    [
        (
            "04n_task3_gem_gender_v2_g1_foreground_mask.ipynb",
            "gender_v2_foreground_mask",
            "latest_completed_gender_e6_parent_run_ids",
        ),
        (
            "04o_task3_gem_gender_v2_g2_translation.ipynb",
            "gender_v2_translation",
            "latest_completed_gender_e6_parent_run_ids",
        ),
        (
            "04p_task3_gem_gender_v2_g3_component_weight.ipynb",
            "gender_v2_component_weight",
            "latest_completed_gender_e6_parent_run_ids",
        ),
        (
            "04q_task3_smallcnn_usage_v2_u1_component_weight.ipynb",
            "usage_v2_component_weight",
            "latest_completed_usage_e2_parent_run_ids",
        ),
    ],
)
def test_dataset_v2_notebooks_run_only_their_two_fold_screen(
    filename: str,
    screen_name: str,
    parent_lookup: str,
) -> None:
    notebook = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert source.count("run_task3_dataset_v2_screen(") == 1
    assert source.count("check_task3_dataset_v2_setup(") == 1
    assert f'    "{screen_name}",' in source
    assert parent_lookup in source
    assert "folds 0 and 4" in source
    assert "run_task3_baseline_cv" not in source
    assert "run_task3_child_cv" not in source
    assert "fashion_product_images_v1" not in source
    saved_errors = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert saved_errors == []


def test_g2_confirmation_notebook_trains_only_missing_folds() -> None:
    path = ROOT / "notebooks/04r_task3_gem_gender_v2_g2_confirmation.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert source.count("run_task3_gender_v2_g2_confirmation(") == 1
    assert source.count("check_task3_gender_v2_g2_confirmation_setup(") == 1
    assert "folds=(1, 2, 3)" in code
    assert "reuses completed G2 folds 0 and 4" in source
    assert "run_task3_dataset_v2_screen(" not in code
    assert "fashion_product_images_v1" not in source
    assert not any(
        output.get("output_type") == "error"
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )
