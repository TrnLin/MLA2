"""Fail-closed checks for the isolated post-submission RF grid."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from fashion.task2.post_submission_rf_grid import (
    CONFIG_PATH,
    _assert_baseline_parity,
    load_grid_config,
)


def test_grid_is_frozen_to_five_canonical_folds_and_small_candidate_set() -> None:
    config = load_grid_config()
    assert config["folds"] == [0, 1, 2, 3, 4]
    assert [row["id"] for row in config["candidates"]] == [
        "r0_baseline",
        "r1_leaf1",
        "r2_leaf4",
        "r3_features025",
        "r4_leaf4_features025",
    ]
    assert config["selection"]["minimum_macro_f1_gain"] == 0.003


@pytest.mark.parametrize("change", ["folds", "baseline", "threshold"])
def test_grid_rejects_changed_protocol(tmp_path, change: str) -> None:
    config = copy.deepcopy(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    if change == "folds":
        config["folds"] = [0, 1, 2, 3]
    elif change == "baseline":
        config["candidates"][0]["min_samples_leaf"] = 1
    else:
        config["selection"]["minimum_macro_f1_gain"] = 0.0
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="changed|canonical"):
        load_grid_config(path)


def test_baseline_parity_checks_both_labels_and_probabilities() -> None:
    original = pd.DataFrame(
        {
            "id": [10, 20],
            "y_pred": ["Fall", "Spring"],
            "prob_Fall": [0.8, 0.1],
            "prob_Spring": [0.1, 0.8],
            "prob_Summer": [0.05, 0.05],
            "prob_Winter": [0.05, 0.05],
        }
    )
    reproduced = original.iloc[::-1].reset_index(drop=True)
    _assert_baseline_parity(reproduced, original, 0)
    wrong_probability = reproduced.copy()
    wrong_probability.loc[0, "prob_Fall"] += 0.01
    with pytest.raises(ValueError, match="probabilities"):
        _assert_baseline_parity(wrong_probability, original, 0)
    wrong_label = reproduced.copy()
    wrong_label.loc[0, "y_pred"] = "Fall"
    with pytest.raises(ValueError, match="predictions"):
        _assert_baseline_parity(wrong_label, original, 0)
    assert np.isfinite(original.filter(like="prob_").to_numpy()).all()
