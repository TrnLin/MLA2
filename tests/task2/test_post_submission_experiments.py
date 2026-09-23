from __future__ import annotations

import json

import pandas as pd
import pytest
import torch
from torch import nn

from fashion.config import ROOT
from fashion.models.season import SeasonModelSpec, build_multitask_season_model
from fashion.task2.post_submission_experiments import (
    DEFAULT_ACTIVATION_CONFIG,
    DEFAULT_GRADIENT_BOOSTING_CONFIG,
    DEFAULT_RANDOM_FOREST_CONFIG,
    _ActivationAccumulator,
    _average_probability_oof,
    _nonfinite_gradient_epoch,
    load_activation_experiment_spec,
    load_gradient_boosting_experiment_spec,
    load_random_forest_experiment_spec,
    replace_relu_with_leaky_relu,
)


def test_repository_post_submission_specs_keep_the_frozen_i2_boundary() -> None:
    activation = load_activation_experiment_spec(DEFAULT_ACTIVATION_CONFIG)
    forest = load_random_forest_experiment_spec(DEFAULT_RANDOM_FOREST_CONFIG)
    boosting = load_gradient_boosting_experiment_spec(DEFAULT_GRADIENT_BOOSTING_CONFIG)

    assert activation.base_i2_config == forest.base_i2_config == boosting.base_i2_config
    assert activation.activation == "leaky_relu"
    assert activation.negative_slope == 0.01
    assert activation.near_dead_positive_rate == 0.01
    assert forest.n_estimators == 300
    assert forest.class_weight == "balanced_subsample"
    assert forest.random_state_policy == "seed_plus_fold"
    assert boosting.max_iter == 200
    assert boosting.learning_rate == 0.05
    assert boosting.class_weight == "balanced"
    assert boosting.early_stopping is False


def test_leaky_relu_replacement_changes_only_parameter_free_modules() -> None:
    model = build_multitask_season_model(
        SeasonModelSpec(family="smallcnn", num_classes=4),
        article_type_classes=124,
    )
    original_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }

    replaced = replace_relu_with_leaky_relu(model.base_model, negative_slope=0.01)

    assert len(replaced) == 8
    assert not any(isinstance(module, nn.ReLU) for module in model.base_model.modules())
    leaky = [
        module for module in model.base_model.modules() if isinstance(module, nn.LeakyReLU)
    ]
    assert len(leaky) == 8
    assert all(module.negative_slope == 0.01 for module in leaky)
    assert set(model.state_dict()) == set(original_state)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, original_state[name])


def test_activation_accumulator_separates_exact_and_near_dead_channels() -> None:
    preactivation = torch.tensor(
        [
            [
                [[-1.0, -1.0], [-1.0, -1.0]],
                [[1.0, -1.0], [-1.0, -1.0]],
            ]
        ]
    )

    relu = _ActivationAccumulator()
    relu.observe_input(preactivation, epsilon=1e-12)
    relu.observe_output(torch.relu(preactivation), epsilon=1e-12)
    relu_rows = relu.rows(
        variant="ReLU",
        fold=0,
        layer="test",
        exact_dead_epsilon=1e-12,
        near_dead_positive_rate=0.01,
    )

    leaky = _ActivationAccumulator()
    leaky.observe_input(preactivation, epsilon=1e-12)
    leaky.observe_output(torch.nn.functional.leaky_relu(preactivation, 0.01), epsilon=1e-12)
    leaky_rows = leaky.rows(
        variant="LeakyReLU",
        fold=0,
        layer="test",
        exact_dead_epsilon=1e-12,
        near_dead_positive_rate=0.01,
    )

    assert relu_rows[0]["exact_dead"] is True
    assert relu_rows[0]["near_dead"] is True
    assert relu_rows[0]["postactivation_zero_rate"] == 1.0
    assert relu_rows[1]["exact_dead"] is False
    assert relu_rows[1]["near_dead"] is False
    assert relu_rows[1]["postactivation_zero_rate"] == 0.75

    # LeakyReLU carries negative values forward, but a channel can still receive
    # almost no positive preactivation. This is why both diagnostics are reported.
    assert leaky_rows[0]["exact_dead"] is False
    assert leaky_rows[0]["near_dead"] is True
    assert leaky_rows[0]["postactivation_zero_rate"] == 0.0
    assert leaky_rows[0]["postactivation_negative_rate"] == 1.0


def test_only_declared_nonfinite_gradient_stops_are_recoverable() -> None:
    assert _nonfinite_gradient_epoch(
        "non-finite refit gradients at epoch=23, batch=139"
    ) == 23
    assert _nonfinite_gradient_epoch("non-finite refit loss at epoch=23, batch=139") is None
    assert _nonfinite_gradient_epoch("some unrelated failure") is None


def test_probability_ensemble_uses_fixed_half_weight_without_changing_identity() -> None:
    base = pd.DataFrame(
        {
            "id": [1, 2],
            "fold": [0, 0],
            "seed": [2753, 2753],
            "y_true": ["Fall", "Spring"],
            "prob_Fall": [0.8, 0.1],
            "prob_Spring": [0.1, 0.7],
            "prob_Summer": [0.05, 0.1],
            "prob_Winter": [0.05, 0.1],
        }
    )
    candidate = base.copy()
    candidate[["prob_Fall", "prob_Spring"]] = [[0.6, 0.3], [0.2, 0.6]]

    ensemble = _average_probability_oof(base, candidate)

    assert ensemble["id"].tolist() == [1, 2]
    assert ensemble["y_true"].tolist() == ["Fall", "Spring"]
    assert ensemble["y_pred"].tolist() == ["Fall", "Spring"]
    assert ensemble["prob_Fall"].tolist() == pytest.approx([0.7, 0.15])
    assert ensemble["prob_Spring"].tolist() == pytest.approx([0.2, 0.65])


def test_repository_post_submission_evidence_keeps_claims_bounded() -> None:
    evidence = ROOT / "results/evidence/task2/post_submission"
    activation = json.loads(
        (evidence / "relu_baseline_activation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    boosting = json.loads(
        (evidence / "boosting_summary.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(evidence / "boosted_model_comparison.csv").set_index(
        "model"
    )
    intervals = pd.read_csv(evidence / "boosted_paired_bootstrap_intervals.csv")

    assert activation["overall"]["validation_folds"] == 5
    assert activation["overall"]["channels_scanned_across_folds"] == 4_800
    assert activation["overall"]["exact_dead_channel_rate"] == 0.0
    assert activation["overall"]["near_dead_channel_rate"] == 0.0
    assert boosting["role"] == "post_submission_exploration_only"
    assert boosting["changes_frozen_final_or_submission"] is False
    assert boosting["winner"]["model"] == "I2 embedding + RF"
    assert comparison["rows"].eq(32_753).all()
    assert comparison.loc["I2 embedding + RF", "macro_f1"] > comparison.loc[
        "I2 ReLU", "macro_f1"
    ]
    assert comparison.loc["I2 embedding + HGB", "macro_f1"] < comparison.loc[
        "I2 ReLU", "macro_f1"
    ]

    rf_interval = intervals.loc[
        intervals["comparison"].eq("EmbeddingRF_minus_ReLU")
        & intervals["metric"].eq("macro_f1")
    ].iloc[0]
    hgb_interval = intervals.loc[
        intervals["comparison"].eq("EmbeddingHGB_minus_ReLU")
        & intervals["metric"].eq("macro_f1")
    ].iloc[0]
    assert rf_interval["ci95_lower"] > 0.0
    assert hgb_interval["ci95_lower"] < 0.0 < hgb_interval["ci95_upper"]
