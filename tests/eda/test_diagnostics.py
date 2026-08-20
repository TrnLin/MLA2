from __future__ import annotations

import pandas as pd
import pytest

from fashion.eda.diagnostics import (
    build_split_balance_inputs,
    build_target_validation_diagnostic,
    build_validation_diagnostics,
    summarize_target,
)


def test_balance_uses_train_selection_and_local_denominators():
    splits = pd.DataFrame(
        {
            "partition": ["train"] * 3 + ["val"] * 4 + ["holdout"] * 20,
            "articleType": ["A", "A", "B"] + ["A", "B", "B", "B"] + ["Leak"] * 20,
            "has_articleType_label": [True] * 27,
        }
    )
    balance, denominators = build_split_balance_inputs(splits, top_n=1)
    assert balance.index.tolist() == ["A"]
    assert denominators == {"train": 3, "val": 4}
    assert balance.loc["A", "train"] == pytest.approx(2 / 3 * 100)
    assert balance.loc["A", "val"] == pytest.approx(1 / 4 * 100)


def test_validation_diagnostic_does_not_read_holdout_classes():
    splits = pd.DataFrame(
        {
            "partition": ["train", "val", "holdout", "quarantine"],
            "articleType": ["A", "B", "Secret", "SecretTwin"],
            "has_articleType_label": [True] * 4,
        }
    )
    diagnostic = build_target_validation_diagnostic(splits, "articleType")
    assert diagnostic["classes"] == ["A"]
    assert "Secret" not in str(diagnostic)
    assert diagnostic["validation_labels_outside_training_classes"] == 1


def test_all_target_diagnostics_are_built(prepared_project):
    diagnostics = build_validation_diagnostics(pd.read_csv(prepared_project.splits))
    assert set(diagnostics) == {"articleType", "season", "gender", "usage"}
    assert diagnostics["articleType"]["training_class_count"] >= 1


def test_target_summary_is_training_scoped(prepared_project):
    splits = pd.read_csv(prepared_project.splits)
    train = splits[splits["partition"].eq("train")]
    summary = summarize_target(train, "usage")
    assert summary["source_partition"] == "train"
    assert summary["valid_labels"] + summary["missing_labels"] == len(train)
