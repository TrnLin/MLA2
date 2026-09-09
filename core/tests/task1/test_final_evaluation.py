import numpy as np
import pandas as pd
import pytest

from fashion.task1.final_evaluation import (
    blind_prediction_frame,
    grouped_intervals,
    score_tables,
    validate_predictions,
)

LABELS = [f"class_{i}" for i in range(124)]


def probabilities():
    p = np.full((4, 124), 0.01 / 123)
    p[np.arange(4), [0, 0, 1, 1]] = 0.99
    return p


def test_prediction_contract_rejects_wrong_ids_nonprobabilities_and_labels():
    frame = blind_prediction_frame([11, 12, 13, 14], probabilities(), LABELS)
    assert not any(name.startswith("true") for name in frame)
    validate_predictions(frame, [11, 12, 13, 14], LABELS)
    for mutate in (
        lambda f: f.assign(id=[11, 11, 13, 14]),
        lambda f: f.assign(id=[11.1, 12, 13, 14]),
        lambda f: f.assign(prob_000=-1),
        lambda f: f.assign(prob_000=np.nan),
        lambda f: f.assign(predicted_label="wrong"),
    ):
        with pytest.raises(ValueError):
            validate_predictions(mutate(frame.copy()), [11, 12, 13, 14], LABELS)
    with pytest.raises(ValueError):
        validate_predictions(frame, [14, 13, 12, 11], LABELS)


def test_score_tables_use_fixed_124_labels_and_development_bands():
    rows = pd.DataFrame(
        {
            "id": [11, 12, 13, 14],
            "articleType": LABELS[:2] * 2,
            "mode": ["L", "RGB", "RGB", "RGB"],
            "product_family_group": ["a", "a", "b", "c"],
        }
    )
    frames = score_tables(rows, probabilities(), LABELS, {LABELS[0]: 10, LABELS[1]: 200})
    assert len(frames["per_class"]) == 124
    assert frames["metrics"].iloc[0].macro_f1 == pytest.approx(1 / 124)
    assert frames["per_class"].support.sum() == 4
    assert set(frames["errors"].id) == {12, 13}
    assert frames["confusion_pairs"].errors.sum() == 2
    assert set(frames["slices"].slice) >= {
        "development_count_1_20",
        "development_count_over_100",
        "grayscale",
    }


def test_bootstrap_resamples_whole_families_and_is_repeatable():
    p = probabilities()
    y = np.array([0, 1, 0, 1])
    # One family: every bootstrap sample must contain all four related rows.
    first = grouped_intervals(y, p, ["family"] * 4, replicates=40, seed=2753)
    pd.testing.assert_frame_equal(
        first, grouped_intervals(y, p, ["family"] * 4, replicates=40, seed=2753)
    )
    accuracy = first.set_index("metric").loc["top1_accuracy"]
    assert accuracy.lower_95 == accuracy.upper_95 == 0.5
    assert first.set_index("metric").loc["macro_f1", "lower_95"] == pytest.approx(1 / 124)
