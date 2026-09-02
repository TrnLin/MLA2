import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_recall_fscore_support

from fashion.task1.evaluation import (
    aggregate_fold_metrics,
    build_prediction_frame,
    classification_metrics,
    per_class_metrics,
    validate_oof_predictions,
)

NUM_CLASSES = 124


def _probabilities() -> np.ndarray:
    probabilities = np.zeros((3, NUM_CLASSES), dtype=float)
    probabilities[0, 4] = 0.8
    probabilities[0, 5] = 0.1
    probabilities[0, 6] = 0.1
    probabilities[1, 8] = 0.7
    probabilities[1, 9] = 0.2
    probabilities[1, 10] = 0.1
    probabilities[2, 12] = 0.6
    probabilities[2, 13] = 0.3
    probabilities[2, 14] = 0.1
    probabilities[2, 11] = 0.05
    return probabilities


def test_classification_metrics_use_all_fixed_classes_and_top_k() -> None:
    y_true = np.array([4, 9, 12])
    probabilities = _probabilities()
    expected_macro = f1_score(
        y_true,
        probabilities.argmax(axis=1),
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )

    metrics = classification_metrics(y_true, probabilities, num_classes=NUM_CLASSES)

    assert metrics["macro_f1"] == pytest.approx(expected_macro)
    assert metrics["top1_accuracy"] == pytest.approx(2 / 3)
    assert metrics["top5_accuracy"] == pytest.approx(1.0)


def test_per_class_metrics_expose_every_class() -> None:
    y_true = np.array([4, 9, 12])
    probabilities = _probabilities()
    class_names = [f"class-{index}" for index in range(NUM_CLASSES)]

    frame = per_class_metrics(y_true, probabilities, class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        probabilities.argmax(axis=1),
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )

    assert list(frame.columns) == [
        "class_index",
        "class_name",
        "precision",
        "recall",
        "f1",
        "support",
    ]
    assert len(frame) == NUM_CLASSES
    np.testing.assert_allclose(frame["precision"], precision)
    np.testing.assert_allclose(frame["recall"], recall)
    np.testing.assert_allclose(frame["f1"], f1)
    np.testing.assert_array_equal(frame["support"], support)


def test_build_prediction_frame_contains_labels_and_all_probabilities() -> None:
    y_true = np.array([4, 9, 12])
    probabilities = _probabilities()
    class_names = [f"class-{index}" for index in range(NUM_CLASSES)]

    frame = build_prediction_frame(np.array([101, 102, 103]), y_true, probabilities, class_names)

    assert list(frame.columns[:5]) == [
        "id",
        "true_index",
        "predicted_index",
        "true_label",
        "predicted_label",
    ]
    assert list(frame.columns[5:]) == [f"prob_{index:03d}" for index in range(NUM_CLASSES)]
    assert frame.loc[0, "true_label"] == "class-4"
    assert frame.loc[0, "predicted_label"] == "class-4"
    assert frame.loc[2, "predicted_label"] == "class-12"


@pytest.mark.parametrize(
    ("labels", "probabilities", "message"),
    [
        (np.array([0, 1]), np.zeros((2, NUM_CLASSES - 1)), "shapes do not align"),
        (np.array([0, 124]), np.zeros((2, NUM_CLASSES)), "valid class indexes"),
        (np.array([0, 1]), np.array([[0.0] * NUM_CLASSES, [np.nan] * NUM_CLASSES]), "finite"),
    ],
)
def test_classification_metrics_reject_invalid_arrays(
    labels: np.ndarray,
    probabilities: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        classification_metrics(labels, probabilities)


def test_prediction_frame_rejects_wrong_class_name_count() -> None:
    with pytest.raises(ValueError, match="124 class names"):
        build_prediction_frame(
            np.array([101]),
            np.array([0]),
            np.zeros((1, NUM_CLASSES)),
            ["only-one"],
        )


def test_prediction_frame_rejects_duplicate_product_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_prediction_frame(
            np.array([101, 101]),
            np.array([0, 1]),
            np.zeros((2, NUM_CLASSES)),
            [f"class-{index}" for index in range(NUM_CLASSES)],
        )


def test_validate_oof_predictions_rejects_duplicate_ids() -> None:
    predictions = pd.DataFrame({"id": [101, 101]})

    with pytest.raises(ValueError, match="duplicate"):
        validate_oof_predictions(predictions, [101, 102])


def test_validate_oof_predictions_rejects_missing_expected_ids() -> None:
    predictions = pd.DataFrame({"id": [101]})

    with pytest.raises(ValueError, match="expected development IDs"):
        validate_oof_predictions(predictions, [101, 102])


def test_aggregate_fold_metrics_returns_mean_and_sample_std() -> None:
    fold_metrics = [
        {"macro_f1": 0.1, "top1_accuracy": 0.2},
        {"macro_f1": 0.2, "top1_accuracy": 0.4},
        {"macro_f1": 0.3, "top1_accuracy": 0.6},
        {"macro_f1": 0.4, "top1_accuracy": 0.8},
        {"macro_f1": 0.5, "top1_accuracy": 1.0},
    ]

    frame = aggregate_fold_metrics(fold_metrics)

    assert list(frame.columns) == ["mean", "std"]
    assert frame.loc["macro_f1", "mean"] == pytest.approx(0.3)
    assert frame.loc["macro_f1", "std"] == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4, 0.5], ddof=1))


def test_aggregate_fold_metrics_requires_five_folds() -> None:
    with pytest.raises(ValueError, match="exactly five folds"):
        aggregate_fold_metrics([{"macro_f1": 0.1}])


def test_aggregate_fold_metrics_requires_unique_fold_labels() -> None:
    rows = [{"fold": fold, "macro_f1": 0.1} for fold in [0, 1, 2, 3, 3]]
    with pytest.raises(ValueError, match="unique labels"):
        from fashion.task1.cnn_experiments import _aggregate_comparison

        _aggregate_comparison(
            pd.DataFrame(
                {
                    "preprocessing_id": ["candidate"] * 5,
                    "fold": [row["fold"] for row in rows],
                    "macro_f1": [row["macro_f1"] for row in rows],
                }
            ),
            ["candidate"],
        )
