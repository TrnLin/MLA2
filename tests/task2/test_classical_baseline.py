from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.task2.classical import (
    HogHsvSpec,
    evaluate_hog_hsv_svm_fold,
    extract_hog_hsv,
    fit_hog_hsv_svm,
)

LABELS = ("Fall", "Spring", "Summer", "Winter")


def _write_pattern(path: Path, colour: tuple[int, int, int], *, vertical: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((80, 60, 3), 255, dtype=np.uint8)
    if vertical:
        array[:, 20:40] = colour
    else:
        array[25:55, :] = colour
    Image.fromarray(array).save(path)


def _fold_frames(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    labels = ["Fall", "Fall", "Summer", "Summer", "Fall", "Summer"]
    for index, label in enumerate(labels, start=1):
        relative = Path("images") / f"{index}.png"
        _write_pattern(
            root / relative,
            (220, 30, 30) if label == "Fall" else (30, 60, 220),
            vertical=label == "Fall",
        )
        rows.append(
            {
                "id": index,
                "path": relative.as_posix(),
                "partition": "development",
                "cv_fold": 0 if index > 4 else 1,
                "season": label,
                "has_season_label": True,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.iloc[:4].copy(), frame.iloc[4:].copy()


def test_hog_hsv_feature_is_deterministic_and_has_normalized_colour_bins(
    tmp_path: Path,
) -> None:
    image = tmp_path / "images" / "sample.png"
    _write_pattern(image, (220, 30, 30), vertical=True)
    spec = HogHsvSpec(image_size=(80, 60))

    first = extract_hog_hsv(image, spec)
    second = extract_hog_hsv(image, spec)
    colour = first[-sum(spec.hsv_bins) :]

    assert np.array_equal(first, second)
    assert first.dtype == np.float32
    assert len(first) > sum(spec.hsv_bins)
    assert colour[:18].sum() == pytest.approx(1.0)
    assert colour[18:26].sum() == pytest.approx(1.0)
    assert colour[26:].sum() == pytest.approx(1.0)


def test_hog_hsv_svm_builds_fixed_label_oof_scores(tmp_path: Path) -> None:
    training, validation = _fold_frames(tmp_path)

    result = evaluate_hog_hsv_svm_fold(
        training,
        validation,
        validation_fold=0,
        root=tmp_path,
    )

    assert result.model.training_product_count == 4
    assert result.model.labels == LABELS
    assert result.model.feature_count > sum(result.model.spec.hsv_bins)
    assert result.oof["id"].tolist() == [5, 6]
    assert result.oof["y_pred"].tolist() == ["Fall", "Summer"]
    probability_columns = [f"prob_{label}" for label in LABELS]
    assert np.allclose(result.oof[probability_columns].sum(axis=1), 1.0)
    assert result.metrics["macro_f1"] == pytest.approx(0.5)
    assert result.metrics["probability_method"] == (
        "uncalibrated_softmax_of_decision_scores"
    )


def test_validation_labels_do_not_change_fitted_svm(tmp_path: Path) -> None:
    training, validation = _fold_frames(tmp_path)
    original = evaluate_hog_hsv_svm_fold(
        training,
        validation,
        validation_fold=0,
        root=tmp_path,
    )
    changed_labels = validation.copy()
    changed_labels["season"] = ["Winter", "Spring"]
    changed = evaluate_hog_hsv_svm_fold(
        training,
        changed_labels,
        validation_fold=0,
        root=tmp_path,
    )

    assert original.model.training_id_sha256 == changed.model.training_id_sha256
    assert np.allclose(
        original.oof[[f"prob_{label}" for label in LABELS]],
        changed.oof[[f"prob_{label}" for label in LABELS]],
    )


def test_svm_rejects_single_observed_training_class(tmp_path: Path) -> None:
    training, _ = _fold_frames(tmp_path)
    training["season"] = "Summer"

    with pytest.raises(ValueError, match="at least two"):
        fit_hog_hsv_svm(training, root=tmp_path)


def test_svm_rejects_image_path_escape(tmp_path: Path) -> None:
    training, _ = _fold_frames(tmp_path)
    training.loc[0, "path"] = "../outside.png"

    with pytest.raises(ValueError, match="escapes"):
        fit_hog_hsv_svm(training, root=tmp_path)
