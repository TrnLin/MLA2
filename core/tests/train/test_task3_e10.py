from __future__ import annotations

import numpy as np
import pytest

from fashion.train.task3_e10 import (
    GENDER_INDEX_TO_AUDIENCE_INDEX,
    gender_audience_metrics,
    gender_audience_probabilities,
)


def test_gender_audience_probabilities_preserve_probability_mass() -> None:
    probabilities = np.asarray(
        [
            [0.10, 0.20, 0.30, 0.15, 0.25],
            [0.60, 0.05, 0.10, 0.20, 0.05],
        ]
    )

    observed = gender_audience_probabilities(probabilities)

    assert observed == pytest.approx(np.asarray([[0.40, 0.45, 0.15], [0.70, 0.10, 0.20]]))
    assert observed.sum(axis=1) == pytest.approx(np.ones(2))


def test_gender_audience_metric_uses_the_fixed_five_to_three_mapping() -> None:
    labels = np.arange(5)
    probabilities = np.eye(5)

    metrics = gender_audience_metrics(labels, probabilities)

    assert GENDER_INDEX_TO_AUDIENCE_INDEX.tolist() == [0, 1, 0, 2, 1]
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
