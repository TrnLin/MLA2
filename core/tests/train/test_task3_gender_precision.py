"""Precision checks preserve controls and flag even small prediction changes."""

from types import SimpleNamespace

import numpy as np
import pytest

from fashion.train.task3_gender_precision import (
    ieee_precision,
    precision_settings,
    probability_difference,
)


@pytest.mark.parametrize("fail", [False, True])
def test_precision_settings_restore_after_success_or_error(fail):
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            fp32_precision="ieee",
            cuda=SimpleNamespace(matmul=SimpleNamespace(fp32_precision="ieee")),
            cudnn=SimpleNamespace(
                fp32_precision="tf32",
                conv=SimpleNamespace(fp32_precision="tf32"),
                rnn=SimpleNamespace(fp32_precision="ieee"),
            ),
        )
    )
    before = precision_settings(torch)
    try:
        with ieee_precision(torch):
            assert set(precision_settings(torch).values()) == {"ieee"}
            if fail:
                raise RuntimeError("inference failed")
    except RuntimeError:
        assert fail
    assert precision_settings(torch) == before


def test_probability_tolerance_does_not_hide_label_flips():
    a = np.array([[0.500001, 0.499999]])
    b = np.array([[0.499999, 0.500001]])
    result = probability_difference(a, b)
    assert result["max_abs_difference"] < 1e-5
    assert result["prediction_flips"] == 1
    assert not result["pass"]
    assert probability_difference(a, a)["pass"]
    assert not probability_difference(a, a + 2e-5)["pass"]
    with pytest.raises(ValueError, match="Non-finite"):
        probability_difference(a, a * np.nan)
    with pytest.raises(ValueError, match="matching"):
        probability_difference(a, a.T)
