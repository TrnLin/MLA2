from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from fashion.train.artifacts import canonical_sha256
from fashion.train.losses import (
    EFFECTIVE_NUMBER_LOSS_ID,
    EffectiveNumberAudit,
    build_effective_number_cross_entropy,
    effective_number_class_weights,
    fit_effective_number_weights,
)


def test_effective_number_weights_match_paper_equation() -> None:
    counts = (100, 10, 1)
    beta = 0.9
    raw = [(1.0 - beta) / (1.0 - beta**count) for count in counts]
    expected = [value * len(raw) / sum(raw) for value in raw]

    weights = effective_number_class_weights(counts, beta=beta)

    assert weights == pytest.approx(expected)
    assert sum(weights) == pytest.approx(3.0)
    assert weights[2] > weights[1] > weights[0]


def test_beta_zero_reduces_to_unweighted_cross_entropy() -> None:
    assert effective_number_class_weights((100, 10, 1), beta=0.0) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("beta", [-0.1, 1.0, float("inf"), float("nan")])
def test_effective_number_weights_reject_invalid_beta(beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        effective_number_class_weights((10, 2), beta=beta)


@pytest.mark.parametrize(
    ("counts", "error_type"),
    [
        ((), ValueError),
        ((10,), ValueError),
        ((10, 0), ValueError),
        ((10, -1), ValueError),
        ((10, 1.5), TypeError),
        ((10, True), TypeError),
    ],
)
def test_effective_number_weights_reject_invalid_counts(
    counts: tuple[object, ...],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        effective_number_class_weights(counts)  # type: ignore[arg-type]


def test_fit_effective_number_weights_uses_canonical_label_order_and_training_ids() -> None:
    labels = ("Fall", "Spring", "Summer", "Winter")
    targets = ("Summer", "Fall", "Summer", "Spring", "Winter", "Summer")
    training_ids = (60, 10, 40, 20, 50, 30)

    audit = fit_effective_number_weights(
        targets,
        labels,
        training_ids=training_ids,
    )

    assert audit.labels == labels
    assert audit.class_counts == (1, 1, 3, 1)
    assert audit.training_product_count == 6
    assert audit.training_id_sha256 == canonical_sha256(sorted(training_ids))
    assert audit.loss_id == EFFECTIVE_NUMBER_LOSS_ID
    assert list(audit.to_dict()["class_counts"]) == list(labels)
    assert math.isclose(sum(audit.class_weights), 4.0, abs_tol=1e-12)


def test_fit_effective_number_weights_rejects_unknown_or_missing_classes() -> None:
    labels = ("Fall", "Spring", "Summer", "Winter")
    with pytest.raises(ValueError, match="outside the canonical order"):
        fit_effective_number_weights(
            ("Fall", "Spring", "Summer", "Winter", "Monsoon"),
            labels,
            training_ids=(1, 2, 3, 4, 5),
        )
    with pytest.raises(ValueError, match="missing canonical classes"):
        fit_effective_number_weights(
            ("Fall", "Summer", "Winter"),
            labels,
            training_ids=(1, 2, 3),
        )


def test_fit_effective_number_weights_rejects_misaligned_or_duplicate_ids() -> None:
    labels = ("Fall", "Spring", "Summer", "Winter")
    targets = labels
    with pytest.raises(ValueError, match="same length"):
        fit_effective_number_weights(targets, labels, training_ids=(1, 2, 3))
    with pytest.raises(ValueError, match="unique"):
        fit_effective_number_weights(targets, labels, training_ids=(1, 2, 2, 4))


def test_effective_number_cross_entropy_matches_pytorch_weighted_loss() -> None:
    audit = fit_effective_number_weights(
        ("Fall", "Spring", "Summer", "Summer", "Winter"),
        ("Fall", "Spring", "Summer", "Winter"),
        training_ids=(1, 2, 3, 4, 5),
    )
    logits = torch.tensor(
        [[2.0, 0.0, -1.0, 0.5], [0.1, 0.4, 1.2, -0.7]],
        requires_grad=True,
    )
    targets = torch.tensor([1, 2])

    criterion = build_effective_number_cross_entropy(audit)
    observed = criterion(logits, targets)
    expected = F.cross_entropy(
        logits,
        targets,
        weight=torch.tensor(audit.class_weights, dtype=torch.float32),
    )
    observed.backward()

    assert float(observed.detach()) == pytest.approx(float(expected.detach()))
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_effective_number_audit_rejects_tampered_weight_scale() -> None:
    audit = EffectiveNumberAudit(
        labels=("Fall", "Spring"),
        class_counts=(10, 2),
        class_weights=(1.0, 2.0),
        beta=0.9999,
        training_product_count=12,
        training_id_sha256="a" * 64,
        loss_id=EFFECTIVE_NUMBER_LOSS_ID,
    )

    with pytest.raises(ValueError, match="mean one"):
        audit.validate()
