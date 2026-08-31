from __future__ import annotations

import importlib

import pytest

from fashion.train.config import Task3BaselineConfig

torch = pytest.importorskip("torch")
nn = torch.nn
Task3BaselineCNN = importlib.import_module("fashion.train.model").Task3BaselineCNN
Task3TinyResNet18PM = importlib.import_module("fashion.train.model").Task3TinyResNet18PM
_pass = importlib.import_module("fashion.train.task3_baseline")._pass


def test_classifier_dropout_changes_behaviour_without_changing_parameters() -> None:
    config = Task3BaselineConfig(target="usage")
    baseline = Task3BaselineCNN(config)
    child = Task3BaselineCNN(config, classifier_dropout=0.2)

    assert isinstance(baseline.classifier_dropout, nn.Identity)
    assert isinstance(child.classifier_dropout, nn.Dropout)
    assert child.classifier_dropout.p == pytest.approx(0.2)
    assert sum(parameter.numel() for parameter in baseline.parameters()) == sum(
        parameter.numel() for parameter in child.parameters()
    )
    with pytest.raises(ValueError, match="classifier dropout"):
        Task3BaselineCNN(config, classifier_dropout=1.0)


@pytest.mark.parametrize("target,expected", [("gender", 394_865), ("usage", 395_253)])
def test_tinyresnet_is_parameter_matched_and_preserves_the_native_stem(
    target: str, expected: int
) -> None:
    config = Task3BaselineConfig(target=target)
    model = Task3TinyResNet18PM(config)

    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    assert model.stem[0].stride == (1, 1)
    assert not any(isinstance(module, nn.MaxPool2d) for module in model.modules())
    assert tuple(model(torch.zeros(2, 3, 80, 60)).shape) == (2, config.num_classes)


def test_weighted_epoch_loss_uses_the_sum_of_sample_weights() -> None:
    logits = torch.tensor(
        [[2.0, -1.0], [-0.5, 0.5], [0.2, 0.8], [1.0, -0.2]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    weights = torch.tensor([1.0, 4.0], dtype=torch.float32)

    class LookupModel(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return logits[images[:, 0].long()]

    loader = [
        {
            "image": torch.tensor([[0.0], [1.0]]),
            "label": labels[:2],
            "id": torch.tensor([10, 11]),
            "cv_fold": torch.tensor([0, 0]),
            "product_family_group": ["a", "b"],
            "path": ["a.jpg", "b.jpg"],
        },
        {
            "image": torch.tensor([[2.0], [3.0]]),
            "label": labels[2:],
            "id": torch.tensor([12, 13]),
            "cv_fold": torch.tensor([0, 0]),
            "product_family_group": ["c", "d"],
            "path": ["c.jpg", "d.jpg"],
        },
    ]
    criterion = nn.CrossEntropyLoss(weight=weights)

    observed, _, _, _ = _pass(LookupModel(), loader, criterion, torch.device("cpu"))
    per_sample = nn.functional.cross_entropy(logits, labels, weight=weights, reduction="none")
    expected = float(per_sample.sum() / weights[labels].sum())

    assert observed == pytest.approx(expected)
