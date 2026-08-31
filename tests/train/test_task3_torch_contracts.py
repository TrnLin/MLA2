from __future__ import annotations

import importlib

import pytest

from fashion.train.config import Task3BaselineConfig

torch = pytest.importorskip("torch")
nn = torch.nn
Task3BaselineCNN = importlib.import_module("fashion.train.model").Task3BaselineCNN
Task3CompactBlurCNN = importlib.import_module("fashion.train.model").Task3CompactBlurCNN
Task3GeM3CNN = importlib.import_module("fashion.train.model").Task3GeM3CNN
Task3TinyConvNeXt18 = importlib.import_module("fashion.train.model").Task3TinyConvNeXt18
Task3TinyHRNet20 = importlib.import_module("fashion.train.model").Task3TinyHRNet20
Task3TinyResNet18PM = importlib.import_module("fashion.train.model").Task3TinyResNet18PM
WeightedFocalCrossEntropy = importlib.import_module(
    "fashion.train.loss"
).WeightedFocalCrossEntropy
WeightedLabelSmoothedCrossEntropy = importlib.import_module(
    "fashion.train.loss"
).WeightedLabelSmoothedCrossEntropy
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


def test_compact_blur_cnn_is_low_capacity_and_keeps_a_ten_by_eight_grid() -> None:
    config = Task3BaselineConfig(target="gender")
    model = Task3CompactBlurCNN(config)

    assert sum(parameter.numel() for parameter in model.parameters()) == 67_069
    assert not any(isinstance(module, nn.MaxPool2d) for module in model.modules())
    assert tuple(model.features(torch.zeros(2, 3, 80, 60)).shape) == (2, 128, 10, 8)
    assert tuple(model(torch.zeros(2, 3, 80, 60)).shape) == (2, 5)


def test_gem_p3_changes_only_global_pooling_and_keeps_the_parameter_count() -> None:
    config = Task3BaselineConfig(target="gender")
    baseline = Task3BaselineCNN(config)
    child = Task3GeM3CNN(config)
    features = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    observed = child.pool(features)
    expected = features.pow(3.0).mean(dim=(-2, -1), keepdim=True).pow(1.0 / 3.0)

    assert observed == pytest.approx(expected)
    assert sum(parameter.numel() for parameter in child.parameters()) == sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert tuple(child(torch.zeros(2, 3, 80, 60)).shape) == (2, 5)


def test_tinyhrnet_keeps_three_resolutions_and_matches_the_e7_budget() -> None:
    config = Task3BaselineConfig(target="gender")
    model = Task3TinyHRNet20(config)

    with torch.inference_mode():
        branches = model.feature_branches(torch.zeros(2, 3, 80, 60))

    assert sum(parameter.numel() for parameter in model.parameters()) == 374_445
    assert [tuple(branch.shape) for branch in branches] == [
        (2, 20, 40, 30),
        (2, 40, 20, 15),
        (2, 80, 10, 8),
    ]
    assert not any(isinstance(module, nn.MaxPool2d) for module in model.modules())
    assert tuple(model(torch.zeros(2, 3, 80, 60)).shape) == (2, 5)


def test_tinyconvnext_uses_large_depthwise_kernels_and_matches_the_e7_budget() -> None:
    config = Task3BaselineConfig(target="usage")
    model = Task3TinyConvNeXt18(config)

    depthwise = [
        module
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
        and module.kernel_size == (7, 7)
        and module.groups == module.in_channels
    ]

    assert sum(parameter.numel() for parameter in model.parameters()) == 384_345
    assert len(depthwise) == 6
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert tuple(model(torch.zeros(2, 3, 80, 60)).shape) == (2, 9)


def test_weighted_label_smoothing_weights_examples_by_the_true_class_only() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, 1.2, -0.4]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    weights = torch.tensor([1.0, 4.0, 2.0], dtype=torch.float32)
    criterion = WeightedLabelSmoothedCrossEntropy(weights, epsilon=0.05)

    observed = criterion(logits, labels)
    soft_targets = torch.tensor(
        [[0.95, 0.025, 0.025], [0.025, 0.95, 0.025]], dtype=torch.float32
    )
    per_sample = -(soft_targets * torch.log_softmax(logits, dim=1)).sum(dim=1)
    expected = (per_sample * weights[labels]).sum() / weights[labels].sum()

    assert observed == pytest.approx(float(expected))
    assert criterion.loss_denominator(labels) == pytest.approx(5.0)


def test_weighted_focal_gamma_one_uses_true_class_weights_and_probabilities() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, 1.2, -0.4]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    weights = torch.tensor([1.0, 4.0, 2.0], dtype=torch.float32)
    criterion = WeightedFocalCrossEntropy(weights, gamma=1.0)

    observed = criterion(logits, labels)
    log_probabilities = torch.log_softmax(logits, dim=1)
    true_log_probability = log_probabilities[torch.arange(2), labels]
    true_probability = true_log_probability.exp()
    per_sample = -(1.0 - true_probability) * true_log_probability
    expected = (per_sample * weights[labels]).sum() / weights[labels].sum()

    assert observed == pytest.approx(float(expected))
    assert criterion.loss_denominator(labels) == pytest.approx(5.0)


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
