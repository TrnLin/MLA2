from __future__ import annotations

import pytest
import torch

from fashion.task1.models import Task1SmallCNN, count_trainable_parameters


def test_small_cnn_maps_fashion_images_to_124_logits():
    model = Task1SmallCNN(num_classes=124)
    output = model(torch.zeros(2, 3, 80, 60))
    assert output.shape == (2, 124)
    assert count_trainable_parameters(model) == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def test_small_cnn_uses_distinct_late_convolutions():
    model = Task1SmallCNN(num_classes=124)
    assert model.conv4 is not model.conv5


def test_small_cnn_rejects_wrong_channel_count():
    model = Task1SmallCNN(num_classes=124)
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.zeros(2, 1, 80, 60))
