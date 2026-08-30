from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from fashion.task1.models import MpsSafeAdaptiveAvgPool2d, Task1SmallCNN, count_trainable_parameters


def test_small_cnn_maps_fashion_images_to_124_logits():
    model = Task1SmallCNN(num_classes=124)
    output = model(torch.zeros(2, 3, 80, 60))
    assert output.shape == (2, 124)
    assert count_trainable_parameters(model) == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_small_cnn_maps_fashion_images_to_124_logits_on_mps():
    model = Task1SmallCNN(num_classes=124).to("mps")
    output = model(torch.zeros(2, 3, 80, 60, device="mps"))
    assert output.shape == (2, 124)


def test_mps_safe_adaptive_pool_matches_pytorch_and_backpropagates():
    inputs = torch.arange(2 * 3 * 10 * 7, dtype=torch.float32).reshape(2, 3, 10, 7)
    inputs.requires_grad_()

    output = MpsSafeAdaptiveAvgPool2d((4, 3))(inputs)
    expected = functional.adaptive_avg_pool2d(inputs, (4, 3))

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().mean().backward()
    assert inputs.grad is not None
    assert torch.count_nonzero(inputs.grad) > 0


def test_small_cnn_uses_distinct_late_convolutions():
    model = Task1SmallCNN(num_classes=124)
    assert model.conv4 is not model.conv5


def test_small_cnn_rejects_wrong_channel_count():
    model = Task1SmallCNN(num_classes=124)
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.zeros(2, 1, 80, 60))
