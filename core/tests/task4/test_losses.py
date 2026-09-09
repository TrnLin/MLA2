from __future__ import annotations

import pytest
import torch

import fashion.task4 as task4
from fashion.task4 import losses as task4_losses
from fashion.task4.losses import (
    batch_hard_family_triplet_loss,
    content_mask_mse_loss,
    family_triplet_masks,
    r4_loss,
    vicreg_loss,
)


def test_vicreg_reports_hand_checked_components_and_exact_weights() -> None:
    first = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    second = first.clone()

    loss = vicreg_loss(first, second, variance_epsilon=0.0)

    assert loss.invariance.item() == pytest.approx(0.0)
    assert loss.variance.item() == pytest.approx(0.0)
    assert loss.covariance.item() == pytest.approx(8.0)
    assert loss.total.item() == pytest.approx(8.0)
    assert loss.invariance_weight == 25.0
    assert loss.variance_weight == 25.0
    assert loss.covariance_weight == 1.0


def test_vicreg_variance_penalizes_collapsed_projection() -> None:
    collapsed = torch.zeros(4, 3)
    spread = torch.tensor(
        [
            [-2.0, -2.0, -2.0],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )

    collapsed_loss = vicreg_loss(collapsed, collapsed)
    spread_loss = vicreg_loss(spread, spread)

    assert collapsed_loss.variance > spread_loss.variance
    assert collapsed_loss.variance.item() == pytest.approx(0.99)


def test_vicreg_hand_calculation_pins_all_nonzero_components_and_total() -> None:
    first = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    second = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    loss = vicreg_loss(first, second, variance_epsilon=0.0)

    assert loss.invariance.item() == pytest.approx(0.75)
    assert loss.variance.item() == pytest.approx(0.3232233047)
    assert loss.covariance.item() == pytest.approx(4.0)
    assert loss.total.item() == pytest.approx(30.83058262)


def test_vicreg_has_finite_gradients() -> None:
    first = torch.randn(5, 8, requires_grad=True)
    second = torch.randn(5, 8, requires_grad=True)

    vicreg_loss(first, second).total.backward()

    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert second.grad is not None and torch.isfinite(second.grad).all()


@pytest.mark.parametrize(
    ("first_shape", "second_shape"),
    [
        ((1, 8), (1, 8)),
        ((3, 8), (2, 8)),
        ((3, 8, 1), (3, 8, 1)),
    ],
)
def test_vicreg_rejects_invalid_batches(
    first_shape: tuple[int, ...],
    second_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="batch|shape|2D"):
        vicreg_loss(torch.zeros(first_shape), torch.zeros(second_shape))


def test_family_triplet_masks_apply_identity_family_and_duplicate_rules() -> None:
    positive, negative = family_triplet_masks(
        product_ids=[1, 2, 3, 4, 5],
        family_groups=["a", "a", "a", "b", "c"],
        sha256=["x", "y", "x", "z", "w"],
        duplicate_groups=["d1", "d2", "d3", "d4", "d5"],
    )

    assert positive.tolist() == [
        [False, True, False, False, False],
        [True, False, True, False, False],
        [False, True, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
    ]
    assert negative.tolist() == [
        [False, False, False, True, True],
        [False, False, False, True, True],
        [False, False, False, True, True],
        [True, True, True, False, True],
        [True, True, True, True, False],
    ]


def test_family_triplet_masks_exclude_shared_duplicate_group() -> None:
    positive, _ = family_triplet_masks(
        product_ids=[1, 2],
        family_groups=["a", "a"],
        sha256=["x", "y"],
        duplicate_groups=["same", "same"],
    )

    assert not positive.any()


def test_family_triplet_masks_accepts_explicit_output_device() -> None:
    positive, negative = family_triplet_masks(
        product_ids=[1, 2],
        family_groups=["a", "b"],
        sha256=["x", "y"],
        duplicate_groups=["d1", "d2"],
        device=torch.device("cpu"),
    )

    assert positive.device == torch.device("cpu")
    assert negative.device == torch.device("cpu")


def test_family_triplet_mask_assignments_are_cpu_before_one_target_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_zeros = torch.zeros
    spies: list[_MaskConstructionSpy] = []

    def tracked_zeros(*shape: object, **kwargs: object) -> _MaskConstructionSpy:
        requested_device = kwargs.pop("device", None)
        assert requested_device in {None, torch.device("cpu"), "cpu"}
        backing = real_zeros(*shape, **kwargs, device="cpu")  # type: ignore[arg-type]
        spy = _MaskConstructionSpy(backing)
        spies.append(spy)
        return spy

    monkeypatch.setattr(task4_losses.torch, "zeros", tracked_zeros)

    positive, negative = family_triplet_masks(
        product_ids=[1, 2, 3],
        family_groups=["a", "a", "b"],
        sha256=["s1", "s2", "s3"],
        duplicate_groups=["d1", "d2", "d3"],
        device=torch.device("meta"),
    )

    assert len(spies) == 2
    assert all(spy.assignment_devices and set(spy.assignment_devices) == {"cpu"} for spy in spies)
    assert [spy.transfer_devices for spy in spies] == [["meta"], ["meta"]]
    assert positive.device.type == "meta"
    assert negative.device.type == "meta"


class _MaskConstructionSpy:
    def __init__(self, backing: torch.Tensor) -> None:
        self.backing = backing
        self.assignment_devices: list[str] = []
        self.transfer_devices: list[str] = []

    def __setitem__(self, key: object, value: object) -> None:
        self.assignment_devices.append(self.backing.device.type)
        self.backing[key] = value

    def to(self, device: torch.device | str) -> torch.Tensor:
        target = torch.device(device)
        self.transfer_devices.append(target.type)
        return self.backing.to(target)


def test_batch_hard_triplet_chooses_hardest_valid_distances_and_margin() -> None:
    embeddings = torch.tensor([[0.0], [1.0], [3.0], [2.0], [4.0]])

    loss = batch_hard_family_triplet_loss(
        embeddings,
        product_ids=[1, 2, 3, 4, 5],
        family_groups=["a", "a", "a", "b", "c"],
        sha256=["s1", "s2", "s3", "s4", "s5"],
        duplicate_groups=["d1", "d2", "d3", "d4", "d5"],
    )

    assert loss.valid_anchor_mask.tolist() == [True, True, True, False, False]
    assert loss.hardest_positive.tolist() == pytest.approx([3.0, 2.0, 3.0])
    assert loss.hardest_negative.tolist() == pytest.approx([2.0, 1.0, 1.0])
    assert loss.per_anchor.tolist() == pytest.approx([1.2, 1.2, 2.2])
    assert loss.total.item() == pytest.approx(4.6 / 3)
    assert loss.margin == 0.2


def test_batch_hard_triplet_skips_anchor_without_negative() -> None:
    loss = batch_hard_family_triplet_loss(
        torch.tensor([[0.0], [1.0]]),
        product_ids=[1, 2],
        family_groups=["a", "a"],
        sha256=["s1", "s2"],
        duplicate_groups=["d1", "d2"],
    )

    assert loss.valid_anchor_mask.tolist() == [False, False]


def test_empty_triplet_result_is_differentiable_on_input_device_and_dtype() -> None:
    embeddings = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)

    loss = batch_hard_family_triplet_loss(
        embeddings,
        product_ids=[1, 2, 3],
        family_groups=["a", "b", "c"],
        sha256=["s1", "s2", "s3"],
        duplicate_groups=["d1", "d2", "d3"],
    )
    loss.total.backward()

    assert loss.total.shape == ()
    assert loss.total.device == embeddings.device
    assert loss.total.dtype == embeddings.dtype
    assert embeddings.grad is not None
    assert torch.equal(embeddings.grad, torch.zeros_like(embeddings))


def test_batch_hard_triplet_has_finite_gradients() -> None:
    embeddings = torch.randn(4, 8, requires_grad=True)

    loss = batch_hard_family_triplet_loss(
        embeddings,
        product_ids=[1, 2, 3, 4],
        family_groups=["a", "a", "b", "b"],
        sha256=["s1", "s2", "s3", "s4"],
        duplicate_groups=["d1", "d2", "d3", "d4"],
    )
    loss.total.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_r4_combines_vicreg_and_unit_weight_triplet() -> None:
    first = torch.tensor(
        [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]],
        requires_grad=True,
    )
    second = first.detach().clone().requires_grad_()

    loss = r4_loss(
        first,
        second,
        triplet_embeddings=first,
        product_ids=[1, 2, 3, 4],
        family_groups=["a", "a", "b", "c"],
        sha256=["s1", "s2", "s3", "s4"],
        duplicate_groups=["d1", "d2", "d3", "d4"],
    )

    assert loss.triplet_weight == 1.0
    assert loss.total.item() == pytest.approx(
        loss.vicreg.total.item() + loss.triplet.total.item()
    )


def test_content_mask_mse_uses_only_included_pixels() -> None:
    reconstruction = torch.tensor(
        [[[[2.0, 100.0], [4.0, 100.0]], [[1.0, 100.0], [3.0, 100.0]]]]
    )
    target = torch.zeros_like(reconstruction)
    mask = torch.tensor([[[True, False], [True, False]]])

    loss = content_mask_mse_loss(reconstruction, target, mask)

    assert loss.total.item() == pytest.approx((4.0 + 16.0 + 1.0 + 9.0) / 4)
    assert loss.included_values == 4


def test_content_mask_mse_backward_limits_gradients_to_included_pixels() -> None:
    reconstruction = torch.tensor(
        [[[[2.0, 5.0], [4.0, 7.0]], [[1.0, 6.0], [3.0, 8.0]]]],
        requires_grad=True,
    )
    target = torch.zeros_like(reconstruction)
    mask = torch.tensor([[[True, False], [True, False]]])

    content_mask_mse_loss(reconstruction, target, mask).total.backward()

    assert reconstruction.grad is not None
    included = reconstruction.grad[:, :, :, 0]
    excluded = reconstruction.grad[:, :, :, 1]
    assert torch.isfinite(included).all()
    assert torch.all(included != 0)
    assert torch.equal(excluded, torch.zeros_like(excluded))


def test_content_mask_mse_accepts_singleton_channel_mask() -> None:
    reconstruction = torch.tensor([[[[2.0, 100.0], [4.0, 100.0]]]])
    target = torch.zeros_like(reconstruction)
    mask = torch.tensor([[[[True, False], [True, False]]]])

    loss = content_mask_mse_loss(reconstruction, target, mask)

    assert loss.total.item() == pytest.approx(10.0)
    assert loss.included_values == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triplet_masks_diagnostics_and_total_stay_on_cuda() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    product_ids = torch.tensor([1, 2, 3, 4], device=device)
    positive, negative = family_triplet_masks(
        product_ids=product_ids,
        family_groups=["a", "a", "b", "b"],
        sha256=["s1", "s2", "s3", "s4"],
        duplicate_groups=["d1", "d2", "d3", "d4"],
        device=device,
    )
    embeddings = torch.randn(4, 8, device=device, requires_grad=True)

    loss = batch_hard_family_triplet_loss(
        embeddings,
        product_ids=product_ids,
        family_groups=["a", "a", "b", "b"],
        sha256=["s1", "s2", "s3", "s4"],
        duplicate_groups=["d1", "d2", "d3", "d4"],
    )

    assert positive.device == device
    assert negative.device == device
    assert loss.valid_anchor_mask.device == device
    assert loss.hardest_positive.device == device
    assert loss.hardest_negative.device == device
    assert loss.per_anchor.device == device
    assert loss.total.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_content_mask_mse_keeps_mask_and_total_on_cuda() -> None:
    reconstruction = torch.ones((1, 2, 2, 2), device="cuda", requires_grad=True)
    target = torch.zeros_like(reconstruction)
    mask = torch.tensor([[[[True, False], [True, False]]]], device="cuda")

    loss = content_mask_mse_loss(reconstruction, target, mask)
    loss.total.backward()

    assert loss.total.device.type == "cuda"
    assert reconstruction.grad is not None
    assert reconstruction.grad.device.type == "cuda"


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros((1, 2, 2), dtype=torch.bool),
        torch.ones((2, 2), dtype=torch.bool),
        torch.ones((1, 2, 3), dtype=torch.bool),
        torch.ones((1, 2, 2), dtype=torch.float32),
        torch.ones((1, 2, 2, 2), dtype=torch.bool),
    ],
)
def test_content_mask_mse_rejects_empty_or_malformed_masks(mask: torch.Tensor) -> None:
    values = torch.zeros((1, 3, 2, 2))

    with pytest.raises(ValueError, match="mask"):
        content_mask_mse_loss(values, values, mask)


def test_task4_package_exports_learned_loss_api() -> None:
    assert task4.vicreg_loss is vicreg_loss
    assert task4.batch_hard_family_triplet_loss is batch_hard_family_triplet_loss
    assert task4.r4_loss is r4_loss
    assert task4.content_mask_mse_loss is content_mask_mse_loss
