"""Frozen objectives for the Task 4 learned-model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F

VICREG_INVARIANCE_WEIGHT = 25.0
VICREG_VARIANCE_WEIGHT = 25.0
VICREG_COVARIANCE_WEIGHT = 1.0
VICREG_VARIANCE_EPSILON = 1e-4
FAMILY_TRIPLET_MARGIN = 0.2
R4_TRIPLET_WEIGHT = 1.0

__all__ = (
    "FAMILY_TRIPLET_MARGIN",
    "R4_TRIPLET_WEIGHT",
    "VICREG_COVARIANCE_WEIGHT",
    "VICREG_INVARIANCE_WEIGHT",
    "VICREG_VARIANCE_EPSILON",
    "VICREG_VARIANCE_WEIGHT",
    "R4LossBreakdown",
    "ReconstructionLossBreakdown",
    "TripletLossBreakdown",
    "VICRegLossBreakdown",
    "batch_hard_family_triplet_loss",
    "content_mask_mse_loss",
    "family_triplet_masks",
    "r4_loss",
    "vicreg_loss",
)


@dataclass(frozen=True, slots=True)
class VICRegLossBreakdown:
    """VICReg total and its unweighted named components."""

    total: torch.Tensor
    invariance: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor
    invariance_weight: float = VICREG_INVARIANCE_WEIGHT
    variance_weight: float = VICREG_VARIANCE_WEIGHT
    covariance_weight: float = VICREG_COVARIANCE_WEIGHT


@dataclass(frozen=True, slots=True)
class TripletLossBreakdown:
    """Batch-hard triplet result and the valid-anchor diagnostics."""

    total: torch.Tensor
    valid_anchor_mask: torch.Tensor
    hardest_positive: torch.Tensor
    hardest_negative: torch.Tensor
    per_anchor: torch.Tensor
    margin: float = FAMILY_TRIPLET_MARGIN


@dataclass(frozen=True, slots=True)
class R4LossBreakdown:
    """Combined R4 objective with reusable nested component values."""

    total: torch.Tensor
    vicreg: VICRegLossBreakdown
    triplet: TripletLossBreakdown
    triplet_weight: float = R4_TRIPLET_WEIGHT


@dataclass(frozen=True, slots=True)
class ReconstructionLossBreakdown:
    """Content-only reconstruction objective and denominator."""

    total: torch.Tensor
    included_values: int


def _validate_projection_pair(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[int, int]:
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("VICReg projections must be 2D")
    if first.shape != second.shape:
        raise ValueError("VICReg projection shapes must match")
    if first.shape[0] < 2:
        raise ValueError("VICReg batch size must be at least two")
    if first.shape[1] < 1:
        raise ValueError("VICReg projections must not be empty")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("VICReg projections must be finite")
    return first.shape


def _off_diagonal_sum_of_squares(covariance: torch.Tensor) -> torch.Tensor:
    dimensions = covariance.shape[0]
    if dimensions == 1:
        return covariance.new_zeros(())
    mask = ~torch.eye(dimensions, dtype=torch.bool, device=covariance.device)
    return covariance[mask].square().sum()


def vicreg_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    variance_epsilon: float = VICREG_VARIANCE_EPSILON,
) -> VICRegLossBreakdown:
    """Return VICReg with frozen 25/25/1 weights on unnormalized projections."""
    batch_size, dimensions = _validate_projection_pair(first, second)
    if variance_epsilon < 0:
        raise ValueError("VICReg variance epsilon must be non-negative")

    invariance = F.mse_loss(first, second)
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    first_std = torch.sqrt(first.var(dim=0, correction=1) + variance_epsilon)
    second_std = torch.sqrt(second.var(dim=0, correction=1) + variance_epsilon)
    variance = 0.5 * (
        F.relu(1.0 - first_std).mean() + F.relu(1.0 - second_std).mean()
    )

    first_covariance = first_centered.T @ first_centered / (batch_size - 1)
    second_covariance = second_centered.T @ second_centered / (batch_size - 1)
    covariance = (
        _off_diagonal_sum_of_squares(first_covariance)
        + _off_diagonal_sum_of_squares(second_covariance)
    ) / dimensions
    total = (
        VICREG_INVARIANCE_WEIGHT * invariance
        + VICREG_VARIANCE_WEIGHT * variance
        + VICREG_COVARIANCE_WEIGHT * covariance
    )
    return VICRegLossBreakdown(
        total=total,
        invariance=invariance,
        variance=variance,
        covariance=covariance,
    )


def _metadata_rows(
    *,
    product_ids: Sequence[object] | torch.Tensor,
    family_groups: Sequence[object],
    sha256: Sequence[object],
    duplicate_groups: Sequence[object],
) -> tuple[list[object], list[object], list[object], list[object]]:
    values = (
        product_ids.tolist() if isinstance(product_ids, torch.Tensor) else list(product_ids),
        list(family_groups),
        list(sha256),
        list(duplicate_groups),
    )
    lengths = {len(column) for column in values}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("triplet metadata columns must have the same non-zero length")
    return values


def family_triplet_masks(
    *,
    product_ids: Sequence[object] | torch.Tensor,
    family_groups: Sequence[object],
    sha256: Sequence[object],
    duplicate_groups: Sequence[object],
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return eligible positive and negative masks for every anchor."""
    ids, families, hashes, duplicates = _metadata_rows(
        product_ids=product_ids,
        family_groups=family_groups,
        sha256=sha256,
        duplicate_groups=duplicate_groups,
    )
    rows = len(ids)
    output_device = (
        product_ids.device if device is None and isinstance(product_ids, torch.Tensor) else device
    )
    positive = torch.zeros((rows, rows), dtype=torch.bool)
    negative = torch.zeros((rows, rows), dtype=torch.bool)
    for anchor in range(rows):
        for candidate in range(rows):
            different_id = ids[anchor] != ids[candidate]
            if not different_id:
                continue
            if families[anchor] != families[candidate]:
                negative[anchor, candidate] = True
            elif (
                hashes[anchor] != hashes[candidate]
                and duplicates[anchor] != duplicates[candidate]
            ):
                positive[anchor, candidate] = True
    if output_device is None:
        return positive, negative
    return positive.to(output_device), negative.to(output_device)


def batch_hard_family_triplet_loss(
    embeddings: torch.Tensor,
    *,
    product_ids: Sequence[object] | torch.Tensor,
    family_groups: Sequence[object],
    sha256: Sequence[object],
    duplicate_groups: Sequence[object],
) -> TripletLossBreakdown:
    """Apply the frozen family-aware Euclidean batch-hard triplet objective."""
    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        raise ValueError("triplet embeddings must be a non-empty 2D tensor")
    if not embeddings.is_floating_point() or not torch.isfinite(embeddings).all():
        raise ValueError("triplet embeddings must be finite floating-point values")
    positive, negative = family_triplet_masks(
        product_ids=product_ids,
        family_groups=family_groups,
        sha256=sha256,
        duplicate_groups=duplicate_groups,
        device=embeddings.device,
    )
    if positive.shape[0] != embeddings.shape[0]:
        raise ValueError("triplet metadata rows must match embedding rows")
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not valid.any():
        empty = embeddings.new_empty((0,))
        return TripletLossBreakdown(
            total=embeddings.sum() * 0.0,
            valid_anchor_mask=valid,
            hardest_positive=empty,
            hardest_negative=empty,
            per_anchor=empty,
        )

    distances = torch.cdist(embeddings, embeddings, p=2)
    valid_positive = positive[valid]
    valid_negative = negative[valid]
    valid_distances = distances[valid]
    hardest_positive = valid_distances.masked_fill(~valid_positive, -torch.inf).max(dim=1).values
    hardest_negative = valid_distances.masked_fill(valid_negative.logical_not(), torch.inf).min(
        dim=1
    ).values
    per_anchor = F.relu(hardest_positive - hardest_negative + FAMILY_TRIPLET_MARGIN)
    return TripletLossBreakdown(
        total=per_anchor.mean(),
        valid_anchor_mask=valid,
        hardest_positive=hardest_positive,
        hardest_negative=hardest_negative,
        per_anchor=per_anchor,
    )


def r4_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    triplet_embeddings: torch.Tensor,
    product_ids: Sequence[object] | torch.Tensor,
    family_groups: Sequence[object],
    sha256: Sequence[object],
    duplicate_groups: Sequence[object],
) -> R4LossBreakdown:
    """Return VICReg plus the frozen unit-weight family triplet term."""
    vicreg = vicreg_loss(first, second)
    triplet = batch_hard_family_triplet_loss(
        triplet_embeddings,
        product_ids=product_ids,
        family_groups=family_groups,
        sha256=sha256,
        duplicate_groups=duplicate_groups,
    )
    return R4LossBreakdown(
        total=vicreg.total + R4_TRIPLET_WEIGHT * triplet.total,
        vicreg=vicreg,
        triplet=triplet,
    )


def content_mask_mse_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    content_mask: torch.Tensor,
) -> ReconstructionLossBreakdown:
    """Return mean squared reconstruction error over content pixels only."""
    if reconstruction.ndim != 4 or reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target must have matching NCHW shapes")
    if content_mask.dtype is not torch.bool:
        raise ValueError("content mask must be boolean")
    if content_mask.ndim == 3:
        mask = content_mask.unsqueeze(1)
    elif content_mask.ndim == 4 and content_mask.shape[1] == 1:
        mask = content_mask
    else:
        raise ValueError("content mask must have shape (N,H,W) or (N,1,H,W)")
    expected = (reconstruction.shape[0], 1, *reconstruction.shape[2:])
    if mask.shape != expected:
        raise ValueError("content mask shape must match reconstruction pixels")
    if not mask.any():
        raise ValueError("content mask must include at least one pixel")

    expanded = mask.expand_as(reconstruction)
    squared_error = (reconstruction - target).square()
    included_values = int(expanded.sum().item())
    total = squared_error.masked_select(expanded).mean()
    return ReconstructionLossBreakdown(total=total, included_values=included_values)
