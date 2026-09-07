"""Loss functions with explicit weighting semantics for Task 3 experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GenderAudienceAuxiliaryCrossEntropy(nn.Module):
    """Combine the five-way Gender loss with a derived three-way audience loss."""

    def __init__(
        self,
        *,
        primary_weight: float = 0.5,
        auxiliary_weight: float = 0.5 * 1.6094379124341003 / 1.0986122886681098,
    ) -> None:
        super().__init__()
        if primary_weight <= 0.0 or auxiliary_weight <= 0.0:
            raise ValueError("auxiliary loss weights must be positive")
        self.primary_weight = float(primary_weight)
        self.auxiliary_weight = float(auxiliary_weight)
        self.register_buffer(
            "audience_index",
            torch.tensor([0, 1, 0, 2, 1], dtype=torch.long),
        )

    def forward(
        self,
        primary_logits: torch.Tensor,
        audience_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if primary_logits.ndim != 2 or primary_logits.shape[1] != 5:
            raise ValueError("the Gender primary head must have five logits")
        if audience_logits.ndim != 2 or audience_logits.shape[1] != 3:
            raise ValueError("the audience helper head must have three logits")
        if target.ndim != 1 or len(target) != len(primary_logits):
            raise ValueError("targets must contain one class index per logit row")
        if len(audience_logits) != len(target):
            raise ValueError("primary and audience logits must contain the same rows")
        if torch.any(target < 0) or torch.any(target >= len(self.audience_index)):
            raise ValueError("Gender targets must use the fixed five-class index order")
        audience_index = self.audience_index.to(device=target.device)
        audience_target = audience_index[target]
        primary_loss = F.cross_entropy(primary_logits, target)
        audience_loss = F.cross_entropy(audience_logits, audience_target)
        return self.primary_weight * primary_loss + self.auxiliary_weight * audience_loss


class SampleWeightedCrossEntropy(nn.Module):
    """Cross-entropy normalised by combined class and per-example weights."""

    def __init__(self, class_weights: torch.Tensor) -> None:
        super().__init__()
        if class_weights.ndim != 1 or len(class_weights) < 2:
            raise ValueError("class weights must be a one-dimensional multi-class tensor")
        if torch.any(class_weights < 0):
            raise ValueError("class weights cannot be negative")
        if not torch.any(class_weights > 0):
            raise ValueError("at least one class weight must be positive")
        self.register_buffer("class_weights", class_weights.detach().clone())

    def _combined_weights(self, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
        if sample_weight.ndim != 1 or len(sample_weight) != len(target):
            raise ValueError("sample weights must contain one value per target")
        if not torch.isfinite(sample_weight).all() or torch.any(sample_weight <= 0):
            raise ValueError("sample weights must be finite and strictly positive")
        return self.class_weights[target] * sample_weight

    def loss_denominator(self, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
        """Return the combined example-weight sum used by this batch."""
        return self._combined_weights(target, sample_weight).sum()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != len(self.class_weights):
            raise ValueError("logits and class-weight dimensions disagree")
        if target.ndim != 1 or len(target) != len(logits):
            raise ValueError("targets must contain one class index per logit row")
        combined = self._combined_weights(target, sample_weight)
        denominator = combined.sum()
        if denominator <= 0:
            raise ValueError("a batch has zero total example weight")
        per_sample = F.cross_entropy(logits, target, reduction="none")
        return (per_sample * combined).sum() / denominator


class WeightedLabelSmoothedCrossEntropy(nn.Module):
    """Smooth targets first, then apply the original true-class example weight."""

    def __init__(self, class_weights: torch.Tensor, *, epsilon: float) -> None:
        super().__init__()
        if class_weights.ndim != 1 or len(class_weights) < 2:
            raise ValueError("class weights must be a one-dimensional multi-class tensor")
        if not 0.0 < epsilon < 1.0:
            raise ValueError("label smoothing epsilon must be between zero and one")
        if torch.any(class_weights < 0):
            raise ValueError("class weights cannot be negative")
        if not torch.any(class_weights > 0):
            raise ValueError("at least one class weight must be positive")
        self.register_buffer("class_weights", class_weights.detach().clone())
        self.epsilon = float(epsilon)

    def loss_denominator(self, target: torch.Tensor) -> torch.Tensor:
        """Return the true-class example-weight sum used by this batch."""
        return self.class_weights[target].sum()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != len(self.class_weights):
            raise ValueError("logits and class-weight dimensions disagree")
        if target.ndim != 1 or len(target) != len(logits):
            raise ValueError("targets must contain one class index per logit row")
        classes = logits.shape[1]
        off_target = self.epsilon / (classes - 1)
        soft_targets = torch.full_like(logits, off_target)
        soft_targets.scatter_(1, target.unsqueeze(1), 1.0 - self.epsilon)
        per_sample = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1)
        sample_weights = self.class_weights[target]
        denominator = sample_weights.sum()
        if denominator <= 0:
            raise ValueError("a batch has zero total example weight")
        return (per_sample * sample_weights).sum() / denominator


class WeightedFocalCrossEntropy(nn.Module):
    """True-class weighted focal cross-entropy with an explicit denominator."""

    def __init__(self, class_weights: torch.Tensor, *, gamma: float) -> None:
        super().__init__()
        if class_weights.ndim != 1 or len(class_weights) < 2:
            raise ValueError("class weights must be a one-dimensional multi-class tensor")
        if gamma < 0.0:
            raise ValueError("focal gamma cannot be negative")
        if torch.any(class_weights < 0):
            raise ValueError("class weights cannot be negative")
        if not torch.any(class_weights > 0):
            raise ValueError("at least one class weight must be positive")
        self.register_buffer("class_weights", class_weights.detach().clone())
        self.gamma = float(gamma)

    def loss_denominator(self, target: torch.Tensor) -> torch.Tensor:
        """Return the true-class example-weight sum used by this batch."""
        return self.class_weights[target].sum()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != len(self.class_weights):
            raise ValueError("logits and class-weight dimensions disagree")
        if target.ndim != 1 or len(target) != len(logits):
            raise ValueError("targets must contain one class index per logit row")
        log_probabilities = F.log_softmax(logits, dim=1)
        true_log_probability = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
        true_probability = true_log_probability.exp()
        sample_weights = self.class_weights[target]
        per_sample = -(1.0 - true_probability).pow(self.gamma) * true_log_probability
        denominator = sample_weights.sum()
        if denominator <= 0:
            raise ValueError("a batch has zero total example weight")
        return (per_sample * sample_weights).sum() / denominator
