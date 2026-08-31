"""Loss functions with explicit weighting semantics for Task 3 experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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
