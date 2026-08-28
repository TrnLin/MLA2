from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from fashion.train.engine import TrainConfig
from fashion.train.metrics import validate_oof
from fashion.train.multitask import (
    masked_multitask_cross_entropy,
    train_masked_multitask_fold,
)


class TinyMultiTaskDataset(Dataset):
    def __init__(self) -> None:
        negative = torch.tensor([[-2.0, -1.0], [-1.5, -2.0], [-1.0, -1.5], [-2.0, -2.0]])
        positive = -negative
        self.images = torch.cat([negative, positive]).repeat((4, 1))
        self.targets = torch.tensor([0] * 4 + [1] * 4).repeat(4)
        self.auxiliary_targets = self.targets.clone()
        self.auxiliary_mask = torch.ones(32, dtype=torch.bool)
        self.auxiliary_mask[::5] = False
        self.ids = torch.arange(100, 132)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        auxiliary_target = self.auxiliary_targets[index]
        if not self.auxiliary_mask[index]:
            auxiliary_target = torch.tensor(-100)
        return {
            "image": self.images[index],
            "target": self.targets[index],
            "auxiliary_target": auxiliary_target,
            "auxiliary_mask": self.auxiliary_mask[index],
            "id": self.ids[index],
        }


class TinyMultiTaskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 4)
        self.season_head = nn.Linear(4, 2)
        self.article_type_head = nn.Linear(4, 2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = torch.tanh(self.encoder(images))
        return {
            "season_logits": self.season_head(embedding),
            "article_type_logits": self.article_type_head(embedding),
        }

    def predict_season_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)["season_logits"]


def _loaders() -> tuple[DataLoader, DataLoader]:
    dataset = TinyMultiTaskDataset()
    return (
        DataLoader(dataset, batch_size=4, shuffle=False),
        DataLoader(dataset, batch_size=8, shuffle=False),
    )


def test_masked_loss_ignores_missing_auxiliary_targets_and_gradients() -> None:
    season_logits = torch.tensor([[3.0, -1.0], [-1.0, 3.0]], requires_grad=True)
    auxiliary_logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]], requires_grad=True)
    season_targets = torch.tensor([0, 1])
    auxiliary_targets = torch.tensor([0, -100])
    mask = torch.tensor([True, False])

    terms = masked_multitask_cross_entropy(
        {
            "season_logits": season_logits,
            "article_type_logits": auxiliary_logits,
        },
        season_targets,
        auxiliary_targets,
        mask,
        auxiliary_weight=0.3,
    )
    expected = F.cross_entropy(season_logits, season_targets) + 0.3 * F.cross_entropy(
        auxiliary_logits[:1], auxiliary_targets[:1]
    )
    terms.total.backward()

    assert torch.allclose(terms.total, expected)
    assert terms.auxiliary_count == 1
    assert torch.count_nonzero(auxiliary_logits.grad[1]) == 0
    assert torch.count_nonzero(auxiliary_logits.grad[0]) > 0


def test_masked_loss_handles_batch_with_no_auxiliary_labels() -> None:
    season_logits = torch.randn(2, 2, requires_grad=True)
    auxiliary_logits = torch.randn(2, 3, requires_grad=True)
    terms = masked_multitask_cross_entropy(
        {
            "season_logits": season_logits,
            "article_type_logits": auxiliary_logits,
        },
        torch.tensor([0, 1]),
        torch.tensor([-100, -100]),
        torch.tensor([False, False]),
        auxiliary_weight=0.1,
    )

    terms.total.backward()

    assert terms.auxiliary_count == 0
    assert float(terms.auxiliary.detach()) == 0.0
    assert torch.count_nonzero(auxiliary_logits.grad) == 0
    assert torch.count_nonzero(season_logits.grad) > 0


def test_multitask_fold_writes_best_season_checkpoint_and_oof(tmp_path: Path) -> None:
    train_loader, validation_loader = _loaders()
    checkpoint = tmp_path / "multitask-fold.pt"
    config = TrainConfig(
        fold=0,
        seed=2753,
        epochs=12,
        learning_rate=0.05,
        weight_decay=0.0,
        batch_size=4,
        effective_batch_size=8,
        warmup_epochs=0,
        patience=4,
        device="cpu",
    )

    result = train_masked_multitask_fold(
        TinyMultiTaskModel(),
        train_loader,
        validation_loader,
        config=config,
        checkpoint_path=checkpoint,
        auxiliary_weight=0.1,
        labels=("negative", "positive"),
    )

    assert result.best_macro_f1 == 1.0
    assert result.metadata["selection_metric"] == "season_macro_f1"
    assert result.metadata["auxiliary_weight"] == 0.1
    assert result.history[0]["train_auxiliary_labeled_samples"] == 25
    assert "validation_season_loss" in result.history[0]
    assert "validation_auxiliary_loss" in result.history[0]
    assert checkpoint.is_file()
    assert result.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audit = validate_oof(
        result.to_oof_frame(),
        expected_ids=range(100, 132),
        labels=("negative", "positive"),
    )
    assert audit["row_count"] == 32


@pytest.mark.parametrize("weight", [0.0, -0.1, float("nan"), float("inf")])
def test_multitask_fold_rejects_invalid_auxiliary_weight(
    tmp_path: Path,
    weight: float,
) -> None:
    train_loader, validation_loader = _loaders()
    with pytest.raises(ValueError, match="finite positive"):
        train_masked_multitask_fold(
            TinyMultiTaskModel(),
            train_loader,
            validation_loader,
            config=TrainConfig(
                fold=0,
                seed=2753,
                epochs=1,
                batch_size=4,
                effective_batch_size=4,
                warmup_epochs=0,
                device="cpu",
            ),
            checkpoint_path=tmp_path / "invalid.pt",
            auxiliary_weight=weight,
            labels=("negative", "positive"),
        )
