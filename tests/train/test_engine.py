from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fashion.train.engine import TrainConfig, train_fold
from fashion.train.metrics import validate_oof


class TinyDataset(Dataset):
    def __init__(self) -> None:
        negative = torch.tensor([[-2.0, -1.0], [-1.5, -2.0], [-1.0, -1.5], [-2.0, -2.0]])
        positive = -negative
        self.images = torch.cat([negative, positive]).repeat((4, 1))
        self.targets = torch.tensor([0] * 4 + [1] * 4).repeat(4)
        self.ids = torch.arange(100, 132)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[index],
            "target": self.targets[index],
            "id": self.ids[index],
        }


def _loaders() -> tuple[DataLoader, DataLoader]:
    dataset = TinyDataset()
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    validation_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    return train_loader, validation_loader


def test_train_fold_writes_best_checkpoint_and_oof(tmp_path: Path) -> None:
    train_loader, validation_loader = _loaders()
    model = nn.Linear(2, 2)
    checkpoint = tmp_path / "fold-0.pt"
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

    result = train_fold(
        model,
        train_loader,
        validation_loader,
        config=config,
        checkpoint_path=checkpoint,
        labels=("negative", "positive"),
    )

    assert result.best_macro_f1 == 1.0
    assert result.best_epoch >= 1
    assert result.epochs_completed <= config.epochs
    assert result.parameter_count == 6
    assert result.device == "cpu"
    assert not result.metadata["amp_enabled"]
    assert checkpoint.is_file()
    assert result.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    oof = result.to_oof_frame()
    audit = validate_oof(
        oof,
        expected_ids=range(100, 132),
        labels=("negative", "positive"),
    )
    assert audit["row_count"] == 32


def test_train_fold_stops_after_unchanged_validation_metric(tmp_path: Path) -> None:
    train_loader, validation_loader = _loaders()
    model = nn.Linear(2, 2)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    config = TrainConfig(
        fold=1,
        seed=2026,
        epochs=10,
        learning_rate=1e-12,
        weight_decay=0.0,
        batch_size=4,
        effective_batch_size=4,
        warmup_epochs=0,
        patience=2,
        device="cpu",
    )

    result = train_fold(
        model,
        train_loader,
        validation_loader,
        config=config,
        checkpoint_path=tmp_path / "early-stop.pt",
        labels=("negative", "positive"),
    )

    assert result.stopped_early
    assert result.epochs_completed == 3
    assert result.best_epoch == 1


def test_train_config_rejects_ambiguous_effective_batch_size() -> None:
    config = TrainConfig(fold=0, seed=2753, batch_size=24, effective_batch_size=128)
    with pytest.raises(ValueError, match="divisible"):
        config.validate()


def test_train_fold_rejects_loader_batch_size_drift(tmp_path: Path) -> None:
    train_loader, validation_loader = _loaders()
    with pytest.raises(ValueError, match="batch_size must match"):
        train_fold(
            nn.Linear(2, 2),
            train_loader,
            validation_loader,
            config=TrainConfig(
                fold=0,
                seed=2753,
                epochs=1,
                batch_size=8,
                effective_batch_size=8,
                warmup_epochs=0,
                device="cpu",
            ),
            checkpoint_path=tmp_path / "wrong-batch.pt",
            labels=("negative", "positive"),
        )


def test_train_fold_rejects_validation_batches_without_ids(tmp_path: Path) -> None:
    dataset = TinyDataset()
    pairs = [(row["image"], row["target"]) for row in dataset]
    loader = DataLoader(pairs, batch_size=4)
    with pytest.raises(ValueError, match="stable sample IDs"):
        train_fold(
            nn.Linear(2, 2),
            loader,
            loader,
            config=TrainConfig(
                fold=0,
                seed=2753,
                epochs=1,
                batch_size=4,
                effective_batch_size=4,
                warmup_epochs=0,
                device="cpu",
            ),
            checkpoint_path=tmp_path / "missing-ids.pt",
            labels=("negative", "positive"),
        )
