"""Frozen configuration for the Task 3 primary learnable baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

Task3Target = Literal["gender", "usage"]

TARGET_CLASS_COUNTS: dict[Task3Target, int] = {"gender": 5, "usage": 9}
TINYRESNET18_PM_WIDTHS = (12, 24, 48, 96)


@dataclass(frozen=True)
class Task3BaselineConfig:
    """One-factor-frozen configuration shared by every baseline fold."""

    target: Task3Target
    image_height: int = 80
    image_width: int = 60
    channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    minimum_learning_rate: float = 0.00001
    seed: int = 2753
    num_workers: int = 2
    mixed_precision: bool = False
    early_stopping: bool = False
    augmentation: str = "none"
    loss_name: str = "cross_entropy"
    optimizer_name: str = "AdamW"
    scheduler_name: str = "CosineAnnealingLR"
    checkpoint_rule: str = "final_epoch"
    model_family: str = "task3_small_cnn"
    scratch: bool = True
    submission_eligible: bool = True

    def __post_init__(self) -> None:
        if self.target not in TARGET_CLASS_COUNTS:
            raise ValueError(f"unsupported Task 3 target: {self.target}")
        if (self.image_height, self.image_width) != (80, 60):
            raise ValueError("the primary baseline input must stay at 80x60")
        if self.channels != (32, 64, 128, 256):
            raise ValueError("the primary baseline channels must stay at 32,64,128,256")
        for field_name in ("batch_size", "epochs", "num_workers"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.batch_size == 0 or self.epochs == 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0 or self.minimum_learning_rate < 0:
            raise ValueError("learning rates must be non-negative and start above zero")
        if self.minimum_learning_rate >= self.learning_rate:
            raise ValueError("minimum learning rate must be below the starting rate")
        if self.augmentation != "none":
            raise ValueError("augmentation is disabled for the primary baseline")
        if self.loss_name != "cross_entropy":
            raise ValueError("the primary baseline uses ordinary cross-entropy")
        if not self.scratch or not self.submission_eligible:
            raise ValueError("the primary baseline must be scratch-trained and eligible")

    @property
    def num_classes(self) -> int:
        return TARGET_CLASS_COUNTS[self.target]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["channels"] = list(self.channels)
        payload["num_classes"] = self.num_classes
        return payload


def config_digest(config: Task3BaselineConfig, *, length: int = 12) -> str:
    """Return a stable short digest for one frozen baseline configuration."""
    if length < 8:
        raise ValueError("config digest length must be at least 8")
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def baseline_parameter_count(target: Task3Target) -> int:
    """Calculate trainable parameters from the declared architecture, without PyTorch."""
    if target not in TARGET_CLASS_COUNTS:
        raise ValueError(f"unsupported Task 3 target: {target}")
    channels = (3, 32, 64, 128, 256)
    convolutions = sum(
        input_channels * output_channels * 3 * 3
        for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True)
    )
    batch_norm = sum(2 * output_channels for output_channels in channels[1:])
    classes = TARGET_CLASS_COUNTS[target]
    output_head = 256 * classes + classes
    return convolutions + batch_norm + output_head


def tinyresnet18_pm_parameter_count(target: Task3Target) -> int:
    """Calculate the parameter-matched TinyResNet contract without PyTorch."""
    if target not in TARGET_CLASS_COUNTS:
        raise ValueError(f"unsupported Task 3 target: {target}")
    widths = TINYRESNET18_PM_WIDTHS
    parameters = 3 * widths[0] * 3 * 3 + 2 * widths[0]
    input_channels = widths[0]
    for stage, output_channels in enumerate(widths):
        for block in range(2):
            stride = 2 if stage > 0 and block == 0 else 1
            parameters += input_channels * output_channels * 3 * 3
            parameters += 2 * output_channels
            parameters += output_channels * output_channels * 3 * 3
            parameters += 2 * output_channels
            if stride != 1 or input_channels != output_channels:
                parameters += input_channels * output_channels
                parameters += 2 * output_channels
            input_channels = output_channels
    classes = TARGET_CLASS_COUNTS[target]
    return parameters + widths[-1] * classes + classes


def tinyresnet18_pm_macs(target: Task3Target) -> int:
    """Return convolution and classifier MACs for the fixed 80x60 input."""
    if target not in TARGET_CLASS_COUNTS:
        raise ValueError(f"unsupported Task 3 target: {target}")
    widths = TINYRESNET18_PM_WIDTHS
    height, width = 80, 60
    macs = height * width * 3 * widths[0] * 3 * 3
    input_channels = widths[0]
    for stage, output_channels in enumerate(widths):
        for block in range(2):
            stride = 2 if stage > 0 and block == 0 else 1
            if stride == 2:
                height = (height + 1) // 2
                width = (width + 1) // 2
            macs += height * width * input_channels * output_channels * 3 * 3
            macs += height * width * output_channels * output_channels * 3 * 3
            if stride != 1 or input_channels != output_channels:
                macs += height * width * input_channels * output_channels
            input_channels = output_channels
    return macs + widths[-1] * TARGET_CLASS_COUNTS[target]
