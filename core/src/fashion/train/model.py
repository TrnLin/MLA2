"""Exact scratch image models used by the Task 3 experiment chain."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from fashion.train.config import (
    COMPACT_BLUR_CNN_WIDTHS,
    NARROW_GEM3_FAMILY,
    TINYCONVNEXT18_DEPTHS,
    TINYCONVNEXT18_WIDTHS,
    TINYHRNET20_WIDTHS,
    TINYRESNET18_PM_WIDTHS,
    Task3BaselineConfig,
    baseline_parameter_count,
    compact_blur_cnn_parameter_count,
    gender_audience_aux_parameter_count,
    narrow_gem3_parameter_count,
    tinyconvnext18_parameter_count,
    tinyhrnet20_parameter_count,
    tinyresnet18_pm_parameter_count,
)


class Task3BaselineCNN(nn.Module):
    """Four-block native-resolution CNN with three spatial reductions."""

    def __init__(self, config: Task3BaselineConfig, *, classifier_dropout: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= classifier_dropout < 1.0:
            raise ValueError("classifier dropout must be in [0, 1)")
        channel_pairs = zip((3, *config.channels[:-1]), config.channels, strict=True)
        blocks: list[nn.Module] = []
        for index, (input_channels, output_channels) in enumerate(channel_pairs):
            blocks.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if index < 3:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_dropout = (
            nn.Dropout(p=classifier_dropout) if classifier_dropout > 0.0 else nn.Identity()
        )
        self.classifier = nn.Linear(config.channels[-1], config.num_classes)
        self._initialise()

        actual = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        expected = (
            narrow_gem3_parameter_count()
            if config.model_family == NARROW_GEM3_FAMILY
            else baseline_parameter_count(config.target)
        )
        if actual != expected:
            raise RuntimeError(
                f"baseline parameter contract failed: expected {expected}, found {actual}"
            )

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = self.pool(features).flatten(1)
        return self.classifier(self.classifier_dropout(pooled))


class FixedGeMPool2d(nn.Module):
    """Fixed generalized-mean pooling with no trainable parameters."""

    def __init__(self, *, power: float = 3.0, epsilon: float = 1e-6) -> None:
        super().__init__()
        if power <= 0.0:
            raise ValueError("GeM power must be positive")
        if epsilon <= 0.0:
            raise ValueError("GeM epsilon must be positive")
        self.power = float(power)
        self.epsilon = float(epsilon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("GeM expects an NCHW feature tensor")
        return (
            inputs.clamp_min(self.epsilon)
            .pow(self.power)
            .mean(dim=(-2, -1), keepdim=True)
            .pow(1.0 / self.power)
        )


class Task3GeM3CNN(Task3BaselineCNN):
    """The accepted SmallCNN with only average pooling changed to fixed GeM p=3."""

    def __init__(self, config: Task3BaselineConfig, *, classifier_dropout: float = 0.0) -> None:
        super().__init__(config, classifier_dropout=classifier_dropout)
        self.pool = FixedGeMPool2d(power=3.0)


class Task3GeM3AudienceCNN(Task3GeM3CNN):
    """E6 GeM model with one training-only three-way audience helper head."""

    def __init__(self, config: Task3BaselineConfig) -> None:
        if config.target != "gender":
            raise ValueError("the audience helper is defined only for the Gender target")
        super().__init__(config)
        self.audience_classifier = nn.Linear(config.channels[-1], 3)
        nn.init.kaiming_uniform_(self.audience_classifier.weight, a=math.sqrt(5))
        nn.init.zeros_(self.audience_classifier.bias)

        actual = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        expected = gender_audience_aux_parameter_count()
        if actual != expected:
            raise RuntimeError(
                f"E10 parameter contract failed: expected {expected}, found {actual}"
            )

    def forward_with_auxiliary(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the five-way logits and the training-only audience logits."""
        features = self.features(images)
        pooled = self.pool(features).flatten(1)
        pooled = self.classifier_dropout(pooled)
        return self.classifier(pooled), self.audience_classifier(pooled)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return only the five-way logits used by validation and inference."""
        primary_logits, _ = self.forward_with_auxiliary(images)
        return primary_logits


class _Task3ResidualBlock(nn.Module):
    """Two 3x3 convolutions with an identity or projected residual path."""

    def __init__(self, input_channels: int, output_channels: int, *, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.shortcut: nn.Module
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        features = self.relu(self.bn1(self.conv1(inputs)))
        features = self.bn2(self.conv2(features))
        return self.relu(features + residual)


class Task3TinyResNet18PM(nn.Module):
    """Parameter-matched residual CNN with no early pooling or pretrained weights."""

    def __init__(self, config: Task3BaselineConfig) -> None:
        super().__init__()
        widths = TINYRESNET18_PM_WIDTHS
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        input_channels = widths[0]
        for stage, output_channels in enumerate(widths):
            for block in range(2):
                stride = 2 if stage > 0 and block == 0 else 1
                stages.append(
                    _Task3ResidualBlock(
                        input_channels,
                        output_channels,
                        stride=stride,
                    )
                )
                input_channels = output_channels
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(widths[-1], config.num_classes)
        self._initialise()

        actual = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        expected = tinyresnet18_pm_parameter_count(config.target)
        if actual != expected:
            raise RuntimeError(
                f"TinyResNet parameter contract failed: expected {expected}, found {actual}"
            )

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self.stem(images))
        return self.classifier(self.pool(features).flatten(1))


class _FixedBlurPool(nn.Module):
    """Channel-wise fixed binomial filtering followed by stride-two sampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        self.channels = channels
        self.register_buffer(
            "kernel",
            (kernel / 16.0).reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] != self.channels:
            raise ValueError(f"BlurPool expected {self.channels} channels, found {inputs.shape[1]}")
        return F.conv2d(
            inputs,
            self.kernel,
            stride=2,
            padding=1,
            groups=self.channels,
        )


class Task3CompactBlurCNN(nn.Module):
    """Low-capacity scratch CNN with fixed anti-aliased spatial reductions."""

    def __init__(self, config: Task3BaselineConfig) -> None:
        super().__init__()
        first, second, third, output = COMPACT_BLUR_CNN_WIDTHS
        self.features = nn.Sequential(
            nn.Conv2d(3, first, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(first),
            nn.ReLU(inplace=True),
            _FixedBlurPool(first),
            nn.Conv2d(first, second, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),
            _FixedBlurPool(second),
            nn.Conv2d(second, third, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(third),
            nn.ReLU(inplace=True),
            _FixedBlurPool(third),
            nn.Conv2d(
                third,
                third,
                kernel_size=3,
                padding=1,
                groups=third,
                bias=False,
            ),
            nn.BatchNorm2d(third),
            nn.ReLU(inplace=True),
            nn.Conv2d(third, output, kernel_size=1, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output, config.num_classes)
        self._initialise()

        actual = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        expected = compact_blur_cnn_parameter_count(config.target)
        if actual != expected:
            raise RuntimeError(
                f"CompactBlurCNN parameter contract failed: expected {expected}, found {actual}"
            )

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(self.pool(features).flatten(1))


class _Task3HRNetFuse(nn.Module):
    """Fuse every HRNet branch into every other resolution."""

    def __init__(self, widths: tuple[int, ...]) -> None:
        super().__init__()
        self.widths = widths
        transforms: dict[str, nn.Module] = {}
        for source in range(len(widths)):
            for target in range(len(widths)):
                if source == target:
                    continue
                key = f"{source}_to_{target}"
                if source > target:
                    transforms[key] = nn.Sequential(
                        nn.Conv2d(widths[source], widths[target], kernel_size=1, bias=False),
                        nn.BatchNorm2d(widths[target]),
                    )
                    continue
                reductions: list[nn.Module] = []
                input_channels = widths[source]
                for step in range(source + 1, target + 1):
                    output_channels = widths[step]
                    reductions.extend(
                        [
                            nn.Conv2d(
                                input_channels,
                                output_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(output_channels),
                        ]
                    )
                    if step != target:
                        reductions.append(nn.ReLU(inplace=True))
                    input_channels = output_channels
                transforms[key] = nn.Sequential(*reductions)
        self.transforms = nn.ModuleDict(transforms)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(branches) != len(self.widths):
            raise ValueError("HRNet fusion received the wrong number of branches")
        outputs: list[torch.Tensor] = []
        for target, target_features in enumerate(branches):
            fused = target_features
            for source, source_features in enumerate(branches):
                if source == target:
                    continue
                transformed = self.transforms[f"{source}_to_{target}"](source_features)
                if source > target:
                    transformed = F.interpolate(
                        transformed,
                        size=target_features.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if transformed.shape[-2:] != target_features.shape[-2:]:
                    raise RuntimeError("HRNet fusion produced an unexpected spatial shape")
                fused = fused + transformed
            outputs.append(self.relu(fused))
        return outputs


class _Task3HRNetExchangeUnit(nn.Module):
    """Apply one residual block per branch, then exchange resolutions."""

    def __init__(self, widths: tuple[int, ...]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _Task3ResidualBlock(channels, channels, stride=1) for channels in widths
        )
        self.fuse = _Task3HRNetFuse(widths)

    def forward(self, branches: list[torch.Tensor]) -> list[torch.Tensor]:
        refined = [block(features) for block, features in zip(self.blocks, branches, strict=True)]
        return self.fuse(refined)


class Task3TinyHRNet20(nn.Module):
    """Scratch three-resolution network for the Gender E7 hypothesis."""

    def __init__(self, config: Task3BaselineConfig) -> None:
        super().__init__()
        high, middle, low = TINYHRNET20_WIDTHS
        self.stem = nn.Sequential(
            nn.Conv2d(3, high, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(high),
            nn.ReLU(inplace=True),
            nn.Conv2d(high, high, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(high),
            nn.ReLU(inplace=True),
        )
        self.high_resolution_blocks = nn.Sequential(
            _Task3ResidualBlock(high, high, stride=1),
            _Task3ResidualBlock(high, high, stride=1),
        )
        self.add_middle_branch = nn.Sequential(
            nn.Conv2d(high, middle, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(middle),
            nn.ReLU(inplace=True),
        )
        self.two_branch_stage = nn.ModuleList(
            [_Task3HRNetExchangeUnit((high, middle)) for _ in range(2)]
        )
        self.add_low_branch = nn.Sequential(
            nn.Conv2d(middle, low, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(low),
            nn.ReLU(inplace=True),
        )
        self.three_branch_stage = _Task3HRNetExchangeUnit((high, middle, low))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(sum(TINYHRNET20_WIDTHS), config.num_classes)
        self._initialise()

        actual = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        expected = tinyhrnet20_parameter_count(config.target)
        if actual != expected:
            raise RuntimeError(
                f"TinyHRNet parameter contract failed: expected {expected}, found {actual}"
            )

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def feature_branches(self, images: torch.Tensor) -> list[torch.Tensor]:
        high = self.high_resolution_blocks(self.stem(images))
        branches = [high, self.add_middle_branch(high)]
        for exchange in self.two_branch_stage:
            branches = exchange(branches)
        branches.append(self.add_low_branch(branches[1]))
        return self.three_branch_stage(branches)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        branches = self.feature_branches(images)
        pooled = torch.cat([self.pool(features).flatten(1) for features in branches], dim=1)
        return self.classifier(pooled)


class _Task3LayerNorm2d(nn.Module):
    """Apply LayerNorm over channels while accepting NCHW tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class _Task3ConvNeXtBlock(nn.Module):
    """One ConvNeXt block with fixed 7x7 depthwise and 4x channel expansion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.contract = nn.Linear(4 * channels, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.depthwise(inputs).permute(0, 2, 3, 1)
        features = self.norm(features)
        features = self.contract(self.activation(self.expand(features)))
        features = features * self.layer_scale
        return inputs + features.permute(0, 3, 1, 2)


class Task3TinyConvNeXt18(nn.Module):
    """Scratch TinyConvNeXt for the Usage E7 architecture hypothesis."""

    def __init__(self, config: Task3BaselineConfig) -> None:
        super().__init__()
        widths = TINYCONVNEXT18_WIDTHS
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1),
            _Task3LayerNorm2d(widths[0]),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*[_Task3ConvNeXtBlock(channels) for _ in range(depth)])
                for channels, depth in zip(widths, TINYCONVNEXT18_DEPTHS, strict=True)
            ]
        )
        self.transitions = nn.ModuleList(
            [
                nn.Sequential(
                    _Task3LayerNorm2d(input_channels),
                    nn.Conv2d(input_channels, output_channels, kernel_size=2, stride=2),
                )
                for input_channels, output_channels in zip(widths[:-1], widths[1:], strict=True)
            ]
        )
        self.head_norm = nn.LayerNorm(widths[-1])
        self.classifier = nn.Linear(widths[-1], config.num_classes)
        self._initialise()

        actual = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        expected = tinyconvnext18_parameter_count(config.target)
        if actual != expected:
            raise RuntimeError(
                f"TinyConvNeXt parameter contract failed: expected {expected}, found {actual}"
            )

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        for stage, blocks in enumerate(self.stages):
            features = blocks(features)
            if stage < len(self.transitions):
                features = self.transitions[stage](features)
        pooled = features.mean(dim=(-2, -1))
        return self.classifier(self.head_norm(pooled))
