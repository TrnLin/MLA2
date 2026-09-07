from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from fashion.train.task3_micro_swin import (  # noqa: E402
    ARTIFACT_ROOT,
    MODEL_FAMILY,
    MicroSwinConfig,
    Task3MicroSwin,
    run_micro_swin_screen,
)


@pytest.mark.parametrize("target,classes", [("gender", 5), ("usage", 9)])
def test_micro_swin_is_a_small_pure_patch_transformer(target: str, classes: int) -> None:
    config = MicroSwinConfig(target=target)  # type: ignore[arg-type]
    model = Task3MicroSwin(config)

    output = model(torch.zeros(2, 3, 80, 60))
    parameters = sum(parameter.numel() for parameter in model.parameters())

    assert tuple(output.shape) == (2, classes)
    assert 2_000_000 <= parameters <= 3_000_000
    assert not any(isinstance(module, nn.Conv2d) for module in model.modules())
    assert config.scratch is True
    assert config.submission_eligible is True


def test_micro_swin_target_views_follow_the_clean_slate_eda() -> None:
    assert MicroSwinConfig(target="gender").input_view == "foreground_masked"
    assert MicroSwinConfig(target="usage").input_view == "full"
    assert MicroSwinConfig(target="usage").to_dict()["pretrained_weights"] is False
    assert MODEL_FAMILY == "scratch_pure_patch_micro_swin"
    assert ARTIFACT_ROOT.startswith("experiments/task3_clean_slate_")


def test_micro_swin_screen_rejects_partial_or_reordered_fold_sets(tmp_path) -> None:
    with pytest.raises(ValueError, match="folds 0 and 4"):
        run_micro_swin_screen(
            "gender",
            output_root=tmp_path,
            root=tmp_path,
            device_name="cpu",
            folds=(4, 0),
        )
