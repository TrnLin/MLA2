from __future__ import annotations

import random

import numpy as np
import torch

from fashion.train.reproducibility import (
    capture_git_state,
    capture_runtime,
    make_torch_generator,
    seed_everything,
)


def _random_values() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(1).item())


def test_seed_everything_repeats_random_streams() -> None:
    seed_everything(2753)
    first = _random_values()
    seed_everything(2753)
    second = _random_values()

    assert first == second
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_seed_and_generator_reject_negative_values() -> None:
    for operation in (seed_everything, make_torch_generator):
        try:
            operation(-1)
        except ValueError as error:
            assert "non-negative" in str(error)
        else:
            raise AssertionError("negative seeds must be rejected")


def test_torch_generator_is_repeatable() -> None:
    first = torch.rand(4, generator=make_torch_generator(2753))
    second = torch.rand(4, generator=make_torch_generator(2753))
    assert torch.equal(first, second)


def test_runtime_and_git_provenance_are_serializable() -> None:
    runtime = capture_runtime()
    git_state = capture_git_state()

    assert runtime["python"]
    assert runtime["packages"]["scipy"]
    assert runtime["packages"]["torch"]
    assert runtime["logical_cpu_count"] >= 1
    assert isinstance(runtime["cuda_available"], bool)
    assert git_state["commit"]
    assert len(git_state["commit"]) == 40
    assert isinstance(git_state["dirty"], bool)
