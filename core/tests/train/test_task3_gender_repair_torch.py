"""CPU parity when PyTorch is available; CUDA budget is checked by the notebook."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_cpu_probe_cannot_pass_gpu_memory_guard():
    from fashion.train.task3_gender_repair_preflight import profile_gpu

    report = profile_gpu(torch.randn(2, 3, 80, 60), torch.tensor([0, 1]))
    assert report["passed"] is False
    assert report["optimizer_steps"] == 0
    assert report["speed_cap"] is None


def test_lower_level_training_refuses_missing_prerequisite_before_fitting(tmp_path):
    from fashion.train.task3_baseline import run_task3_baseline_fold
    from fashion.train.task3_dataset_v2 import dataset_v2_spec

    spec = dataset_v2_spec("gender_translation_mild_darkening", [f"g2-{f}" for f in range(5)])
    with pytest.raises(ValueError, match="requires completed"):
        run_task3_baseline_fold("gender", 0, output_root=tmp_path, root=tmp_path, child_spec=spec)
    assert not list(tmp_path.rglob("runs.csv"))
