from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.axes import Axes

from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import (
    _plot_g2_augmentation_ablation,
    build_g2_augmentation_evidence,
)
from fashion.task2.experiments import load_experiment_config
from fashion.train.artifacts import canonical_sha256


def _write_config(
    root: Path,
    *,
    variant: str,
    learning_rate: float = 3e-4,
) -> Path:
    experiment_id = {
        "A0": "g1-c2-resnet18",
        "A1": "g2-a1-c2-resnet18",
    }[variant]
    stage = {
        "A0": "g1_family_screen",
        "A1": "g2_augmentation_ablation",
    }[variant]
    path = root / "configs/task2" / f"{variant.lower()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "experiment_id": experiment_id,
                "method": "deep",
                "model_family": "resnet18_small_stem",
                "stage": stage,
                "folds": [0, 1, 2, 3, 4],
                "seeds": [2753],
                "loss_id": "cross_entropy",
                "data": {
                    "image_size": [80, 60],
                    "augmentation": variant.lower(),
                    "batch_size": 32,
                    "validation_batch_size": 128,
                    "num_workers": 4,
                    "pin_memory": True,
                },
                "optimisation": {
                    "epochs": 8,
                    "learning_rate": learning_rate,
                    "weight_decay": 1e-4,
                    "effective_batch_size": 128,
                    "gradient_clip_norm": 1.0,
                    "warmup_epochs": 1.0,
                    "patience": 8,
                    "min_delta": 1e-4,
                    "use_amp": True,
                    "device": "auto",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _write_experiment_evidence(
    root: Path,
    *,
    variant: str,
    config_path: Path,
    pooled_macro_f1: float,
) -> Path:
    experiment_id = {
        "A0": "g1-c2-resnet18",
        "A1": "g2-a1-c2-resnet18",
    }[variant]
    stage = {
        "A0": "g1_family_screen",
        "A1": "g2_augmentation_ablation",
    }[variant]
    evidence = root / "results/evidence/task2" / variant.lower()
    evidence.mkdir(parents=True, exist_ok=True)
    run_ids = [f"{experiment_id}-f{fold}" for fold in range(5)]
    fold_scores = [pooled_macro_f1 - 0.002 + fold * 0.001 for fold in range(5)]
    config_sha256 = canonical_sha256(load_experiment_config(config_path).to_dict())
    pooled_path = evidence / "pooled_metrics.json"
    fold_summary_path = evidence / "fold_summary.csv"
    fold_metrics_path = evidence / "fold_metrics.csv"
    registry_path = evidence / "registry_snapshot.csv"
    class_offsets = {
        "Fall": -0.02,
        "Spring": 0.02,
        "Summer": 0.01,
        "Winter": -0.01,
    }
    pooled_path.write_text(
        json.dumps(
            {
                "macro_f1": pooled_macro_f1,
                "per_class": {
                    label: {
                        "precision": pooled_macro_f1 + offset,
                        "recall": pooled_macro_f1 + offset,
                        "f1": pooled_macro_f1 + offset,
                        "support": support,
                    }
                    for (label, support), offset in zip(
                        (
                            ("Fall", 6),
                            ("Spring", 2),
                            ("Summer", 8),
                            ("Winter", 4),
                        ),
                        class_offsets.values(),
                        strict=True,
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "metric": "macro_f1",
                "fold_mean": sum(fold_scores) / len(fold_scores),
                "fold_sd": 0.002,
            }
        ]
    ).to_csv(fold_summary_path, index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "run_id": run_ids[fold],
                "fold": fold,
                "seed": 2753,
                "source": "cache",
                "macro_f1": fold_scores[fold],
            }
            for fold in range(5)
        ]
    ).to_csv(fold_metrics_path, index=False)
    pd.DataFrame(
        [
            {
                "run_id": run_ids[fold],
                "stage": stage,
                "experiment_id": experiment_id,
                "model_family": "resnet18_small_stem",
                "benchmark_only": False,
                "final_eligible": True,
                "scratch": True,
                "fold": fold,
                "seed": 2753,
                "git_dirty": False,
                "config_sha256": config_sha256,
                "split_sha256": "a" * 64,
                "label_map_sha256": "b" * 64,
                "implementation_sha256": "c" * 64,
                "transform_id": f"{variant.lower()}-transform",
                "loss_id": "cross_entropy",
                "primary_metric_value": fold_scores[fold],
                "runtime_seconds": (60.0 if variant == "A0" else 65.0) + fold,
                "peak_vram_mb": 600.0,
                "best_epoch": 7,
                "parameter_count": 11_170_884,
                "status": "completed",
            }
            for fold in range(5)
        ]
    ).to_csv(registry_path, index=False)
    artifacts = {
        "pooled_metrics": pooled_path,
        "fold_summary": fold_summary_path,
        "fold_metrics": fold_metrics_path,
        "registry_snapshot": registry_path,
    }
    manifest = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "seed": 2753,
        "folds": list(range(5)),
        "run_ids": run_ids,
        "coverage": {
            "row_count": 20,
            "unique_id_count": 20,
            "expected_row_count": 20,
            "id_set_sha256": "d" * 64,
            "labels": ["Fall", "Spring", "Summer", "Winter"],
            "protected_id_count": 0,
        },
        "pooled_macro_f1": pooled_macro_f1,
        "artifacts": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": compute_sha256(path),
            }
            for name, path in artifacts.items()
        },
    }
    manifest_path = evidence / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _inputs(
    root: Path,
    *,
    a0_score: float = 0.710,
    a1_score: float = 0.714,
    a1_learning_rate: float = 3e-4,
) -> dict[str, Path]:
    a0_config = _write_config(root, variant="A0")
    a1_config = _write_config(
        root,
        variant="A1",
        learning_rate=a1_learning_rate,
    )
    return {
        "a0_config_path": a0_config,
        "a1_config_path": a1_config,
        "a0_manifest_path": _write_experiment_evidence(
            root,
            variant="A0",
            config_path=a0_config,
            pooled_macro_f1=a0_score,
        ),
        "a1_manifest_path": _write_experiment_evidence(
            root,
            variant="A1",
            config_path=a1_config,
            pooled_macro_f1=a1_score,
        ),
    }


def _build(root: Path, **input_overrides) -> dict:
    return build_g2_augmentation_evidence(
        **_inputs(root, **input_overrides),
        project_root=root,
        evidence_directory=root / "results/evidence/task2/g2_augmentation_ablation",
        figure_directory=root / "results/figures/task2",
    )


def _read_decision(root: Path, manifest: dict) -> dict:
    path = root / "results" / manifest["artifacts"]["decision"]["path"]
    return json.loads(path.read_text(encoding="utf-8"))


def _write_robustness(root: Path, deltas: list[float]) -> Path:
    path = root / "results/evidence/task2/robustness.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "probe": [f"probe_{index}" for index in range(len(deltas))],
            "delta_a1_minus_a0_macro_f1": deltas,
        }
    ).to_csv(path, index=False)
    return path


def test_g2_augmentation_gate_rejects_a1_below_quality_threshold(
    tmp_path: Path,
) -> None:
    manifest = _build(tmp_path, a0_score=0.710, a1_score=0.712)
    result_root = tmp_path / "results"
    comparison = pd.read_csv(
        result_root / manifest["artifacts"]["comparison"]["path"]
    )
    per_class = pd.read_csv(
        result_root / manifest["artifacts"]["per_class_comparison"]["path"]
    )
    decision = _read_decision(tmp_path, manifest)

    assert comparison["variant"].tolist() == ["A0", "A1"]
    assert comparison.loc[1, "delta_vs_a0_macro_f1"] == pytest.approx(0.002)
    assert per_class["label"].tolist() == ["Fall", "Spring", "Summer", "Winter"]
    assert decision["quality_gate_passed"] is False
    assert decision["robustness_evidence_status"] == "not_required"
    assert decision["selected_variant"] == "A0"
    assert manifest["selected_experiment_id"] == "g1-c2-resnet18"
    figure = result_root / manifest["artifacts"]["figure"]["path"]
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_g2_augmentation_gate_stays_pending_without_robustness(
    tmp_path: Path,
) -> None:
    manifest = _build(tmp_path, a0_score=0.710, a1_score=0.713)
    decision = _read_decision(tmp_path, manifest)

    assert decision["quality_gate_passed"] is True
    assert decision["decision_status"] == "pending"
    assert decision["robustness_evidence_status"] == "required"
    assert decision["selected_variant"] is None
    assert manifest["selected_experiment_id"] is None


def test_g2_augmentation_gate_selects_a1_with_acceptable_robustness(
    tmp_path: Path,
) -> None:
    robustness_path = _write_robustness(tmp_path, [-0.004, 0.002, -0.010])
    manifest = build_g2_augmentation_evidence(
        **_inputs(tmp_path, a0_score=0.710, a1_score=0.714),
        project_root=tmp_path,
        evidence_directory=tmp_path
        / "results/evidence/task2/g2_augmentation_ablation",
        figure_directory=tmp_path / "results/figures/task2",
        robustness_evidence_path=robustness_path,
    )
    decision = _read_decision(tmp_path, manifest)

    assert decision["observed_worst_robustness_loss"] == pytest.approx(0.010)
    assert decision["robustness_gate_passed"] is True
    assert decision["selected_variant"] == "A1"
    assert manifest["selected_experiment_id"] == "g2-a1-c2-resnet18"
    assert manifest["robustness_input"]["sha256"] == compute_sha256(robustness_path)


def test_g2_augmentation_gate_retains_a0_when_robustness_loss_is_too_large(
    tmp_path: Path,
) -> None:
    robustness_path = _write_robustness(tmp_path, [-0.011, 0.002])
    manifest = build_g2_augmentation_evidence(
        **_inputs(tmp_path, a0_score=0.710, a1_score=0.714),
        project_root=tmp_path,
        evidence_directory=tmp_path
        / "results/evidence/task2/g2_augmentation_ablation",
        figure_directory=tmp_path / "results/figures/task2",
        robustness_evidence_path=robustness_path,
    )
    decision = _read_decision(tmp_path, manifest)

    assert decision["robustness_gate_passed"] is False
    assert decision["selected_variant"] == "A0"


def test_g2_augmentation_gate_rejects_hidden_optimisation_change(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path, a1_learning_rate=1e-3)

    with pytest.raises(ValueError, match="differ outside"):
        build_g2_augmentation_evidence(
            **inputs,
            project_root=tmp_path,
            evidence_directory=tmp_path
            / "results/evidence/task2/g2_augmentation_ablation",
            figure_directory=tmp_path / "results/figures/task2",
        )


def test_g2_augmentation_quality_panel_does_not_use_truncated_bars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = pd.DataFrame(
        [
            {"variant": "A0", "pooled_macro_f1": 0.707, "spring_f1": 0.738},
            {"variant": "A1", "pooled_macro_f1": 0.697, "spring_f1": 0.727},
        ]
    )
    per_class = pd.DataFrame(
        {
            "label": ["Fall", "Spring", "Summer", "Winter"],
            "delta_a1_minus_a0_f1": [-0.035, -0.011, 0.0004, 0.004],
        }
    )
    bar_calls: list[tuple] = []
    original = Axes.bar

    def capture(self, *args, **kwargs):
        bar_calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", capture)
    _plot_g2_augmentation_ablation(
        comparison,
        per_class,
        minimum_gain=0.003,
        decision_status="closed",
        selected_variant="A0",
        output_path=tmp_path / "g2_a.png",
    )

    assert len(bar_calls) == 1, "only signed per-class deltas may use a bar chart"
