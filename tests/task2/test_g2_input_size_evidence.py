from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.axes import Axes

from fashion.data.hashing import compute_sha256
from fashion.task2.evidence import (
    _plot_g2_input_size_ablation,
    build_g2_input_size_evidence,
)
from fashion.task2.experiments import load_experiment_config
from fashion.train.artifacts import canonical_sha256


def _write_config(
    root: Path,
    *,
    variant: str,
    image_size: tuple[int, int],
    learning_rate: float = 3e-4,
) -> Path:
    experiment_id = {
        "P0": "g1-c2-resnet18",
        "P1": "g2-p1-c2-resnet18",
    }[variant]
    stage = {"P0": "g1_family_screen", "P1": "g2_input_size_ablation"}[
        variant
    ]
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
                    "image_size": list(image_size),
                    "augmentation": "a0",
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
        "P0": "g1-c2-resnet18",
        "P1": "g2-p1-c2-resnet18",
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
    pooled_path.write_text(
        json.dumps(
            {
                "macro_f1": pooled_macro_f1,
                "per_class": {
                    label: {
                        "precision": pooled_macro_f1,
                        "recall": pooled_macro_f1,
                        "f1": pooled_macro_f1 + (0.02 if label == "Spring" else 0.0),
                        "support": 5,
                    }
                    for label in ("Fall", "Spring", "Summer", "Winter")
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
    size_token = "80x60" if variant == "P0" else "128x96"
    runtime = 60.0 if variant == "P0" else 120.0
    vram = 600.0 if variant == "P0" else 1300.0
    pd.DataFrame(
        [
            {
                "run_id": run_ids[fold],
                "stage": {
                    "P0": "g1_family_screen",
                    "P1": "g2_input_size_ablation",
                }[variant],
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
                "transform_id": f"a0-{size_token}",
                "loss_id": "cross_entropy",
                "primary_metric_value": fold_scores[fold],
                "runtime_seconds": runtime + fold,
                "peak_vram_mb": vram,
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
    p0_score: float = 0.710,
    p1_score: float = 0.716,
    p1_learning_rate: float = 3e-4,
) -> dict[str, Path]:
    p0_config = _write_config(root, variant="P0", image_size=(80, 60))
    p1_config = _write_config(
        root,
        variant="P1",
        image_size=(128, 96),
        learning_rate=p1_learning_rate,
    )
    return {
        "p0_config_path": p0_config,
        "p1_config_path": p1_config,
        "p0_manifest_path": _write_experiment_evidence(
            root,
            variant="P0",
            config_path=p0_config,
            pooled_macro_f1=p0_score,
        ),
        "p1_manifest_path": _write_experiment_evidence(
            root,
            variant="P1",
            config_path=p1_config,
            pooled_macro_f1=p1_score,
        ),
    }


def _build(root: Path, **input_overrides) -> dict:
    return build_g2_input_size_evidence(
        **_inputs(root, **input_overrides),
        project_root=root,
        evidence_directory=root / "results/evidence/task2/g2_input_size_ablation",
        figure_directory=root / "results/figures/task2",
    )


def test_g2_size_gate_selects_p1_only_at_frozen_gain(tmp_path: Path) -> None:
    manifest = _build(tmp_path, p0_score=0.710, p1_score=0.716)
    result_root = tmp_path / "results"
    comparison = pd.read_csv(
        result_root / manifest["artifacts"]["comparison"]["path"]
    )
    paired = pd.read_csv(
        result_root / manifest["artifacts"]["paired_fold_metrics"]["path"]
    )
    decision = json.loads(
        (result_root / manifest["artifacts"]["decision"]["path"]).read_text(
            encoding="utf-8"
        )
    )

    assert comparison["variant"].tolist() == ["P0", "P1"]
    assert comparison.loc[1, "delta_vs_p0_macro_f1"] == pytest.approx(0.006)
    assert comparison.loc[1, "runtime_ratio_vs_p0"] > 1.0
    assert len(paired) == 5
    assert decision["minimum_gain"] == pytest.approx(0.005)
    assert decision["selected_variant"] == "P1"
    assert manifest["selected_experiment_id"] == "g2-p1-c2-resnet18"
    figure = result_root / manifest["artifacts"]["figure"]["path"]
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_g2_size_gate_retains_p0_below_threshold(tmp_path: Path) -> None:
    manifest = _build(tmp_path, p0_score=0.710, p1_score=0.714)
    decision_path = tmp_path / "results" / manifest["artifacts"]["decision"]["path"]
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    assert decision["observed_p1_minus_p0_macro_f1"] == pytest.approx(0.004)
    assert decision["selected_variant"] == "P0"
    assert decision["selected_experiment_id"] == "g1-c2-resnet18"


def test_g2_size_gate_rejects_an_optimisation_change(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, p1_learning_rate=1e-3)

    with pytest.raises(ValueError, match="differ outside"):
        build_g2_input_size_evidence(
            **inputs,
            project_root=tmp_path,
            evidence_directory=tmp_path
            / "results/evidence/task2/g2_input_size_ablation",
            figure_directory=tmp_path / "results/figures/task2",
        )


def test_g2_quality_panel_does_not_use_truncated_bars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = pd.DataFrame(
        [
            {"variant": "P0", "pooled_macro_f1": 0.707, "spring_f1": 0.738},
            {
                "variant": "P1",
                "pooled_macro_f1": 0.705,
                "spring_f1": 0.739,
                "delta_vs_p0_macro_f1": -0.002,
            },
        ]
    )
    paired = pd.DataFrame(
        {
            "fold": list(range(5)),
            "delta_p1_minus_p0_macro_f1": [-0.004, -0.005, -0.001, 0.005, -0.002],
        }
    )
    bar_calls: list[tuple] = []
    original = Axes.bar

    def capture(self, *args, **kwargs):
        bar_calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", capture)
    _plot_g2_input_size_ablation(
        comparison,
        paired,
        minimum_gain=0.005,
        selected_variant="P0",
        output_path=tmp_path / "g2.png",
    )

    assert len(bar_calls) == 1, "only signed fold deltas should use a zero-based bar chart"
