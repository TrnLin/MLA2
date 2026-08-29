from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from fashion.data.hashing import compute_sha256
from fashion.task2.robustness import RobustnessTables, load_robustness_cost_spec
from fashion.task2.robustness_evidence import (
    _plot_deployment_cost,
    _plot_robustness,
    build_robustness_cost_decision,
    load_verified_slice_manifest,
)
from fashion.train.artifacts import ArtifactVerificationError


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _declaration(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": compute_sha256(path),
    }


def _slice_manifest(root: Path) -> Path:
    decision = _write(
        root / "results/decision.json",
        json.dumps(
            {
                "current_candidate": "I2",
                "candidate_selection_affected": False,
                "ultimate_winner_frozen": False,
            }
        ),
    )
    artifacts = {
        "registry_snapshot": _declaration(
            root,
            _write(root / "results/registry.csv", "run_id\nrun-1\n"),
        ),
        "decision": _declaration(root, decision),
        "slice_metrics": _declaration(
            root,
            _write(root / "results/slice_metrics.csv", "candidate,macro_f1\nI2,0.7\n"),
        ),
    }
    canonical_inputs = {
        "splits": _declaration(
            root,
            _write(root / "data/processed/splits.csv", "id,partition\n1,development\n"),
        ),
        "label_maps": _declaration(
            root,
            _write(root / "data/processed/label_maps.json", "{}\n"),
        ),
    }
    analysis_config = _declaration(
        root,
        _write(root / "configs/task2/slices.json", "{}\n"),
    )
    stability_manifest = _declaration(
        root,
        _write(root / "results/stability/manifest.json", "{}\n"),
    )
    input_predictions = {
        f"run-{index}": _declaration(
            root,
            _write(root / f"tmp/run-{index}/oof.csv", f"id\n{index}\n"),
        )
        for index in range(20)
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "G6-SLICE",
        "decision_status": "closed",
        "analysis_role": "development_oof_diagnosis_only",
        "candidate_selection_affected": False,
        "ultimate_winner_frozen": False,
        "artifacts": artifacts,
        "canonical_inputs": canonical_inputs,
        "analysis_config": analysis_config,
        "stability_manifest": stability_manifest,
        "input_predictions": input_predictions,
    }
    path = root / "results/slices/manifest.json"
    return _write(path, json.dumps(manifest))


def _tables() -> RobustnessTables:
    conditions = (
        "clean",
        "jpeg_quality_85",
        "brightness_0_85",
        "brightness_1_15",
        "gaussian_blur_radius_1",
    )
    rows = []
    comparison = []
    for index, condition in enumerate(conditions):
        c2_macro_f1 = 0.72 - index * 0.02
        i2_macro_f1 = 0.75 - index * 0.015
        for candidate, value, clean in (
            ("C2", c2_macro_f1, 0.72),
            ("I2", i2_macro_f1, 0.75),
        ):
            delta = value - clean
            rows.append(
                {
                    "candidate": candidate,
                    "condition": condition,
                    "macro_f1": value,
                    "delta_macro_f1_vs_clean": delta,
                    "prediction_agreement_with_clean": 1.0 - index * 0.03,
                    "material_macro_f1_degradation": delta < -0.01,
                }
            )
        comparison.append(
            {
                "condition": condition,
                "i2_minus_c2_macro_f1": i2_macro_f1 - c2_macro_f1,
            }
        )
    return RobustnessTables(
        pooled_metrics=pd.DataFrame(rows),
        fold_metrics=pd.DataFrame(),
        candidate_comparison=pd.DataFrame(comparison),
    )


def _cost() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": candidate,
                "device": device,
                "available": True,
                "model_only_median_ms": latency,
                "end_to_end_median_ms": latency + 1.0,
                "parameter_and_buffer_bytes": size,
                "training_checkpoint_bytes": size * 3,
            }
            for candidate, size, cpu_latency, cuda_latency in (
                ("C2", 44_000_000, 4.0, 1.0),
                ("I2", 4_800_000, 2.0, 0.7),
            )
            for device, latency in (("cpu", cpu_latency), ("cuda", cuda_latency))
        ]
    )


def test_slice_boundary_verifies_every_required_input(tmp_path: Path) -> None:
    path = _slice_manifest(tmp_path)

    manifest, resolved, sections = load_verified_slice_manifest(
        path.relative_to(tmp_path),
        project_root=tmp_path,
    )

    assert resolved == path
    assert manifest["gate"] == "G6-SLICE"
    assert set(sections["canonical_inputs"]) == {"splits", "label_maps"}
    assert len(sections["input_predictions"]) == 20


def test_slice_boundary_rejects_changed_prediction_bytes(tmp_path: Path) -> None:
    path = _slice_manifest(tmp_path)
    _write(tmp_path / "tmp/run-3/oof.csv", "changed\n")

    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        load_verified_slice_manifest(path, project_root=tmp_path)


def test_decision_preserves_g5_candidate_and_reports_cost_ratios() -> None:
    decision = build_robustness_cost_decision(
        _tables(),
        _cost(),
        load_robustness_cost_spec(),
    )

    assert decision["current_candidate"] == "I2"
    assert decision["candidate_selection_affected"] is False
    assert decision["ultimate_winner_frozen"] is False
    assert decision["i2_above_c2_in_every_condition"] is True
    assert decision["model_size"]["i2_to_c2_parameter_and_buffer_ratio"] < 1


def test_robustness_and_cost_figures_are_nonempty_pngs(tmp_path: Path) -> None:
    robustness_path = _plot_robustness(
        _tables(),
        tmp_path / "robustness.png",
        material_threshold=0.01,
    )
    cost_path = _plot_deployment_cost(_cost(), tmp_path / "cost.png")

    for path in (robustness_path, cost_path):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 25_000
        with Image.open(path) as image:
            assert image.width >= 2_000
            assert image.height >= 700
