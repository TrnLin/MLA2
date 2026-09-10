from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fashion.task2.bootstrap_evidence as bootstrap_evidence_module
from fashion.data.hashing import compute_sha256
from fashion.task2.bootstrap import PairedBootstrapSpec, load_paired_bootstrap_spec
from fashion.task2.bootstrap_evidence import (
    BOOTSTRAP_IMPLEMENTATION_PATHS,
    build_paired_bootstrap_evidence,
    load_verified_calibration_manifest,
)
from fashion.task2.calibration_evidence import CALIBRATION_IMPLEMENTATION_PATHS
from fashion.task2.slices import CandidateOOFPack
from fashion.train.artifacts import (
    ArtifactVerificationError,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.metrics import SEASON_LABELS


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _declaration(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": compute_sha256(path),
    }


def _declared_files(
    root: Path,
    *,
    directory: str,
    prefix: str,
    count: int,
) -> dict[str, dict[str, str]]:
    return {
        f"{prefix}-{index}": _declaration(
            root,
            _write(root / f"{directory}/{prefix}-{index}.txt", f"{prefix}-{index}\n"),
        )
        for index in range(count)
    }


def _calibration_manifest(root: Path) -> Path:
    runtime_payload = {"python": "3.12.0", "packages": {"scipy": "1.18.1"}}
    runtime = _write(root / "results/calibration/runtime.json", json.dumps(runtime_payload))
    decision = _write(
        root / "results/calibration/decision.json",
        json.dumps(
            {
                "gate": "G6-CALIBRATION",
                "decision_status": "closed",
                "analysis_role": "development_oof_cross_fitted_calibration_only",
                "current_candidate": "I2",
                "candidate_selection_affected": False,
                "ultimate_winner_frozen": False,
                "cross_fitted_evaluation_claim_allowed": True,
                "deployment_temperature_evaluation_claim_allowed": False,
                "app_threshold_frozen": False,
            }
        ),
    )
    artifacts = {
        "calibration_reliability_figure": _declaration(
            root, _write(root / "results/calibration/reliability.png", "png\n")
        ),
        "calibration_summary": _declaration(
            root, _write(root / "results/calibration/summary.csv", "candidate\nC2\n")
        ),
        "decision": _declaration(root, decision),
        "deployment_temperatures": _declaration(
            root, _write(root / "results/calibration/deployment.csv", "temperature\n1.2\n")
        ),
        "fold_temperatures": _declaration(
            root, _write(root / "results/calibration/folds.csv", "temperature\n1.2\n")
        ),
        "reliability_bins": _declaration(
            root, _write(root / "results/calibration/bins.csv", "bin\n0\n")
        ),
        "review_budget_summary": _declaration(
            root, _write(root / "results/calibration/review.csv", "budget\n0.2\n")
        ),
        "risk_coverage": _declaration(
            root, _write(root / "results/calibration/risk.csv", "coverage\n1\n")
        ),
        "risk_coverage_figure": _declaration(
            root, _write(root / "results/calibration/risk.png", "png\n")
        ),
        "runtime": _declaration(root, runtime),
    }
    splits = _write(root / "data/processed/splits.csv", "id\n1\n")
    label_maps = _write(root / "data/processed/label_maps.json", "{}\n")
    analysis = _write(root / "configs/task2/calibration.json", "{}\n")
    robustness = _write(root / "results/robustness/manifest.json", "{}\n")
    slices = _write(root / "results/slices/manifest.json", "{}\n")
    calibrated_oof = _write(root / "tmp/calibration/oof.csv", "id\n1\n")
    manifest = {
        "analysis_config": _declaration(root, analysis),
        "analysis_role": "development_oof_cross_fitted_calibration_only",
        "app_threshold_frozen": False,
        "artifacts": artifacts,
        "candidate_selection_affected": False,
        "canonical_inputs": {
            "label_maps": _declaration(root, label_maps),
            "splits": _declaration(root, splits),
        },
        "cross_fitted_evaluation_claim_allowed": True,
        "decision_status": "closed",
        "deployment_temperature_evaluation_claim_allowed": False,
        "gate": "G6-CALIBRATION",
        "git_commit": "f" * 40,
        "git_dirty": False,
        "holdout_opened": False,
        "implementation_files_at_head": ["src/fashion/task2/calibration.py"],
        "implementation_sha256": "1" * 64,
        "input_predictions": _declared_files(
            root,
            directory="tmp/oof",
            prefix="primary",
            count=10,
        ),
        "labels": list(SEASON_LABELS),
        "robustness_manifest": _declaration(root, robustness),
        "runtime_sha256": canonical_sha256(runtime_payload),
        "schema_version": "1.0.0",
        "slice_manifest": _declaration(root, slices),
        "stability_coverage_sha256": "2" * 64,
        "temporary_calibrated_oof": {
            **_declaration(root, calibrated_oof),
            "row_count": 65_506,
            "rows_per_candidate": {"C2": 32_753, "I2": 32_753},
        },
        "ultimate_winner_frozen": False,
    }
    path = root / "results/calibration/manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_calibration_boundary_verifies_every_declared_file(tmp_path: Path) -> None:
    path = _calibration_manifest(tmp_path)

    manifest, resolved, sections = load_verified_calibration_manifest(
        path,
        project_root=tmp_path,
    )

    assert resolved == path
    assert manifest["gate"] == "G6-CALIBRATION"
    assert len(sections["artifacts"]) == 10
    assert len(sections["input_predictions"]) == 10
    assert sections["decision"]["current_candidate"] == "I2"


def test_calibration_boundary_rejects_prediction_hash_drift(tmp_path: Path) -> None:
    path = _calibration_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    prediction = next(iter(payload["input_predictions"].values()))
    _write(tmp_path / prediction["path"], "changed\n")

    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        load_verified_calibration_manifest(path, project_root=tmp_path)


def test_calibration_boundary_rejects_unknown_fields(tmp_path: Path) -> None:
    path = _calibration_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["notes"] = "drift"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown=.*notes"):
        load_verified_calibration_manifest(path, project_root=tmp_path)


def test_calibration_boundary_cross_checks_runtime_semantic_hash(tmp_path: Path) -> None:
    path = _calibration_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime_sha256"] = "e" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime semantic hash"):
        load_verified_calibration_manifest(path, project_root=tmp_path)


def _small_spec() -> PairedBootstrapSpec:
    return replace(
        load_paired_bootstrap_spec(),
        expected_row_count=8,
        expected_group_count=4,
        replicates=101,
        batch_size=13,
    )


def _development() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(1, 9)),
            "partition": ["development"] * 8,
            "season": [
                "Fall",
                "Fall",
                "Spring",
                "Spring",
                "Summer",
                "Summer",
                "Winter",
                "Winter",
            ],
            "product_family_group": ["g1", "g2", "g2", "g3", "g3", "g4", "g4", "g4"],
        }
    )


def _packs() -> list[CandidateOOFPack]:
    spec = load_paired_bootstrap_spec()
    development = _development()
    true = development.set_index("id")["season"]
    predictions = {
        ("C2", 2753): ["Fall", "Spring", "Spring", "Fall", "Summer", "Winter", "Winter", "Fall"],
        ("I2", 2753): ["Fall", "Fall", "Spring", "Spring", "Summer", "Winter", "Winter", "Winter"],
        ("C2", 2026): ["Fall", "Fall", "Summer", "Spring", "Summer", "Fall", "Winter", "Fall"],
        ("I2", 2026): ["Fall", "Spring", "Spring", "Spring", "Summer", "Summer", "Winter", "Fall"],
    }
    packs = []
    for pair in spec.pairs:
        for candidate, experiment_id in (
            ("C2", pair.c2_experiment_id),
            ("I2", pair.i2_experiment_id),
        ):
            ids = np.asarray([8, 3, 1, 7, 4, 6, 2, 5])
            predicted = np.asarray(predictions[(candidate, pair.seed)], dtype=object)
            frame = pd.DataFrame(
                {
                    "id": ids,
                    "y_true": true.loc[ids].to_numpy(),
                    "y_pred": predicted[ids - 1],
                }
            )
            registry = pd.DataFrame(
                {
                    "run_id": [f"{experiment_id}-f{fold}" for fold in range(5)],
                    "experiment_id": [experiment_id] * 5,
                    "seed": [pair.seed] * 5,
                    "fold": list(range(5)),
                }
            )
            packs.append(
                CandidateOOFPack(
                    candidate=candidate,
                    experiment_id=experiment_id,
                    seed=pair.seed,
                    oof=frame,
                    registry=registry,
                )
            )
    return packs


def test_bootstrap_builder_is_deterministic_and_preserves_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    spec = _small_spec()
    config_path = _write(root / "configs/task2/bootstrap.json", "{}\n")
    calibration_path = _write(root / "results/calibration/manifest.json", "{}\n")
    slice_path = _write(root / "results/slices/manifest.json", "{}\n")
    stability_path = _write(root / "results/stability/manifest.json", "{}\n")
    splits_path = _write(root / "data/processed/splits.csv", "fixture\n")
    label_map_path = _write(
        root / "data/processed/label_maps.json",
        json.dumps(
            {
                "season": {
                    "classes": list(SEASON_LABELS),
                    "label_to_index": {label: index for index, label in enumerate(SEASON_LABELS)},
                }
            }
        ),
    )
    registry_path = _write(root / "results/slices/registry.csv", "run_id\nfixture\n")
    stability_summary_path = _write(root / "results/stability/summary.csv", "candidate\nfixture\n")
    slice_config_path = _write(root / "configs/task2/slices.json", "{}\n")
    input_configs = {
        name: _write(root / f"configs/task2/{name}.json", "{}\n")
        for name in ("c2-primary", "c2-stability", "i2-primary", "i2-stability")
    }
    prediction_declarations = _declared_files(
        root,
        directory="tmp/oof",
        prefix="prediction",
        count=20,
    )
    prediction_paths = {
        run_id: root / declaration["path"]
        for run_id, declaration in prediction_declarations.items()
    }
    coverage = {"row_count": 8, "id_sha256": canonical_sha256(list(range(1, 9)))}
    coverage_hash = canonical_sha256(coverage)
    calibration = {
        "stability_coverage_sha256": coverage_hash,
        "input_predictions": dict(list(prediction_declarations.items())[:10]),
    }
    calibration_sections = {
        "canonical_inputs": {"splits": splits_path, "label_maps": label_map_path},
        "slice_manifest": slice_path,
    }
    slice_manifest = {
        "coverage_sha256": coverage_hash,
        "slice_assignment_sha256": "3" * 64,
        "input_predictions": prediction_declarations,
    }
    slice_sections = {
        "canonical_inputs": {"splits": splits_path, "label_maps": label_map_path},
        "stability_manifest": stability_path,
        "artifacts": {"registry_snapshot": registry_path},
        "analysis_config": slice_config_path,
        "input_predictions": prediction_paths,
    }
    stability = {"coverage_sha256": coverage_hash}
    stability_sections = {
        "artifacts": {"seed_stability": stability_summary_path},
        "input_configs": input_configs,
    }
    splits = pd.concat(
        [
            _development(),
            pd.DataFrame(
                [
                    {
                        "id": 99,
                        "partition": "holdout",
                        "season": "Fall",
                        "product_family_group": "protected-1",
                    },
                    {
                        "id": 100,
                        "partition": "quarantine",
                        "season": "Winter",
                        "product_family_group": "protected-2",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    packs = _packs()
    captured: dict[str, object] = {}

    monkeypatch.setattr(bootstrap_evidence_module, "load_paired_bootstrap_spec", lambda _: spec)

    def fake_load_calibration(path, *, project_root):
        assert Path(path) == calibration_path
        assert Path(project_root) == root
        return calibration, calibration_path, calibration_sections

    def fake_load_slice(path, *, project_root):
        assert Path(path) == slice_path
        assert Path(project_root) == root
        return slice_manifest, slice_path, slice_sections

    def fake_load_stability(path, *, project_root):
        assert Path(path) == stability_path
        assert Path(project_root) == root
        return stability, stability_path, stability_sections

    monkeypatch.setattr(
        bootstrap_evidence_module,
        "load_verified_calibration_manifest",
        fake_load_calibration,
    )
    monkeypatch.setattr(
        bootstrap_evidence_module,
        "load_verified_slice_manifest",
        fake_load_slice,
    )
    monkeypatch.setattr(
        bootstrap_evidence_module,
        "load_verified_stability_manifest",
        fake_load_stability,
    )
    monkeypatch.setattr(bootstrap_evidence_module, "load_splits", lambda _path: splits)

    def fake_get_samples(frame, *, partition, target):
        assert frame is splits
        assert partition == "development"
        assert target == "season"
        return frame.loc[frame["partition"].eq(partition)].copy()

    monkeypatch.setattr(bootstrap_evidence_module, "get_samples", fake_get_samples)
    monkeypatch.setattr(
        bootstrap_evidence_module,
        "load_declared_config_hashes",
        lambda paths: {name: "a" * 64 for name in paths},
    )
    monkeypatch.setattr(bootstrap_evidence_module, "load_slice_analysis_spec", lambda _: object())

    def fake_load_packs(_registry, _slice_spec, **kwargs):
        captured["protected_ids"] = set(kwargs["protected_ids"])
        captured["expected_ids"] = list(kwargs["expected_ids"])
        return (
            packs,
            {pack.experiment_id: coverage for pack in packs},
            prediction_declarations,
        )

    monkeypatch.setattr(bootstrap_evidence_module, "load_candidate_oof_packs", fake_load_packs)
    monkeypatch.setattr(
        bootstrap_evidence_module,
        "capture_git_state",
        lambda _root: {"commit": "f" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        bootstrap_evidence_module,
        "verify_implementation_at_head",
        lambda *_paths, root: ("src/fashion/task2/bootstrap_evidence.py",),
    )
    monkeypatch.setattr(
        bootstrap_evidence_module,
        "implementation_sha256",
        lambda *_paths, root: "1" * 64,
    )
    fixed_runtime = {
        "python": "3.12.0",
        "packages": {"numpy": "2.4.2", "scipy": "1.18.1"},
        "cuda_available": False,
    }
    monkeypatch.setattr(bootstrap_evidence_module, "capture_runtime", lambda: fixed_runtime)
    arguments = {
        "project_root": root,
        "analysis_config_path": config_path,
        "calibration_manifest_path": calibration_path,
        "evidence_directory": root / "results/evidence/task2/paired_bootstrap",
        "figure_directory": root / "results/figures/task2",
        "temporary_directory": root / "tmp/task2/paired_bootstrap",
    }

    first = build_paired_bootstrap_evidence(**arguments)
    second = build_paired_bootstrap_evidence(**arguments)

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert captured["protected_ids"] == {99, 100}
    assert captured["expected_ids"] == list(range(1, 9))
    assert first["holdout_opened"] is False
    assert first["random_seed_generalisability_claim_allowed"] is False
    assert first["ultimate_winner_frozen"] is False
    assert len(first["input_predictions"]) == 20
    assert first["temporary_bootstrap_draws"]["row_count"] == 202
    assert first["temporary_bootstrap_draws"]["rows_per_comparison"] == {
        "primary_interval": 101,
        "stability_sensitivity": 101,
    }
    for declaration in (
        *first["artifacts"].values(),
        *first["input_predictions"].values(),
        first["analysis_config"],
        first["calibration_manifest"],
        first["slice_manifest"],
        first["stability_manifest"],
        *first["canonical_inputs"].values(),
        first["temporary_bootstrap_draws"],
    ):
        declared_path = Path(declaration["path"])
        assert not declared_path.is_absolute()
        assert ".." not in declared_path.parts
        verify_artifact(root / declared_path, declaration["sha256"])
    assert first["manifest_sha256"] == compute_sha256(first["manifest_path"])


def test_bootstrap_implementation_hash_covers_upstream_analysis() -> None:
    assert set(CALIBRATION_IMPLEMENTATION_PATHS).issubset(BOOTSTRAP_IMPLEMENTATION_PATHS)
    assert "src/fashion/task2/bootstrap.py" in BOOTSTRAP_IMPLEMENTATION_PATHS
    assert "src/fashion/task2/bootstrap_evidence.py" in BOOTSTRAP_IMPLEMENTATION_PATHS
