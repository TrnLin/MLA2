from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fashion.task2.calibration_evidence as calibration_evidence_module
from fashion.data.hashing import compute_sha256
from fashion.task2.calibration import (
    CalibrationCandidate,
    CalibrationSpec,
    CalibrationTables,
    load_calibration_spec,
)
from fashion.task2.calibration_evidence import (
    CALIBRATION_IMPLEMENTATION_PATHS,
    build_calibration_decision,
    build_calibration_evidence,
    load_verified_robustness_manifest,
)
from fashion.task2.robustness import ROBUSTNESS_IMPLEMENTATION_PATHS
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


def _robustness_manifest(root: Path) -> Path:
    runtime_payload = {"python": "3.12.0", "packages": {"scipy": "1.18.1"}}
    runtime = _write(
        root / "results/robustness/runtime.json",
        json.dumps(runtime_payload),
    )
    decision = _write(
        root / "results/robustness/decision.json",
        json.dumps(
            {
                "gate": "G6-ROBUSTNESS-COST",
                "decision_status": "closed",
                "current_candidate": "I2",
                "candidate_selection_affected": False,
                "ultimate_winner_frozen": False,
            }
        ),
    )
    artifacts = {
        "decision": _declaration(root, decision),
        "pooled_metrics": _declaration(
            root,
            _write(root / "results/robustness/pooled.csv", "candidate,macro_f1\nI2,0.7\n"),
        ),
        "clean_reconciliation": _declaration(
            root,
            _write(root / "results/robustness/clean.csv", "candidate,agreement\nI2,1\n"),
        ),
        "runtime": _declaration(root, runtime),
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
    manifest = {
        "analysis_config": _declaration(
            root,
            _write(root / "configs/task2/robustness.json", "{}\n"),
        ),
        "analysis_role": "development_stress_and_machine_cost_diagnosis_only",
        "artifacts": artifacts,
        "candidate_selection_affected": False,
        "canonical_inputs": canonical_inputs,
        "decision_status": "closed",
        "gate": "G6-ROBUSTNESS-COST",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "implementation_files_at_head": ["src/fashion/task2/robustness_evidence.py"],
        "implementation_sha256": "b" * 64,
        "input_checkpoints": _declared_files(
            root,
            directory="tmp/checkpoints",
            prefix="checkpoint",
            count=10,
        ),
        "input_cost_results": _declared_files(
            root,
            directory="tmp/cost",
            prefix="cost",
            count=4,
        ),
        "input_probe_predictions": _declared_files(
            root,
            directory="tmp/probes",
            prefix="probe",
            count=50,
        ),
        "runtime_sha256": canonical_sha256(runtime_payload),
        "schema_version": "1.0.0",
        "slice_manifest": _declaration(
            root,
            _write(root / "results/slices/manifest.json", "{}\n"),
        ),
        "stability_coverage_sha256": "d" * 64,
        "ultimate_winner_frozen": False,
    }
    return _write(root / "results/robustness/manifest.json", json.dumps(manifest))


def _decision_tables(*, expected_row_count: int = 32_753) -> CalibrationTables:
    identities = {
        "C2": "g3-c2-t0-resnet18",
        "I2": "g4-i2-article-type-lambda-0-3-c1",
    }
    summary_rows = []
    for candidate, raw_nll, calibrated_nll in (("C2", 0.9, 0.8), ("I2", 0.8, 0.7)):
        for method, nll, brier, ece, confidence in (
            ("uncalibrated", raw_nll, 0.35, 0.08, 0.84),
            ("cross_fitted_temperature", calibrated_nll, 0.31, 0.04, 0.76),
        ):
            summary_rows.append(
                {
                    "candidate": candidate,
                    "calibration_method": method,
                    "accuracy": 0.72,
                    "macro_f1": 0.71,
                    "nll": nll,
                    "brier": brier,
                    "ece": ece,
                    "mean_confidence": confidence,
                }
            )
    fold_size, fold_remainder = divmod(expected_row_count, 5)
    evaluation_support = tuple(fold_size + int(fold < fold_remainder) for fold in range(5))
    fold_temperatures = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "experiment_id": identities[candidate],
                "seed": 2753,
                "evaluation_fold": fold,
                "fit_folds": "|".join(str(value) for value in range(5) if value != fold),
                "fit_fold_count": 4,
                "calibration_rows": expected_row_count - evaluation_rows,
                "evaluation_rows": evaluation_rows,
                "temperature": 1.1 + 0.01 * fold,
                "fit_nll_before": 0.9,
                "fit_nll_after": 0.8,
            }
            for candidate in ("C2", "I2")
            for fold, evaluation_rows in enumerate(evaluation_support)
        ]
    )
    review = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "calibration_method": "cross_fitted_temperature",
                "review_budget": 0.2,
                "review_rate": 0.2,
                "coverage": 0.8,
                "selective_risk": risk,
                "selective_macro_f1": 1.0 - risk,
            }
            for candidate, risk in (("C2", 0.18), ("I2", 0.15))
        ]
    )
    deployment = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "experiment_id": identities[candidate],
                "seed": 2753,
                "temperature": 1.2,
                "fit_rows": expected_row_count,
                "fit_scope": "all_primary_seed_oof_rows",
                "purpose": "future_frozen_bundle_confidence_only",
                "evaluation_claim_allowed": False,
            }
            for candidate in ("C2", "I2")
        ]
    )
    return CalibrationTables(
        calibration_summary=pd.DataFrame(summary_rows),
        fold_temperatures=fold_temperatures,
        reliability_bins=pd.DataFrame(),
        risk_coverage=pd.DataFrame(),
        review_budget_summary=review,
        deployment_temperatures=deployment,
        calibrated_oof=pd.DataFrame(),
    )


def _builder_spec() -> CalibrationSpec:
    return CalibrationSpec(
        analysis_id="g6-cross-fitted-calibration",
        expected_row_count=10,
        candidates=(
            CalibrationCandidate("C2", "g3-c2-t0-resnet18", 2753),
            CalibrationCandidate("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
        ),
        folds=tuple(range(5)),
        probability_floor=1e-12,
        temperature_bounds=(0.05, 10.0),
        optimizer_tolerance=1e-8,
        ece_bins=15,
        coverage_start=0.1,
        coverage_stop=1.0,
        coverage_step=0.01,
        review_budgets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    )


def _builder_tables() -> CalibrationTables:
    base = _decision_tables(expected_row_count=10)
    reliability = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "calibration_method": method,
                "bin": bin_index,
            }
            for candidate in ("C2", "I2")
            for method in ("uncalibrated", "cross_fitted_temperature")
            for bin_index in range(15)
        ]
    )
    risk = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "calibration_method": method,
                "requested_coverage": round(0.1 + step * 0.01, 2),
            }
            for candidate in ("C2", "I2")
            for method in ("uncalibrated", "cross_fitted_temperature")
            for step in range(91)
        ]
    )
    review = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "calibration_method": method,
                "review_budget": budget,
                "review_rate": budget,
                "coverage": 1.0 - budget,
                "selective_risk": 0.2 - budget * 0.1,
                "selective_macro_f1": 0.7 + budget * 0.1,
            }
            for candidate in ("C2", "I2")
            for method in ("uncalibrated", "cross_fitted_temperature")
            for budget in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
        ]
    )
    calibrated_oof = pd.DataFrame(
        [
            {"candidate": candidate, "id": identifier}
            for candidate in ("C2", "I2")
            for identifier in range(1, 11)
        ]
    )
    return CalibrationTables(
        calibration_summary=base.calibration_summary,
        fold_temperatures=base.fold_temperatures,
        reliability_bins=reliability,
        risk_coverage=risk,
        review_budget_summary=review,
        deployment_temperatures=base.deployment_temperatures,
        calibrated_oof=calibrated_oof,
    )


def _builder_pack(
    candidate: str,
    experiment_id: str,
    seed: int,
) -> CandidateOOFPack:
    run_ids = [f"{candidate.lower()}-f{fold}-s{seed}" for fold in range(5)]
    registry = pd.DataFrame(
        {
            "run_id": run_ids,
            "fold": range(5),
            "seed": seed,
            "experiment_id": experiment_id,
        }
    )
    oof = pd.DataFrame(
        {
            "id": range(1, 11),
            "y_true": [SEASON_LABELS[index % 4] for index in range(10)],
        }
    )
    return CandidateOOFPack(
        candidate=candidate,
        experiment_id=experiment_id,
        seed=seed,
        oof=oof,
        registry=registry,
    )


def test_robustness_boundary_verifies_every_declared_input(tmp_path: Path) -> None:
    path = _robustness_manifest(tmp_path)

    manifest, resolved, sections = load_verified_robustness_manifest(
        path.relative_to(tmp_path),
        project_root=tmp_path,
    )

    assert resolved == path
    assert manifest["gate"] == "G6-ROBUSTNESS-COST"
    assert len(sections["input_checkpoints"]) == 10
    assert len(sections["input_probe_predictions"]) == 50
    assert len(sections["input_cost_results"]) == 4


def test_robustness_boundary_rejects_changed_probe_bytes(tmp_path: Path) -> None:
    path = _robustness_manifest(tmp_path)
    _write(tmp_path / "tmp/probes/probe-17.txt", "changed\n")

    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        load_verified_robustness_manifest(path, project_root=tmp_path)


def test_robustness_boundary_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    path = _robustness_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["notes"] = "undeclared drift"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown=.*notes"):
        load_verified_robustness_manifest(path, project_root=tmp_path)


def test_robustness_boundary_cross_checks_runtime_semantic_hash(tmp_path: Path) -> None:
    path = _robustness_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime_sha256"] = "e" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime semantic hash"):
        load_verified_robustness_manifest(path, project_root=tmp_path)


def test_calibration_decision_preserves_g5_candidate_and_threshold_boundary() -> None:
    decision = build_calibration_decision(_decision_tables(), load_calibration_spec())

    assert decision["current_candidate"] == "I2"
    assert decision["candidate_selection_affected"] is False
    assert decision["ultimate_winner_frozen"] is False
    assert decision["cross_fitted_evaluation_claim_allowed"] is True
    assert decision["deployment_temperature_evaluation_claim_allowed"] is False
    assert decision["app_threshold_frozen"] is False
    assert decision["probability_quality_by_candidate"]["I2"]["nll_improved"] is True


def test_calibration_decision_rejects_changed_class_predictions() -> None:
    tables = _decision_tables()
    mask = tables.calibration_summary["calibration_method"].eq("cross_fitted_temperature")
    tables.calibration_summary.loc[mask, "accuracy"] = 0.73

    with pytest.raises(ValueError, match="changed class predictions"):
        build_calibration_decision(tables, load_calibration_spec())


def test_calibration_decision_rejects_string_deployment_flag() -> None:
    tables = _decision_tables()
    tables.deployment_temperatures["evaluation_claim_allowed"] = "false"

    with pytest.raises(ValueError, match="deployment-temperature boundary"):
        build_calibration_decision(tables, load_calibration_spec())


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_evaluation_fold", "wrong_fit_scope", "wrong_fit_rows", "nonfinite_temperature"],
)
def test_calibration_decision_rejects_temperature_audit_drift(mutation: str) -> None:
    tables = _decision_tables()
    if mutation == "duplicate_evaluation_fold":
        tables.fold_temperatures.loc[0, "evaluation_fold"] = 1
    elif mutation == "wrong_fit_scope":
        tables.deployment_temperatures.loc[0, "fit_scope"] = "held_out_fold"
    elif mutation == "wrong_fit_rows":
        tables.deployment_temperatures.loc[0, "fit_rows"] = 32_752
    else:
        tables.deployment_temperatures.loc[0, "temperature"] = np.nan

    with pytest.raises(ValueError):
        build_calibration_decision(tables, load_calibration_spec())


def test_calibration_builder_is_deterministic_and_preserves_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    spec = _builder_spec()
    config_path = _write(root / "configs/task2/calibration.json", "{}\n")
    robustness_path = _write(root / "results/robustness/manifest.json", "{}\n")
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
    splits = pd.DataFrame(
        [
            {
                "id": identifier,
                "partition": "development",
                "season": SEASON_LABELS[(identifier - 1) % 4],
            }
            for identifier in range(1, 11)
        ]
        + [
            {"id": 99, "partition": "holdout", "season": "Fall"},
            {"id": 100, "partition": "quarantine", "season": "Winter"},
        ]
    )
    packs = [
        _builder_pack("C2", "g3-c2-t0-resnet18", 2753),
        _builder_pack("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
        _builder_pack("C2", "c2-stability", 2026),
        _builder_pack("I2", "i2-stability", 2026),
    ]
    prediction_artifacts: dict[str, dict[str, str]] = {}
    for pack in packs:
        for run_id in pack.registry["run_id"].astype(str):
            prediction_artifacts[run_id] = _declaration(
                root,
                _write(root / f"tmp/oof/{run_id}.csv", f"run_id\n{run_id}\n"),
            )
    slice_manifest = {"input_predictions": prediction_artifacts}
    coverage = {"row_count": 10, "id_sha256": canonical_sha256(list(range(1, 11)))}
    coverage_sha256 = canonical_sha256(coverage)
    robustness = {"stability_coverage_sha256": coverage_sha256}
    robustness_sections = {
        "canonical_inputs": {"splits": splits_path, "label_maps": label_map_path},
        "slice_manifest": slice_path,
    }
    slice_sections = {
        "canonical_inputs": {"splits": splits_path, "label_maps": label_map_path},
        "stability_manifest": stability_path,
        "artifacts": {"registry_snapshot": registry_path},
        "analysis_config": slice_config_path,
        "input_predictions": {
            run_id: root / declaration["path"]
            for run_id, declaration in prediction_artifacts.items()
        },
    }
    stability = {"coverage_sha256": coverage_sha256}
    stability_sections = {
        "artifacts": {"seed_stability": stability_summary_path},
        "input_configs": input_configs,
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(calibration_evidence_module, "load_calibration_spec", lambda _: spec)

    def fake_load_robustness(
        path: str | Path,
        *,
        project_root: str | Path,
    ) -> tuple[dict[str, str], Path, dict[str, object]]:
        assert Path(path) == robustness_path
        assert Path(project_root) == root
        return robustness, robustness_path, robustness_sections

    def fake_load_slice(
        path: str | Path,
        *,
        project_root: str | Path,
    ) -> tuple[dict[str, object], Path, dict[str, object]]:
        assert Path(path) == robustness_sections["slice_manifest"]
        assert Path(project_root) == root
        return slice_manifest, slice_path, slice_sections

    def fake_load_stability(
        path: str | Path,
        *,
        project_root: str | Path,
    ) -> tuple[dict[str, str], Path, dict[str, object]]:
        assert Path(path) == slice_sections["stability_manifest"]
        assert Path(project_root) == root
        return stability, stability_path, stability_sections

    monkeypatch.setattr(
        calibration_evidence_module,
        "load_verified_robustness_manifest",
        fake_load_robustness,
    )
    monkeypatch.setattr(
        calibration_evidence_module,
        "load_verified_slice_manifest",
        fake_load_slice,
    )
    monkeypatch.setattr(
        calibration_evidence_module,
        "load_verified_stability_manifest",
        fake_load_stability,
    )
    monkeypatch.setattr(calibration_evidence_module, "load_splits", lambda _path: splits)

    def fake_get_samples(
        frame: pd.DataFrame,
        *,
        partition: str,
        target: str,
    ) -> pd.DataFrame:
        assert frame is splits
        assert partition == "development"
        assert target == "season"
        return frame.loc[frame["partition"].eq(partition), ["id", "season"]].copy()

    monkeypatch.setattr(calibration_evidence_module, "get_samples", fake_get_samples)
    monkeypatch.setattr(
        calibration_evidence_module,
        "load_declared_config_hashes",
        lambda paths: {name: "a" * 64 for name in paths},
    )
    monkeypatch.setattr(
        calibration_evidence_module,
        "load_slice_analysis_spec",
        lambda _path: object(),
    )

    def fake_load_packs(
        _registry: pd.DataFrame,
        _slice_spec: object,
        **kwargs: object,
    ) -> tuple[list[CandidateOOFPack], dict[str, object], dict[str, dict[str, str]]]:
        captured["protected_ids"] = set(kwargs["protected_ids"])
        captured["expected_ids"] = list(kwargs["expected_ids"])
        return (
            packs,
            {pack.experiment_id: coverage for pack in packs},
            prediction_artifacts,
        )

    monkeypatch.setattr(calibration_evidence_module, "load_candidate_oof_packs", fake_load_packs)
    monkeypatch.setattr(
        calibration_evidence_module,
        "capture_git_state",
        lambda _root: {"commit": "f" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        calibration_evidence_module,
        "verify_implementation_at_head",
        lambda *_paths, root: ("src/fashion/task2/calibration_evidence.py",),
    )
    monkeypatch.setattr(
        calibration_evidence_module,
        "implementation_sha256",
        lambda *_paths, root: "1" * 64,
    )
    fixed_runtime = {
        "python": "3.12.0",
        "packages": {"scipy": "1.18.1"},
        "cuda_available": False,
    }
    monkeypatch.setattr(calibration_evidence_module, "capture_runtime", lambda: fixed_runtime)

    def fake_analyse(
        selected: list[CandidateOOFPack],
        _spec: CalibrationSpec,
    ) -> CalibrationTables:
        captured["selected"] = [
            (pack.candidate, pack.experiment_id, pack.seed) for pack in selected
        ]
        return _builder_tables()

    monkeypatch.setattr(calibration_evidence_module, "analyse_calibration_packs", fake_analyse)

    def fake_plot(_frame: pd.DataFrame, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\nfixed-calibration-fixture")
        return path

    monkeypatch.setattr(calibration_evidence_module, "plot_calibration_reliability", fake_plot)
    monkeypatch.setattr(calibration_evidence_module, "plot_risk_coverage", fake_plot)
    arguments = {
        "project_root": root,
        "analysis_config_path": config_path,
        "robustness_manifest_path": robustness_path,
        "evidence_directory": root / "results/evidence/task2/calibration",
        "figure_directory": root / "results/figures/task2",
        "temporary_directory": root / "tmp/task2/calibration",
    }

    first = build_calibration_evidence(**arguments)
    second = build_calibration_evidence(**arguments)

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert captured["protected_ids"] == {99, 100}
    assert captured["expected_ids"] == list(range(1, 11))
    assert captured["selected"] == [
        ("C2", "g3-c2-t0-resnet18", 2753),
        ("I2", "g4-i2-article-type-lambda-0-3-c1", 2753),
    ]
    assert len(first["input_predictions"]) == 10
    assert all("s2753" in run_id for run_id in first["input_predictions"])
    assert first["holdout_opened"] is False
    assert first["app_threshold_frozen"] is False
    assert first["cross_fitted_evaluation_claim_allowed"] is True
    assert first["deployment_temperature_evaluation_claim_allowed"] is False
    assert first["temporary_calibrated_oof"]["rows_per_candidate"] == {"C2": 10, "I2": 10}
    for declaration in (
        *first["artifacts"].values(),
        *first["input_predictions"].values(),
        first["analysis_config"],
        first["robustness_manifest"],
        first["slice_manifest"],
        *first["canonical_inputs"].values(),
        first["temporary_calibrated_oof"],
    ):
        declared_path = Path(declaration["path"])
        assert not declared_path.is_absolute()
        assert ".." not in declared_path.parts
        verify_artifact(root / declared_path, declaration["sha256"])
    assert first["manifest_sha256"] == compute_sha256(first["manifest_path"])

    stored_manifest = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))

    def declared_paths(value: object) -> list[str]:
        if isinstance(value, dict):
            paths = [str(value["path"])] if "path" in value else []
            return paths + [path for nested in value.values() for path in declared_paths(nested)]
        if isinstance(value, list):
            return [path for nested in value for path in declared_paths(nested)]
        return []

    for value in declared_paths(stored_manifest):
        assert not Path(value).is_absolute()
        assert ".." not in Path(value).parts


def test_calibration_implementation_hash_covers_analysis_and_runtime_provenance() -> None:
    assert set(ROBUSTNESS_IMPLEMENTATION_PATHS).issubset(CALIBRATION_IMPLEMENTATION_PATHS)
    assert "src/fashion/task2/calibration.py" in CALIBRATION_IMPLEMENTATION_PATHS
    assert "src/fashion/task2/calibration_evidence.py" in CALIBRATION_IMPLEMENTATION_PATHS
    assert "src/fashion/task2/stability.py" in CALIBRATION_IMPLEMENTATION_PATHS
    assert "src/fashion/task2/stability_evidence.py" in CALIBRATION_IMPLEMENTATION_PATHS
