from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import fashion.task2.handoff as handoff_module
from fashion.data.hashing import compute_sha256
from fashion.task2.handoff import (
    audit_task2_artifacts,
    build_task2_handoff_evidence,
    load_verified_task2_handoff,
)
from fashion.task2.inference import SeasonPrediction
from fashion.train.artifacts import ArtifactVerificationError


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _declaration(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": compute_sha256(path),
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    partition: str = "development",
) -> dict[str, object]:
    root = tmp_path.resolve()
    ultimate_path = _write(root / "ultimate/manifest.json", "{}\n")
    model_manifest = _write(root / "models/task2_season.manifest.json", "{}\n")
    freeze_path = _write(root / "evidence/selection_freeze.json", "{}\n")
    bundle_path = _write(root / "models/task2_season.pt", "weights")
    history_path = _write(root / "evidence/history.csv", "epoch,loss\n1,1.0\n")
    runtime_path = _write(root / "evidence/runtime.json", "{}\n")
    label_maps_path = _write(root / "data/label_maps.json", "{}\n")
    inference_source = _write(root / "src/fashion/task2/inference.py", "# image only\n")
    registry_path = _write(root / "results/runs.csv", "run_id,status\nrefit-run,completed\n")
    image_path = _write(root / "data/images/1163.jpg", "fixture")
    splits_path = root / "data/splits.csv"
    pd.DataFrame(
        [
            {
                "id": "1163",
                "path": image_path.relative_to(root).as_posix(),
                "partition": partition,
                "has_season_label": True,
            }
        ]
    ).to_csv(splits_path, index=False)
    freeze_declaration = _declaration(freeze_path, root)
    ultimate = {
        "selected_candidate": "I2",
        "selected_experiment_id": "g4-i2",
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "artifacts": {"selection_freeze": freeze_declaration},
    }
    refit = {
        "selected_candidate": "I2",
        "selected_experiment_id": "g4-i2",
        "run_id": "refit-run",
        "holdout_opened": False,
        "holdout_metrics_present": False,
        "scratch": True,
        "weights": None,
        "final_eligible": True,
        "selection_freeze": freeze_declaration,
        "bundle": _declaration(bundle_path, root),
        "artifacts": {
            "history": _declaration(history_path, root),
            "runtime": _declaration(runtime_path, root),
        },
        "canonical_inputs": {
            "splits": _declaration(splits_path, root),
            "label_maps": _declaration(label_maps_path, root),
        },
    }
    freeze = {
        "limitations": ["brightness shift remains difficult"],
    }
    monkeypatch.setattr(
        handoff_module,
        "load_verified_ultimate_judgement_manifest",
        lambda *args, **kwargs: (
            ultimate,
            ultimate_path,
            {"selection_freeze": freeze_path},
        ),
    )
    monkeypatch.setattr(
        handoff_module,
        "load_verified_development_refit_manifest",
        lambda *args, **kwargs: (
            refit,
            model_manifest,
            {"labels": list(handoff_module.SEASON_LABELS)},
        ),
    )
    monkeypatch.setattr(
        handoff_module,
        "load_verified_selection_freeze",
        lambda *args, **kwargs: (freeze, freeze_path),
    )
    prediction = SeasonPrediction(
        image_path=image_path.as_posix(),
        predicted_label="Summer",
        probabilities={
            "Fall": 0.05,
            "Spring": 0.05,
            "Summer": 0.8,
            "Winter": 0.1,
        },
        confidence=0.8,
        review_required=None,
        latency_ms=12.5,
        run_id="refit-run",
        manifest_sha256=compute_sha256(model_manifest),
        bundle_sha256=compute_sha256(bundle_path),
    )
    return {
        "root": root,
        "ultimate_path": ultimate_path,
        "model_manifest": model_manifest,
        "registry_path": registry_path,
        "inference_source": inference_source,
        "prediction": prediction,
    }


def test_handoff_audits_and_packages_only_the_locked_task2_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )

    manifest, manifest_path = build_task2_handoff_evidence(
        audit,
        fixture["prediction"],
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        model_manifest_path=fixture["model_manifest"],
        output_directory=Path(fixture["root"]) / "evidence/final_handoff",
    )
    verified, verified_path, verified_audit, smoke = load_verified_task2_handoff(
        manifest_path,
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
    )

    assert set(audit["artifact"]) == handoff_module._REQUIRED_AUDIT_ARTIFACTS
    assert audit["status"].eq("PASS").all()
    assert manifest["status"] == "ready_for_group_freeze"
    assert manifest["task2_component_ready"] is True
    assert manifest["group_freeze_verified"] is False
    assert manifest["notebook_06_unlocked"] is False
    assert manifest["holdout_opened"] is False
    assert manifest["evaluation_claim_allowed"] is False
    assert verified == manifest
    assert verified_path == manifest_path
    assert len(verified_audit) == len(audit)
    assert smoke["input_scope"] == "one labelled development image"
    assert smoke["holdout_opened"] is False
    assert "latency_ms" not in smoke


def test_handoff_rejects_inference_from_a_different_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    changed = replace(fixture["prediction"], manifest_sha256="a" * 64)

    with pytest.raises(ValueError, match="different model manifest"):
        build_task2_handoff_evidence(
            audit,
            changed,
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
            model_manifest_path=fixture["model_manifest"],
            output_directory=Path(fixture["root"]) / "evidence/final_handoff",
        )


def test_handoff_rejects_a_protected_smoke_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, partition="holdout")
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )

    with pytest.raises(ValueError, match="labelled development data"):
        build_task2_handoff_evidence(
            audit,
            fixture["prediction"],
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
            model_manifest_path=fixture["model_manifest"],
            output_directory=Path(fixture["root"]) / "evidence/final_handoff",
        )


def test_handoff_verifier_rejects_tampered_smoke_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    audit = audit_task2_artifacts(
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        ultimate_manifest_path=fixture["ultimate_path"],
        model_manifest_path=fixture["model_manifest"],
    )
    _, manifest_path = build_task2_handoff_evidence(
        audit,
        fixture["prediction"],
        project_root=fixture["root"],
        registry_path=fixture["registry_path"],
        model_manifest_path=fixture["model_manifest"],
        output_directory=Path(fixture["root"]) / "evidence/final_handoff",
    )
    smoke_path = manifest_path.parent / "inference_smoke.json"
    smoke_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match="SHA-256 mismatch"):
        load_verified_task2_handoff(
            manifest_path,
            project_root=fixture["root"],
            registry_path=fixture["registry_path"],
        )


def test_handoff_source_has_no_final_evaluation_unlock() -> None:
    source = Path(handoff_module.__file__).read_text(encoding="utf-8")

    assert "evaluation_unlocked" not in source
    assert "load_protected" not in source
    assert '"notebook_06_unlocked": False' in source
