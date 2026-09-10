from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.task2.model_comparison import (
    DEVELOPMENT_MODEL_SPECS,
    build_development_model_comparison_evidence,
    load_verified_development_model_comparison,
)


def test_complete_development_comparison_covers_every_registered_configuration(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    figure = tmp_path / "figures" / "all_models.png"
    manifest = build_development_model_comparison_evidence(
        project_root=ROOT,
        evidence_directory=evidence,
        figure_path=figure,
    )

    assert manifest["source_partition"] == "development_oof"
    assert manifest["model_count"] == 20 == len(DEVELOPMENT_MODEL_SPECS)
    assert "does not create, replace, or compare internal-holdout predictions" in manifest[
        "claim_boundary"
    ]
    catalog = pd.read_csv(evidence / "all_model_oof_metrics.csv")
    assert catalog["experiment_id"].tolist() == [
        spec.experiment_id for spec in DEVELOPMENT_MODEL_SPECS
    ]
    assert set(catalog["n_samples"]) == {32_753}
    assert catalog["coverage_id_set_sha256"].nunique() == 1
    assert set(catalog["seed"]) == {2753, 2026}
    assert catalog["coverage_id_set_sha256"].nunique() == 1
    assert catalog["split_sha256"].nunique() == 1
    assert catalog["label_map_sha256"].nunique() == 1
    assert manifest["canonical_inputs"] == {
        "coverage_id_set_sha256": catalog["coverage_id_set_sha256"].iloc[0],
        "split_sha256": catalog["split_sha256"].iloc[0],
        "label_map_sha256": catalog["label_map_sha256"].iloc[0],
    }
    assert manifest["selection_freeze"]["selected_experiment_id"] == (
        "g4-i2-article-type-lambda-0-3-c1"
    )
    assert set(catalog["phase"]) == {
        "Baselines",
        "Family screen",
        "Controlled ablations",
        "Compact tuning",
        "Full-budget finalists",
        "Targeted interventions",
        "Pretraining benchmark",
        "Seed stability",
    }
    assert catalog.loc[catalog["benchmark_only"], "experiment_id"].tolist() == [
        "g4-p0s-resnet18-standard-scratch",
        "g4-pstar-resnet18-standard-pretrained"
    ]
    assert catalog.loc[catalog["selected_for_freeze"], "experiment_id"].tolist() == [
        "g4-i2-article-type-lambda-0-3-c1"
    ]
    assert catalog["macro_f1"].between(0, 1).all()
    assert catalog["accuracy"].between(0, 1).all()
    assert catalog["balanced_accuracy"].between(0, 1).all()
    png_header = figure.read_bytes()[:24]
    assert png_header[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png_header[16:24])
    assert width >= 2_000
    assert height >= 1_500
    for declaration in manifest["artifacts"].values():
        path = Path(declaration["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            path = evidence / Path(declaration["path"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == declaration["sha256"]


def test_comparison_loader_rejects_a_changed_catalog(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    manifest = build_development_model_comparison_evidence(
        project_root=ROOT,
        evidence_directory=evidence,
        figure_path=tmp_path / "comparison.png",
    )
    catalog_path = evidence / "all_model_oof_metrics.csv"
    catalog_path.write_text(
        catalog_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="catalog artifact hash"):
        load_verified_development_model_comparison(
            manifest["manifest_path"],
            project_root=ROOT,
        )


def test_comparison_loader_rejects_a_changed_selection_freeze_link(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    manifest = build_development_model_comparison_evidence(
        project_root=ROOT,
        evidence_directory=evidence,
        figure_path=tmp_path / "comparison.png",
    )
    manifest_path = Path(manifest["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["selection_freeze"]["selected_experiment_id"] = "g3-c2-t0-resnet18"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="disagrees with the selection freeze"):
        load_verified_development_model_comparison(manifest_path, project_root=ROOT)
