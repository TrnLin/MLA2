from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from fashion.config import ROOT
from fashion.task2.evidence import build_task2_selection_story_evidence


def test_selection_story_links_eda_to_incremental_model_choices(tmp_path: Path) -> None:
    output = tmp_path / "selection_story"
    manifest = build_task2_selection_story_evidence(
        project_root=ROOT,
        evidence_directory=output,
    )

    assert manifest["selected_finalist_experiment_ids"] == [
        "g2-t1-c1-smallcnn",
        "g1-c2-resnet18",
    ]
    assert set(manifest["input_manifests"]) == {
        "B0",
        "B1",
        "G1",
        "G2-P",
        "G2-A",
        "G2-T",
    }
    for declaration in manifest["artifacts"].values():
        path = output / Path(declaration["path"]).name
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == declaration["sha256"]

    ladder = pd.read_csv(output / "incremental_model_selection.csv")
    assert ladder["step"].tolist() == [
        "B0 majority",
        "B1 HOG + HSV",
        "C1 SmallCNN T0",
        "C2 ResNet18 T0",
        "C3 MobileNetV3 alternative",
        "C1-T1 selected finalist",
        "C2-T0 retained finalist",
    ]
    assert ladder.loc[0, "pooled_macro_f1"] == pytest.approx(0.165704, abs=1e-6)
    assert ladder.loc[1, "pooled_macro_f1"] == pytest.approx(0.609561, abs=1e-6)
    assert ladder.loc[5, "pooled_macro_f1"] == pytest.approx(0.708075, abs=1e-6)

    reflection = pd.read_csv(output / "eda_reflection.csv")
    assert reflection["verdict_after_measured_gates"].value_counts().to_dict() == {
        "supported": 3,
        "contradicted": 2,
        "partly supported": 1,
        "still untested": 1,
    }
    unresolved = reflection.loc[
        reflection["verdict_after_measured_gates"].eq("still untested"),
        "earlier_eda_insight",
    ].item()
    for warning in ("ArticleType", "file size", "acquisition year"):
        assert warning in unresolved


def test_selection_story_rejects_a_tampered_gate_artifact_hash(tmp_path: Path) -> None:
    source = ROOT / "results/evidence/task2/g1_family_screen/manifest.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["artifacts"]["leaderboard"]["sha256"] = "0" * 64
    tampered = tmp_path / "g1_manifest.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="leaderboard artifact hash"):
        build_task2_selection_story_evidence(
            g1_manifest_path=tampered,
            project_root=ROOT,
            evidence_directory=tmp_path / "selection_story",
        )
