from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

from fashion.config import ROOT

LAUNCHER = ROOT / "scripts/build_task2_handoff.py"


def _load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_task2_handoff_script", LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Task 2 handoff launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handoff_launcher_has_one_windows_safe_main_guard() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "__main__"
            for comparator in node.test.comparators
        )
    ]

    assert len(guards) == 1
    assert "evaluation_unlocked" not in source
    assert "styles_prediction.csv" not in source


def test_handoff_launcher_help_keeps_final_evaluation_locked() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "keeps Notebook 06 and holdout evaluation locked" in completed.stdout
    assert "{cpu,cuda}" in completed.stdout


def test_handoff_launcher_emits_verified_locked_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_launcher()
    smoke_image = tmp_path / "1163.jpg"
    smoke_image.write_bytes(b"fixture")
    audit = pd.DataFrame({"status": ["PASS", "PASS"]})
    manifest_path = ROOT / "results/evidence/task2/final_handoff/manifest.json"
    manifest = {
        "status": "ready_for_group_freeze",
        "task2_component_ready": True,
        "group_freeze_verified": False,
        "notebook_06_unlocked": False,
        "holdout_opened": False,
        "run_id": "refit-run",
        "next_gate": "whole-group freeze before Notebook 06",
    }
    smoke = {
        "product_id": "1163",
        "predicted_label": "Summer",
        "bundle_sha256": "b" * 64,
    }
    monkeypatch.setattr(module, "audit_task2_artifacts", lambda: audit)
    monkeypatch.setattr(module, "load_season_bundle", lambda *, device: object())
    monkeypatch.setattr(
        module,
        "predict_season",
        lambda bundle, path: SimpleNamespace(predicted_label="Summer"),
    )
    monkeypatch.setattr(
        module,
        "build_task2_handoff_evidence",
        lambda checked, prediction: (manifest, manifest_path),
    )
    monkeypatch.setattr(
        module,
        "load_verified_task2_handoff",
        lambda path: (manifest, path, audit, smoke),
    )

    assert module.main(["--smoke-image", str(smoke_image), "--device", "cpu"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["task2_component_ready"] is True
    assert summary["notebook_06_unlocked"] is False
    assert summary["holdout_opened"] is False
    assert summary["audit_checks_passed"] == summary["audit_checks_total"] == 2


def test_handoff_launcher_returns_json_error(monkeypatch, capsys) -> None:
    module = _load_launcher()
    monkeypatch.setattr(
        module,
        "audit_task2_artifacts",
        lambda: (_ for _ in ()).throw(ValueError("handoff mismatch")),
    )

    assert module.main([]) == 2
    assert json.loads(capsys.readouterr().err) == {"error": "handoff mismatch"}
