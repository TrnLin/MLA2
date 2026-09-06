from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from fashion.config import ROOT

LAUNCHER = ROOT / "scripts/predict_task2_season.py"


def _load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("predict_task2_season_script", LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load prediction launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prediction_launcher_has_one_windows_safe_main_guard() -> None:
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
    assert "load_season_bundle" in source
    assert "predict_manifest" in source
    assert "prediction_manifest.csv" not in source


def test_prediction_launcher_help_states_image_only_and_holdout_boundaries() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "image-only Season inference" in completed.stdout
    assert "never opens holdout labels" in completed.stdout
    assert "{auto,cpu,cuda}" in completed.stdout


def test_prediction_launcher_emits_ordered_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_launcher()
    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in paths:
        path.write_bytes(b"fixture")
    fake_bundle = object()
    monkeypatch.setattr(module, "load_season_bundle", lambda *, device: fake_bundle)

    def fake_predict(bundle, image_paths):
        assert bundle is fake_bundle
        return tuple(
            SimpleNamespace(
                to_dict=lambda path=path: {
                    "image_path": path.as_posix(),
                    "predicted_label": "Summer",
                    "probabilities": {
                        "Fall": 0.1,
                        "Spring": 0.1,
                        "Summer": 0.7,
                        "Winter": 0.1,
                    },
                }
            )
            for path in image_paths
        )

    monkeypatch.setattr(module, "predict_manifest", fake_predict)

    assert module.main([str(path) for path in paths] + ["--device", "cpu"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert [Path(item["image_path"]).name for item in output] == ["first.png", "second.png"]
    assert all(item["predicted_label"] == "Summer" for item in output)


def test_prediction_launcher_returns_json_error(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    module = _load_launcher()
    image = tmp_path / "broken.png"
    monkeypatch.setattr(module, "load_season_bundle", lambda *, device: object())
    monkeypatch.setattr(
        module,
        "predict_manifest",
        lambda bundle, paths: (_ for _ in ()).throw(ValueError("invalid image")),
    )

    assert module.main([str(image)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {"error": "invalid image"}
