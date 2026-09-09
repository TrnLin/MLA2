"""Exercise real stage files with synthetic images; never use project holdout labels."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fashion.data.hashing import compute_sha256
from fashion.task1 import final_evaluation_runner as runner


@pytest.fixture
def final_project(tmp_path, monkeypatch):
    root = tmp_path
    labels = [f"class_{i}" for i in range(124)]
    for directory in [
        "data/processed",
        "data/raw/teacher/train",
        "data/raw/teacher/test",
        "models",
        "configs/task1",
        "src/fashion",
    ]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "src/fashion/example.py").write_text("# frozen source\n")
    (root / "models/task1_article_type.pt").write_bytes(b"synthetic checkpoint")
    (root / "results").mkdir()
    pd.DataFrame([{"run_id": "fixture-refit", "status": "completed", "task": "task1"}]).to_csv(
        root / "results/runs.csv", index=False
    )
    history = root / "results/training_history.csv"
    pd.DataFrame([{"epoch": 20, "loss": 0.5}]).to_csv(history, index=False)
    (root / "models/task1_article_type.manifest.json").write_text(
        json.dumps(
            {
                "run_id": "fixture-refit",
                "history": {
                    "path": "results/training_history.csv",
                    "sha256": compute_sha256(history),
                },
            }
        )
    )
    rows = []
    for i in range(1, 10):
        path = root / f"image_{i}.png"
        Image.new("RGB", (6, 8), (i * 20, 20, 50)).save(path)
        partition = "development" if i <= 5 else "holdout" if i <= 7 else "quarantine"
        row = dict(
            id=i,
            path=path.name,
            sha256=compute_sha256(path),
            partition=partition,
            cv_fold=i - 1 if i <= 5 else "",
            product_family_group=f"f{i}",
            product_name_key=f"n{i}",
            duplicate_group=f"d{i}",
            mode="RGB",
            is_cross_role_exact_duplicate=False,
            is_cross_role_near_duplicate=False,
            has_conflicting_target_labels=False,
            conflicting_targets="",
            quarantine_reason="fixture" if partition == "quarantine" else "",
        )
        for target in ["articleType", "gender", "season", "usage"]:
            row[target] = labels[i % 2] if i <= 5 else ""
            row[f"has_{target}_label"] = i <= 5
        rows.append(row)
    pd.DataFrame(rows[:8]).to_csv(root / "data/processed/splits.csv", index=False)
    pd.DataFrame([rows[8]])[["id", "path", "sha256"]].to_csv(
        root / "data/processed/prediction_manifest.csv", index=False
    )
    pd.DataFrame(
        [
            dict(id=i, **{t: labels[i % 2] for t in ["articleType", "gender", "season", "usage"]})
            for i in range(1, 9)
        ]
    ).to_csv(root / "data/raw/teacher/train/styles_train.csv", index=False)
    pd.DataFrame([dict(id=9, gender="", articleType="", season="", usage="")]).to_csv(
        root / "data/raw/teacher/test/styles_prediction.csv", index=False
    )
    mapping = {
        "articleType": {
            "num_classes": 124,
            "classes": labels,
            "label_to_index": dict(zip(labels, range(124))),
            "source_scope": "development",
        }
    }
    (root / "data/processed/label_maps.json").write_text(json.dumps(mapping))
    config = json.loads((runner.ROOT / "configs/task1/final_evaluation.json").read_text())
    config["expected_rows"] = dict(development=5, holdout=2, quarantine=1, official_test=1)
    config["bootstrap"]["replicates"] = 20
    config["cost"]["repeats"] = 2
    (root / "configs/task1/final_evaluation.json").write_text(json.dumps(config))

    def predict_paths(paths, *, batch_size=128):
        p = np.full((len(paths), 124), 0.01 / 123)
        p[:, 0] = 0.99
        return p

    monkeypatch.setattr(
        runner,
        "_load_bundle",
        lambda root: SimpleNamespace(
            class_names=tuple(labels), manifest={}, predict_paths=predict_paths
        ),
    )
    return root


def test_stage_order_one_shot_receipts_and_read_only_replay(final_project, monkeypatch):
    root = final_project
    with pytest.raises((ValueError, FileNotFoundError), match="score|evaluation|predict-holdout"):
        runner.predict_test(root)
    # Label-free prediction must work even when the raw labels are absent.
    raw = root / "data/raw/teacher/train/styles_train.csv"
    raw.rename(raw.with_suffix(".hidden"))
    runner.predict_holdout(root)
    with pytest.raises(ValueError, match="unlock"):
        runner.score_holdout(root)
    raw.with_suffix(".hidden").rename(raw)
    runner.score_holdout(root, evaluation_unlocked=True)
    runner.predict_test(root)
    before = {str(p): compute_sha256(p) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        runner, "_load_bundle", lambda *_: pytest.fail("audit must not load a model")
    )
    raw.rename(raw.with_suffix(".hidden"))
    evidence = runner.audit_final_evaluation(root)
    assert evidence["test_prediction_receipt"]["rows"] == 1
    raw.with_suffix(".hidden").rename(raw)
    assert before == {str(p): compute_sha256(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="already|partial"):
        runner.predict_holdout(root)


def test_tampered_prediction_blocks_label_access(final_project, monkeypatch):
    root = final_project
    runner.predict_holdout(root)
    path = root / runner.EVIDENCE / "holdout_predictions.csv"
    path.write_text(path.read_text().replace("0.99", "0.98"))
    monkeypatch.setattr(
        runner,
        "load_splits_for_final_evaluation",
        lambda *a, **kw: pytest.fail("labels accessed before hashes checked"),
    )
    with pytest.raises(RuntimeError, match="SHA-256"):
        runner.score_holdout(root, evaluation_unlocked=True)


def test_changed_checkpoint_blocks_official_export(final_project):
    root = final_project
    runner.predict_holdout(root)
    runner.score_holdout(root, evaluation_unlocked=True)
    (root / "models/task1_article_type.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256"):
        runner.predict_test(root)
    assert not (root / "results/article_type_test_predictions.csv").exists()


def test_notebook_replays_twice_without_writes_or_inference(final_project, monkeypatch):
    root = final_project
    runner.predict_holdout(root)
    runner.score_holdout(root, evaluation_unlocked=True)
    runner.predict_test(root)
    notebook = json.loads(
        (runner.ROOT / "notebooks/02_task1_final_eval.ipynb").read_text(encoding="utf-8")
    )
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert any("audit_final_evaluation" in cell for cell in code)
    before = {str(p): compute_sha256(p) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setenv("FASHION_PROJECT_ROOT", str(root))
    monkeypatch.setattr(runner, "_load_bundle", lambda *_: pytest.fail("notebook loaded model"))
    monkeypatch.setattr(
        runner,
        "load_splits_for_final_evaluation",
        lambda *a, **kw: pytest.fail("notebook opened protected labels"),
    )
    for _ in range(2):
        namespace = {"display": lambda *a, **kw: None}
        for cell in code:
            exec(compile(cell, "notebook06", "exec"), namespace)
    assert before == {str(p): compute_sha256(p) for p in root.rglob("*") if p.is_file()}


def test_replay_checks_score_math_and_training_record(final_project):
    root = final_project
    runner.predict_holdout(root)
    runner.score_holdout(root, evaluation_unlocked=True)
    metrics_path = root / runner.EVIDENCE / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics["macro_f1"] = 1.0
    metrics.to_csv(metrics_path, index=False)
    manifest_path = root / runner.EVIDENCE / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tables"]["metrics"]["sha256"] = compute_sha256(metrics_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metrics|score"):
        runner.audit_final_evaluation(root)
    pd.DataFrame([{"run_id": "someone-else", "status": "completed"}]).to_csv(
        root / "results/runs.csv", index=False
    )
    with pytest.raises(ValueError, match="registry|training"):
        runner.audit_final_evaluation(root)
