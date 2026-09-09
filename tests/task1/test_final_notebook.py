"""Exercise the notebook's stage boundaries without running any real model."""
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/02_task1_final_eval.ipynb"


def cell(tag):
    matches = [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells
               if tag in c.metadata.get("tags", [])]
    assert len(matches) == 1, f"Missing unique stage cell: {tag}"
    return matches[0]


@pytest.fixture
def stage_env(tmp_path):
    calls = []
    evidence_dir = tmp_path / "results/evidence/task1/final_evaluation"
    evidence_dir.mkdir(parents=True)
    model = tmp_path / "models/task1_article_type.manifest.json"
    model.parent.mkdir()

    def record(name, path=None):
        def run(*args, **kwargs):
            calls.append(name)
            if path:
                path.write_text("{}")
            return {"verified": True}
        return run

    env = dict(PROJECT_ROOT=tmp_path, EVIDENCE_DIR=evidence_dir,
               RUN_MISSING_STAGES=False, UNLOCK_HOLDOUT_SCORING=False,
               torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
               load_verified_task1_refit_manifest=record("verify-refit"),
               run_task1_refit=record("train", model),
               predict_holdout=record("predict", evidence_dir / "prediction_receipt.json"),
               score_holdout=record("score", evidence_dir / "evaluation_manifest.json"),
               predict_test=record("export", evidence_dir / "test_prediction_receipt.json"),
               audit_final_evaluation=record("audit"))
    return env, calls, model


def completed(env, model):
    model.write_text("{}")
    for name in ("prediction_receipt.json", "evaluation_manifest.json", "test_prediction_receipt.json"):
        (env["EVIDENCE_DIR"] / name).write_text("{}")


def test_default_controls_are_replay_only():
    env = {}
    exec(cell("task1-controls"), env)
    assert env["RUN_MISSING_STAGES"] is False
    assert env["UNLOCK_HOLDOUT_SCORING"] is False


def test_completed_replay_never_runs_a_model_or_unlocks_labels_without_cuda(stage_env):
    env, calls, model = stage_env
    completed(env, model)
    for name in ("refit", "predict", "score", "export"):
        exec(cell(f"task1-stage-{name}"), env)
    assert "verify-refit" in calls and "audit" in calls
    assert not set(calls) & {"train", "predict", "score", "export"}


@pytest.mark.parametrize("stage,missing", [("refit", "model"), ("predict", "prediction_receipt.json"), ("score", "evaluation_manifest.json")])
def test_missing_replay_stage_stops_without_work(stage_env, stage, missing):
    env, calls, model = stage_env
    completed(env, model)
    (model if missing == "model" else env["EVIDENCE_DIR"] / missing).unlink()
    with pytest.raises(FileNotFoundError, match="RUN_MISSING_STAGES"):
        exec(cell(f"task1-stage-{stage}"), env)
    assert not set(calls) & {"train", "predict", "score", "export"}


def test_fresh_execution_follows_one_way_order(stage_env):
    env, calls, model = stage_env
    env.update(RUN_MISSING_STAGES=True, UNLOCK_HOLDOUT_SCORING=True)
    env["torch"].cuda.is_available = lambda: True
    for name in ("refit", "predict", "score", "export"):
        exec(cell(f"task1-stage-{name}"), env)
    assert [c for c in calls if c in {"train", "predict", "score", "export"}] == ["train", "predict", "score", "export"]


def test_missing_scoring_needs_explicit_unlock(stage_env):
    env, calls, _ = stage_env
    env["RUN_MISSING_STAGES"] = True
    with pytest.raises(RuntimeError, match="UNLOCK_HOLDOUT_SCORING"):
        exec(cell("task1-stage-score"), env)
    assert "score" not in calls


def test_missing_export_is_pending_in_replay(stage_env):
    env, calls, _ = stage_env
    exec(cell("task1-stage-export"), env)
    assert "export" not in calls


def test_execution_reuses_completed_stages_without_new_unlock(stage_env):
    env, calls, model = stage_env
    completed(env, model)
    env["RUN_MISSING_STAGES"] = True
    for name in ("refit", "predict", "score", "export"):
        exec(cell(f"task1-stage-{name}"), env)
    assert not set(calls) & {"train", "predict", "score", "export"}


def test_stage_failure_is_not_swallowed_or_retried(stage_env):
    env, calls, _ = stage_env
    env["RUN_MISSING_STAGES"] = True
    marker = env["EVIDENCE_DIR"] / "predict_holdout_attempt.json"
    marker.write_text("partial")

    def failed(root):
        calls.append("predict")
        raise RuntimeError("partial attempt requires inspection")

    env["predict_holdout"] = failed
    with pytest.raises(RuntimeError, match="partial"):
        exec(cell("task1-stage-predict"), env)
    assert calls == ["predict"]
    assert marker.read_text() == "partial"


def test_combined_notebook_replaces_the_old_runner():
    assert not (ROOT / "notebooks/02_task1_final_run.ipynb").exists()
    nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))
