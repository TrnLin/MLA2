"""Audit saved G-WD1 results and draw comparisons without training or inference."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from fashion.data import load_splits
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_weight_decay import (
    _verify_baseline_controls,
    check_weight_decay_sources,
    evaluate_weight_decay_screen,
    weight_decay_config,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REPORT = Path(__file__).resolve().parent


def compare(actual, saved, path="decision"):
    if isinstance(actual, dict):
        assert actual.keys() == saved.keys(), path
        for key in actual:
            compare(actual[key], saved[key], path + "." + key)
    elif isinstance(actual, list):
        assert len(actual) == len(saved), path
        for i, (left, right) in enumerate(zip(actual, saved, strict=True)):
            compare(left, right, f"{path}[{i}]")
    elif isinstance(actual, float):
        assert np.isclose(actual, saved, atol=1e-12, rtol=0), (path, actual, saved)
    else:
        assert actual == saved, (path, actual, saved)


sources, classes, spec = check_weight_decay_sources(
    g2_directory=ROOT / "reports/task3/g2_gate_audit_20260905/drive",
    e6_directory=ROOT / "results/evidence/task3/experiments/t3_gender_e6_gem_p3/gender",
    source_registry_path=REPORT / "runs.csv",
    root=ROOT,
)
config = weight_decay_config(spec, fold=0, device_name="cuda")
source_audit = json.loads((REPORT / "gender/source_audit.json").read_text())
assert source_audit["spec"] == spec.to_dict()
assert source_audit["baseline_controls"] == config.to_dict()
assert source_audit["folds"] == [0, 4]
assert source_audit["source_sha256"] == {
    run["run_id"]: run["sha256"] for group in sources.values() for run in group.values()
}
registry = pd.read_csv(REPORT / "runs.csv", keep_default_na=False)
splits = load_splits(ROOT / "data/processed/splits.csv")
child, paths = {}, {}
for directory in sorted((REPORT / "gender").glob("t3_gender_weight_decay_001_*")):
    if not directory.is_dir():
        continue
    run = inspect_gender_run(
        directory, registry=registry, splits=splits, classes=classes, root=ROOT
    )
    fold = run["fold"]
    assert fold in (0, 4) and fold not in child
    assert run["config"]["child_experiment"] == spec.to_dict()
    assert run["config"]["parent_run_id"] == sources["G2"][fold]["run_id"]
    _verify_baseline_controls(run["config"], config)
    child[fold], paths[fold] = run, directory
assert set(child) == {0, 4}
result = evaluate_weight_decay_screen(child, sources, classes)
result["registry_and_artifact_integrity"] = True
result["run_ids"] = {str(f): child[f]["run_id"] for f in (0, 4)}
compare(result, json.loads((REPORT / "gender/screen_decision.json").read_text()))
pooled = pd.concat([child[f]["predictions"] for f in (0, 4)]).sort_values("id")
saved = pd.read_csv(REPORT / "gender/oof_predictions.csv", keep_default_na=False)
pd.testing.assert_frame_equal(
    pooled.reset_index(drop=True), saved.sort_values("id").reset_index(drop=True),
    check_exact=False, atol=1e-12, rtol=0,
)
pd.testing.assert_frame_equal(
    pd.DataFrame(result["folds"]), pd.read_csv(REPORT / "gender/clean_gap_comparison.csv"),
    check_exact=False, atol=1e-12, rtol=0,
)
result["review"] = {
    "verified_source_runs": 10, "verified_child_runs": 2,
    "checkpoint_inference_repeated_locally": False,
}
(REPORT / "verified_decision.json").write_text(json.dumps(result, indent=2) + "\n")
pd.DataFrame(result["checks"]).to_csv(REPORT / "verified_gates.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
for i, fold in enumerate((0, 4)):
    history = pd.read_csv(paths[fold] / "history.csv")
    axes[i].plot(history.epoch, history.train_macro_f1, "--", label="Training (augmented)")
    axes[i].plot(history.epoch, history.validation_macro_f1, label="Validation")
    axes[i].set(title=f"G-WD1: fold {fold}", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1.02))
    axes[i].legend(fontsize=8)
rows = pd.DataFrame(result["folds"])
for i, (key, label) in enumerate((("parent_gap", "G2"), ("candidate_gap", "G-WD1"))):
    bars = axes[2].bar(np.arange(2) + (i - .5) * .32, rows[key], .32, label=label)
    axes[2].bar_label(bars, fmt="%.4f", padding=3)
axes[2].set(xticks=range(2), xticklabels=["Fold 0", "Fold 4"], ylim=(0, .35),
            title="Final clean training–validation gap", ylabel="F1 gap")
axes[2].legend()
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(REPORT / "overfitting_review.png", dpi=150)
print(json.dumps({
    "status": result["status"], "candidate_f1": result["candidate"]["macro_f1"],
    "parent_f1": result["comparison"]["macro_f1"], "gap": result["mean_clean_gap"],
    "parent_gap": result["parent_mean_clean_gap"], "bootstrap": result["bootstrap"],
    "failed_checks": [row for row in result["checks"] if row["status"] != "pass"],
    "folds": result["folds"],
}, indent=2))
