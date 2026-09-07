"""Five fresh scratch folds of the passing fixed epoch-25 SAM recipe."""

import hashlib
import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_samples
from fashion.data.gender_name_truth import VARIANT_RELATIVE_PATH, load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.registry import REGISTRY_COLUMNS
from fashion.train.task3_dataset_v2 import _write_json
from fashion.train.task3_decisions import oof_metrics, robustness_changes, validate_oof
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_dropout_darkening import (
    _save_source_audit,
    _verify_training_evidence,
)
from fashion.train.task3_gender_ieee import (
    POLICY,
    evaluate_gender_ieee,
    evaluation_identity,
    load_ieee_evaluation,
)
from fashion.train.task3_gender_name_truth import label_contract, write_original_label_diagnostic
from fashion.train.task3_gender_narrow import MEMORY_LIMIT, RUNTIME, require_narrow_prerequisites
from fashion.train.task3_gender_sam25 import (
    COSINE_T_MAX,
    EPOCHS,
    SAM25Spec,
    check_gender_sam25_sources,
    verify_sam25_evidence,
)
from fashion.train.task3_gender_sam25 import _source_identity as screen_source_identity

NAME = "gender_name_truth_mixup_alpha020_sam005_epoch25_cv"
FOLDS = (0, 1, 2, 3, 4)
RULE_VERSION = "sam25_five_fold_fixed_recipe_v1"
SCREEN_COMMIT = "5f0789506239944f2e836348b00bdc5e34d5658d"
SCREEN_DECISION_SHA256 = "4d495669701cb8e5fe05dbb6dfd8e6e66c7232f0d0dac6d97df729e28f533ca4"
SCREEN_RUN_IDS = {
    0: "t3_gender_name_truth_mixup_alpha020_sam005_epoch25_gender_smallcnngem3_f0_s2753_"
    "5fcf4e0e0035_20260906T101038Z1eda8d",
    4: "t3_gender_name_truth_mixup_alpha020_sam005_epoch25_gender_smallcnngem3_f4_s2753_"
    "5fcf4e0e0035_20260906T102136Z29db40",
}


@dataclass(frozen=True)
class SAM25CVSpec(SAM25Spec):
    comparison_run_ids: tuple[str, ...]

    def __getattr__(self, key):
        overrides = {
            "name": NAME,
            "experiment_id": f"t3_{NAME}",
            "hypothesis_id": f"t3_{NAME}",
            "artifact_dir": f"experiments/t3_{NAME}",
            "run_prefix": f"t3_{NAME}",
            "changed_factor": "five_fold_evaluation_of_fixed_sam25_recipe",
            "parent_artifact_dir": "experiments/t3_gender_v2_g2_translation",
            "parent_run_ids": self.comparison_run_ids,
        }
        return overrides[key] if key in overrides else super().__getattr__(key)

    def parent_run_id_for_fold(self, fold):
        if fold not in FOLDS or len(self.comparison_run_ids) != 5:
            raise ValueError("SAM25 CV requires five same-fold G2 comparison references")
        return self.comparison_run_ids[fold]

    def to_dict(self):
        payload = super().to_dict()
        payload.pop("improvement_rules")
        payload.update(
            parent_run_ids=list(self.comparison_run_ids),
            parent_folds=list(FOLDS),
            parent_role="same_fold_G2_comparison_only_no_weights_loaded",
            recipe_source_run_ids={str(f): r for f, r in SCREEN_RUN_IDS.items()},
            recipe_source_commit=SCREEN_COMMIT,
            screen_rule_version=RULE_VERSION,
            execution_scope="five_fresh_scratch_folds_reuse_only_completed_cv_runs",
            independent_blind_test=False,
        )
        return payload


def sam25_cv_spec(comparison_run_ids, *, root=ROOT):
    ids = tuple(comparison_run_ids)
    if len(ids) != 5 or len(set(ids)) != 5 or not all(isinstance(r, str) and r for r in ids):
        raise ValueError("SAM25 CV needs five distinct G2 references in fold order")
    return SAM25CVSpec(json.dumps(label_contract(root), sort_keys=True), ids)


def sam25_cv_config(spec, *, fold, device_name, root=ROOT):
    if fold not in FOLDS or device_name != "cuda":
        raise ValueError("SAM25 CV requires CUDA and canonical folds 0,1,2,3,4")
    if (
        not isinstance(spec, SAM25CVSpec)
        or spec.to_dict() != sam25_cv_spec(spec.comparison_run_ids, root=root).to_dict()
    ):
        raise ValueError("SAM25 CV differs from the frozen recipe")
    return replace(Task3BaselineConfig(target="gender"), epochs=EPOCHS)


def training_splits(spec, *, root=ROOT):
    sam25_cv_config(spec, fold=0, device_name="cuda", root=root)
    return load_gender_name_truth_variant(root)


def _historical_hash(root, path):
    """Read immutable screen code; new CV routing must not rewrite the old audit."""
    content = subprocess.check_output(["git", "show", f"{SCREEN_COMMIT}:{path}"], cwd=root)
    return hashlib.sha256(content).hexdigest()


def _screen_evaluation(run, *, directory, splits, classes, contract, root):
    identity = evaluation_identity(run, root=root, label_variant=contract)
    identity["runtime"] = dict(RUNTIME)
    for path in identity["dependencies"]:
        if path.startswith("src/"):
            identity["dependencies"][path] = _historical_hash(root, path)
    return load_ieee_evaluation(
        directory / "comparison_name_truth_ieee" / run["run_id"],
        run=run,
        splits=splits,
        classes=classes,
        identity=identity,
    )


def check_gender_sam25_cv_sources(*, sam25_directory, root=ROOT, **paths):
    """Bind the full CV run to the exact reviewed passing screen and its parents."""
    root, directory = Path(root), Path(sam25_directory)
    decision_path = directory / "screen_decision.json"
    if compute_sha256(decision_path) != SCREEN_DECISION_SHA256:
        raise ValueError("SAM25 CV requires the exact reviewed passing 04aj decision")
    decision = json.loads(decision_path.read_text())
    if decision["status"] != "pass" or decision["run_ids"] != {
        str(f): r for f, r in SCREEN_RUN_IDS.items()
    }:
        raise ValueError("SAM25 screen status or run IDs changed")
    sources, classes, screen_spec, evidence = check_gender_sam25_sources(root=root, **paths)
    previous = json.loads((directory / "source_audit.json").read_text())["identity"]
    expected = screen_source_identity(sources, screen_spec, evidence, paths, root=root)
    expected["implementation_sha256"] = {
        name: _historical_hash(root, f"src/fashion/train/{name}")
        for name in expected["implementation_sha256"]
    }
    if previous != expected:
        raise ValueError("SAM25 screen source audit differs from its recorded code or data")
    splits = load_gender_name_truth_variant(root)
    registry = pd.read_csv(paths["source_registry_path"], keep_default_na=False)
    screened = {}
    for fold, run_id in SCREEN_RUN_IDS.items():
        run = inspect_gender_run(
            directory / run_id,
            registry=registry,
            splits=splits,
            classes=classes,
            root=root,
            expected_epochs=EPOCHS,
        )
        _verify_training_evidence(
            run,
            screen_spec,
            sources["MixUp20"][fold]["run_id"],
            evidence,
            audit_sha256=compute_sha256(directory / "source_audit.json"),
            expected_epochs=EPOCHS,
        )
        verify_sam25_evidence(run, fold=fold, splits=splits, directory=directory / run_id)
        run["directory"] = str(directory / run_id)
        screened[fold] = _screen_evaluation(
            run,
            directory=directory,
            splits=splits,
            classes=classes,
            contract=screen_spec.to_dict()["gender_label_variant"],
            root=root,
        )
    pooled = oof_metrics(pd.concat([r["predictions"] for r in screened.values()]), classes)
    for key in ("macro_f1", "nll", "ece_15"):
        if not np.isclose(pooled[key], decision["candidate"][key], rtol=0, atol=1e-10):
            raise ValueError("Reviewed SAM25 scores differ from saved IEEE predictions")
    if pooled["confusion_matrix"] != decision["candidate"]["confusion_matrix"]:
        raise ValueError("Reviewed SAM25 confusion matrix changed")
    sources["SAM25Screen"] = screened
    spec = sam25_cv_spec([sources["G2"][f]["run_id"] for f in FOLDS], root=root)
    return sources, classes, spec, evidence


def _source_identity(sources, spec, evidence, paths, *, root):
    root = Path(root)
    dependencies = {
        "task3_gender_sam25_cv.py",
        "task3_gender_sam25.py",
        "task3_gender_sam.py",
        "task3_baseline.py",
        "task3_g2_audit.py",
        "task3_gender_dropout_darkening.py",
        "task3_gender_name_truth.py",
        "task3_gender_ieee.py",
        "task3_gender_diagnostic.py",
        "task3_gender_precision.py",
        "task3_gender_mixup.py",
        "sam.py",
        "mixup.py",
        "augmentation.py",
        "data.py",
        "config.py",
        "model.py",
        "metrics.py",
        "registry.py",
        "task3_decisions.py",
    }
    return {
        "spec": spec.to_dict(),
        "baseline_controls": sam25_cv_config(spec, fold=0, device_name="cuda", root=root).to_dict(),
        "cosine_t_max": COSINE_T_MAX,
        "paths": {k: str(Path(v).resolve()) for k, v in paths.items()},
        "source_sha256": {
            r["run_id"]: r["sha256"] for group in sources.values() for r in group.values()
        },
        "screen_evaluation_sha256": {
            r["run_id"]: compute_sha256(
                Path(r["evaluation_directory"]) / "evaluation_manifest.json"
            )
            for r in sources["SAM25Screen"].values()
        },
        "screen_decision_sha256": compute_sha256(
            Path(paths["sam25_directory"]) / "screen_decision.json"
        ),
        "screen_audit_sha256": compute_sha256(Path(paths["sam25_directory"]) / "source_audit.json"),
        "precision_evidence_sha256": evidence["artifact_sha256"],
        "implementation_sha256": {
            name: compute_sha256(root / "src/fashion/train" / name) for name in sorted(dependencies)
        },
        "data_sha256": {
            name: compute_sha256(root / name)
            for name in (
                "data/processed/splits.csv",
                "data/processed/label_maps.json",
                "src/fashion/data/gender_name_truth.py",
                "src/fashion/data/dataset.py",
                "src/fashion/data/images.py",
            )
        },
        "folds": list(FOLDS),
        "comparison_precision": POLICY,
    }


def require_sam25_cv_prerequisites(
    path, *, spec, fold, parent_run_directory=None, root=ROOT, device_name="cuda"
):
    sam25_cv_config(spec, fold=fold, device_name=device_name, root=root)
    if path is None:
        raise ValueError("SAM25 CV source audit is required before training")
    identity = json.loads(Path(path).read_text())["identity"]
    paths = identity["paths"]
    precision = require_narrow_prerequisites(paths["precision_directory"], root=root)
    sources, _, expected, evidence = check_gender_sam25_cv_sources(root=root, **paths)
    if (
        spec.to_dict() != expected.to_dict()
        or precision["artifact_sha256"] != evidence["artifact_sha256"]
        or identity != _source_identity(sources, spec, evidence, paths, root=root)
    ):
        raise ValueError("SAM25 CV source evidence changed")
    parent = Path(sources["G2"][fold]["directory"])
    if (
        parent_run_directory is not None
        and Path(parent_run_directory).resolve() != parent.resolve()
    ):
        raise ValueError("SAM25 CV requires its same-fold G2 comparison reference")
    return {
        "precision": precision,
        "parent_directory": parent,
        "prerequisite_sha256": compute_sha256(path),
    }


def _completed_fold(spec, fold, *, destination, registry):
    """Reuse one completed registered CV fit; failed or interrupted fits stay recorded."""
    selected = registry.loc[
        registry.experiment_id.eq(spec.experiment_id)
        & pd.to_numeric(registry.validation_fold, errors="coerce").eq(fold)
        & registry.status.eq("complete")
    ]
    if len(selected) > 1:
        raise ValueError(f"Multiple completed SAM25 CV runs for fold {fold}; choose one explicitly")
    return destination / str(selected.iloc[0].run_id) if len(selected) else None


def summarize_cv(runs, *, splits, classes):
    if set(runs) != set(FOLDS):
        raise ValueError("SAM25 CV summary requires all five folds")
    expected = get_samples(splits, partition="development", target="gender")
    predictions = validate_oof(
        pd.concat([runs[f]["predictions"] for f in FOLDS], ignore_index=True),
        expected,
        target="gender",
        classes=classes,
        run_ids_by_fold={f: runs[f]["run_id"] for f in FOLDS},
    )
    folds = []
    for fold in FOLDS:
        m = runs[fold]["metrics"]
        if m.get("comparison_precision") != POLICY:
            raise ValueError("Five-fold reporting requires matched IEEE evaluation")
        folds.append(
            {
                "fold": fold,
                "run_id": runs[fold]["run_id"],
                "train_f1": m["final_train_eval_macro_f1"],
                "validation_f1": m["macro_f1"],
                "gap": m["final_train_validation_macro_f1_gap"],
                "train_seconds": m["train_seconds"],
                "peak_memory_bytes": m["peak_memory_bytes"],
            }
        )
    scopes = {}
    for label, scope in (
        ("all_five", FOLDS),
        ("screen_folds", (0, 4)),
        ("additional_folds", (1, 2, 3)),
    ):
        subset = predictions.loc[predictions.cv_fold.isin(scope)]
        scopes[label] = {
            "folds": list(scope),
            "metrics": oof_metrics(subset, classes),
            "mean_clean_gap": float(np.mean([r["gap"] for r in folds if r["fold"] in scope])),
        }
    robust = pd.concat([runs[f]["robustness"] for f in FOLDS], ignore_index=True)
    changes = robustness_changes(
        robust,
        clean_by_fold={f: runs[f]["metrics"]["macro_f1"] for f in FOLDS},
        run_ids_by_fold={f: runs[f]["run_id"] for f in FOLDS},
    )
    report = {
        "status": "complete_for_review",
        "experiment_id": f"t3_{NAME}",
        "folds": folds,
        "scopes": scopes,
        "mean_induced_corruption_change": changes.to_dict(),
        "independent_test_evidence": False,
        "evaluation_note": (
            "Each development image is scored only by its held-out fold model. "
            "Five-model ensemble performance has not been measured."
        ),
        "selection_note": (
            "Folds 0/4 selected this recipe; folds 1/2/3 had prior use by earlier recipes. "
            "No final holdout or teacher-test evaluation here."
        ),
    }
    return report, predictions, robust


def _require_resources(metrics):
    memory = metrics["peak_memory_bytes"]
    if not np.isfinite(memory) or not 0 < memory < MEMORY_LIMIT:
        raise ValueError("SAM25 CV exceeded the 3 GB memory limit; later folds were not started")
    if metrics["parameter_count"] != 390_181:
        raise ValueError("SAM25 CV architecture changed")


def plot_cv_summary(report, path):
    """Save the per-fold fit gaps and row-normalized pooled confusion matrix."""
    import matplotlib.pyplot as plt

    table = pd.DataFrame(report["folds"])
    metrics = report["scopes"]["all_five"]["metrics"]
    classes = [r["class_name"] for r in metrics["per_class"]]
    matrix = np.asarray(metrics["confusion_matrix"])
    fractions = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(5)
    axes[0].bar(x - 0.18, table.train_f1 * 100, 0.36, label="Clean training", color="#718796")
    axes[0].bar(x + 0.18, table.validation_f1 * 100, 0.36, label="Validation", color="#337b8c")
    axes[0].set(title="F1 per fold (%)", xticks=x, xlabel="Validation fold", ylim=(0, 105))
    axes[0].legend(frameon=False)
    bars = axes[1].bar(x, table.gap * 100, color="#c78c53")
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].margins(y=0.2)
    axes[1].set(title="Clean train–validation gap (points)", xticks=x, xlabel="Validation fold")
    axes[2].imshow(fractions, vmin=0, vmax=1, cmap="Blues")
    for row in range(5):
        for col in range(5):
            axes[2].text(
                col,
                row,
                f"{100 * fractions[row, col]:.0f}%",
                ha="center",
                va="center",
                color="white" if fractions[row, col] > 0.5 else "black",
            )
    axes[2].set(
        title="Pooled OOF: share of each true class",
        xticks=x,
        yticks=x,
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="True",
    )
    for ax in axes[:2]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fixed SAM25: five-fold development evaluation", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_model_manifest(runs, *, destination, classes, spec):
    folds = []
    for fold in FOLDS:
        run = runs[fold]
        directory = Path(run["directory"])
        names = ("final_epoch.pt", "normalization.json", "config.json")
        folds.append(
            {
                "fold": fold,
                "run_id": run["run_id"],
                "files": {
                    name: {
                        "path": str((directory / name).relative_to(destination)),
                        "sha256": compute_sha256(directory / name),
                    }
                    for name in names
                },
            }
        )
    manifest = {
        "version": "sam25_cv_model_set_v1",
        "class_names": list(classes),
        "recipe": spec.to_dict(),
        "folds": folds,
        "checkpoint_epoch": EPOCHS,
        "inference": (
            "For new images, use each fold's own normalization and average the five "
            "softmax probability vectors; take argmax in class_names order."
        ),
        "evaluation": "OOF uses one held-out model per image, never the five-model average.",
        "all_development_refit": False,
    }
    _write_json(manifest, destination / "model_manifest.json")


def run_gender_sam25_cv(
    *, output_root, registry_path, registry_mirrors=(), root=ROOT, device_name="cuda", **paths
):
    """Train five scratch models once, verify completed folds on resume, save full OOF."""
    from fashion.train.task3_baseline import run_task3_baseline_fold

    if device_name != "cuda":
        raise ValueError("SAM25 CV requires CUDA")
    root, output_root = Path(root), Path(output_root)
    sources, classes, spec, evidence = check_gender_sam25_cv_sources(root=root, **paths)
    require_narrow_prerequisites(paths["precision_directory"], root=root)
    splits = training_splits(spec, root=root)
    destination = output_root / spec.artifact_dir / "gender"
    destination.mkdir(parents=True, exist_ok=True)
    audit = destination / "source_audit.json"
    _save_source_audit(
        audit,
        _source_identity(sources, spec, evidence, paths, root=root),
        registry_path=paths["source_registry_path"],
    )
    archive = destination / "label_variant"
    archive.mkdir(exist_ok=True)
    for source in sorted((root / VARIANT_RELATIVE_PATH).iterdir()):
        if source.suffix in {".json", ".csv"}:
            target = archive / source.name
            if target.exists() and compute_sha256(source) != compute_sha256(target):
                raise ValueError("Archived name-truth labels changed")
            if not target.exists():
                target.write_bytes(source.read_bytes())
    runs = {}
    for fold in FOLDS:
        registry = (
            pd.read_csv(registry_path, keep_default_na=False)
            if Path(registry_path).is_file()
            else pd.DataFrame(columns=REGISTRY_COLUMNS)
        )
        directory = _completed_fold(spec, fold, destination=destination, registry=registry)
        if directory is None:
            print(f"SAM25 CV: training fold {fold} from scratch for 25 epochs", flush=True)
            result = run_task3_baseline_fold(
                "gender",
                fold,
                output_root=output_root,
                registry_path=registry_path,
                registry_mirrors=registry_mirrors,
                root=root,
                device_name=device_name,
                child_spec=spec,
                parent_run_directory=sources["G2"][fold]["directory"],
                prerequisite_path=audit,
            )
            directory = Path(result["run_dir"])
        else:
            print(f"SAM25 CV: verifying completed fold {fold}", flush=True)
        run = inspect_gender_run(
            directory,
            registry=pd.read_csv(registry_path, keep_default_na=False),
            splits=splits,
            classes=classes,
            root=root,
            expected_epochs=EPOCHS,
        )
        if run["fold"] != fold:
            raise ValueError("Completed CV model belongs to the wrong fold")
        _verify_training_evidence(
            run,
            spec,
            sources["G2"][fold]["run_id"],
            evidence,
            audit_sha256=compute_sha256(audit),
            expected_epochs=EPOCHS,
        )
        verify_sam25_evidence(run, fold=fold, splits=splits, directory=directory)
        _require_resources(run["metrics"])
        run["directory"] = str(directory)
        runs[fold] = evaluate_gender_ieee(
            run,
            splits=splits,
            classes=classes,
            root=root,
            output=destination / "comparison_name_truth_ieee" / run["run_id"],
            label_variant=spec.to_dict()["gender_label_variant"],
        )
    report, predictions, robust = summarize_cv(runs, splits=splits, classes=classes)
    aggregate = destination / "aggregate"
    aggregate.mkdir(exist_ok=True)
    predictions.to_csv(aggregate / "oof_predictions.csv", index=False)
    robust.to_csv(aggregate / "robustness.csv", index=False)
    pd.DataFrame(report["folds"]).to_csv(aggregate / "fold_metrics.csv", index=False)
    metrics = report["scopes"]["all_five"]["metrics"]
    _write_json(metrics, aggregate / "metrics.json")
    pd.DataFrame(metrics["per_class"]).to_csv(aggregate / "per_class.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes).to_csv(
        aggregate / "confusion_matrix.csv", index_label="true_label"
    )
    predictions.loc[predictions.true_index.ne(predictions.predicted_index)].to_csv(
        aggregate / "errors.csv", index=False
    )
    write_original_label_diagnostic(
        {"SAM25CV": runs}, splits=splits, classes=classes, destination=aggregate, root=root
    )
    _save_model_manifest(runs, destination=destination, classes=classes, spec=spec)
    report["model_manifest"] = str(destination / "model_manifest.json")
    report["aggregate_directory"] = str(aggregate)
    plot_cv_summary(report, aggregate / "fold_summary.png")
    figure = root / "results/figures/task3/gender_sam25_cv.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes((aggregate / "fold_summary.png").read_bytes())
    _write_json(report, destination / "cv_summary.json")
    return report
