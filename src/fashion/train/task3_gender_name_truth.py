"""Label-only scratch trial over the frozen dropout/darkening/grayscale recipe."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples, load_splits
from fashion.data.gender_name_truth import (
    VARIANT_RELATIVE_PATH,
    load_gender_name_truth_variant,
)
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import _reusable_fold, _write_json, dataset_v2_spec
from fashion.train.task3_decisions import CORE_CORRUPTIONS, oof_metrics, validate_oof
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_dropout_darkening import (
    DARKENING_RUN_IDS,
    GRAYSCALE_NAME,
    _save_source_audit,
    _train_fold,
    _verify_training_evidence,
    compare_with_dropout,
)
from fashion.train.task3_gender_dropout_darkening import (
    _source_identity as grayscale_source_identity,
)
from fashion.train.task3_gender_grayscale import check_gender_grayscale_sources
from fashion.train.task3_gender_ieee import POLICY, evaluate_gender_ieee
from fashion.train.task3_gender_narrow import (
    FOLDS,
    MEMORY_LIMIT,
    TRAINING_PRECISION,
    evaluate_gender_narrow_screen,
    require_narrow_prerequisites,
)
from fashion.train.task3_gender_weight_decay import _pool

NAME = "gender_name_truth_dropout_030_grayscale_010"
RULE_VERSION = "gname_truth_loss003_gap005_v1"
PARENT_RUN_IDS = (
    "t3_gender_dropout_030_mild_darkening_grayscale_010_gender_smallcnngem3_f0_s2753_"
    "eb37119b7e68_20260905T144455Ze47ef6",
    "t3_gender_dropout_030_mild_darkening_grayscale_010_gender_smallcnngem3_f4_s2753_"
    "eb37119b7e68_20260905T145521Z7d0a34",
)


def label_contract(root=ROOT):
    """Validate labels against names and bind their bytes to every new run."""
    splits = load_gender_name_truth_variant(root)
    return {
        **splits.attrs["gender_label_variant"],
        "summary_sha256": compute_sha256(Path(root) / VARIANT_RELATIVE_PATH / "summary.json"),
    }


@dataclass(frozen=True)
class NameTruthSpec:
    """Delegate unchanged controls without altering older recipe serialization."""

    label_contract_json: str

    def __getattr__(self, key):
        overrides = {
            "name": NAME,
            "experiment_id": f"t3_{NAME}",
            "hypothesis_id": f"t3_{NAME}",
            "artifact_dir": f"experiments/t3_{NAME}",
            "run_prefix": f"t3_{NAME}",
            "changed_factor": "development_gender_labels_from_explicit_product_name_cues",
            "parent_artifact_dir": f"experiments/t3_{GRAYSCALE_NAME}",
            "parent_run_ids": PARENT_RUN_IDS,
        }
        if key in overrides:
            return overrides[key]
        return getattr(dataset_v2_spec(GRAYSCALE_NAME, DARKENING_RUN_IDS), key)

    def parent_run_id_for_fold(self, fold):
        if fold not in FOLDS:
            raise ValueError("Name-truth screen allows only folds 0 and 4")
        return PARENT_RUN_IDS[FOLDS.index(fold)]

    def to_dict(self):
        payload = dataset_v2_spec(GRAYSCALE_NAME, DARKENING_RUN_IDS).to_dict()
        for key in (
            "name",
            "experiment_id",
            "hypothesis_id",
            "artifact_dir",
            "run_prefix",
            "changed_factor",
            "parent_artifact_dir",
        ):
            payload[key] = getattr(self, key)
        payload.update(
            parent_run_ids=list(PARENT_RUN_IDS),
            gender_label_variant=json.loads(self.label_contract_json),
            screen_rule_version=RULE_VERSION,
            independent_blind_test=False,
        )
        return payload


def name_truth_spec(root=ROOT):
    return NameTruthSpec(json.dumps(label_contract(root), sort_keys=True))


def name_truth_config(spec, *, fold, device_name, root=ROOT):
    if not isinstance(spec, NameTruthSpec) or spec.to_dict() != name_truth_spec(root).to_dict():
        raise ValueError("Name-truth labels or frozen recipe changed")
    if fold not in FOLDS or device_name != "cuda":
        raise ValueError("Name-truth training requires CUDA and only folds 0 and 4")
    return Task3BaselineConfig(target="gender")


def training_splits(spec, *, root=ROOT):
    """The trainer uses this verified frame for both training and validation."""
    if spec.to_dict() != name_truth_spec(root).to_dict():
        raise ValueError("Name-truth dataset differs from the frozen training contract")
    return load_gender_name_truth_variant(root)


def rescore_original_labels(predictions, *, corrected, original, classes, run_id):
    """Change only the scoring truth; never change probabilities or predicted classes."""
    checked = validate_oof(
        predictions,
        corrected,
        target="gender",
        classes=classes,
        run_ids_by_fold={int(f): run_id for f in corrected.cv_fold.unique()},
    )
    original = original.set_index("id").loc[checked.id].reset_index()
    rescored = checked.copy()
    rescored["true_label"] = original.gender.to_numpy()
    rescored["true_index"] = rescored.true_label.map(dict(zip(classes, range(len(classes)))))
    rescored = validate_oof(
        rescored,
        original,
        target="gender",
        classes=classes,
        run_ids_by_fold={int(f): run_id for f in original.cv_fold.unique()},
    )
    pd.testing.assert_frame_equal(
        checked.drop(columns=["true_label", "true_index"]),
        rescored.drop(columns=["true_label", "true_index"]),
    )
    return rescored


def write_original_label_diagnostic(groups, *, splits, classes, destination, root):
    """Score the same persisted IEEE probabilities on both label sets, with no GPU work."""
    canonical = load_splits(Path(root) / "data/processed/splits.csv")
    rows, detail, source_hashes = [], [], {}
    for group, runs in groups.items():
        for fold, run in runs.items():
            corrected = [get_samples(f, target="gender") for f in get_cv_split(splits, fold)]
            original = [get_samples(f, target="gender") for f in get_cv_split(canonical, fold)]
            directory = Path(run["evaluation_directory"])
            manifest_path = directory / "evaluation_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest != run["evaluation_manifest"]:
                raise ValueError("Evaluation manifest changed before original-label scoring")
            source_hashes[run["run_id"]] = compute_sha256(manifest_path)
            for view, filename, part in (
                ("clean_train", "clean_train_predictions.csv", 0),
                ("clean_validation", "oof_predictions.csv", 1),
                *((name, f"{name}_predictions.csv", 1) for name in CORE_CORRUPTIONS),
            ):
                path = directory / filename
                if compute_sha256(path) != manifest["files"][filename]:
                    raise ValueError("IEEE probabilities changed before original-label scoring")
                predictions = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
                teacher = rescore_original_labels(
                    predictions,
                    corrected=corrected[part],
                    original=original[part],
                    classes=classes,
                    run_id=run["run_id"],
                )
                scores = {
                    "name_truth": oof_metrics(predictions, classes),
                    "original_teacher": oof_metrics(teacher, classes),
                }
                row = {"group": group, "fold": fold, "run_id": run["run_id"], "view": view}
                detail.append({**row, "scores": scores})
                for basis, metrics in scores.items():
                    for metric in ("macro_f1", "nll", "ece_15"):
                        row[f"{basis}_{metric}"] = metrics[metric]
                rows.append(row)
    table = Path(destination) / "label_basis_comparison.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    _write_json(
        {
            "purpose": "descriptive label sensitivity; no acceptance gates",
            "probabilities_and_predictions_unchanged": True,
            "source_evaluation_manifest_sha256": source_hashes,
            "canonical_split_sha256": compute_sha256(Path(root) / "data/processed/splits.csv"),
            "name_truth_contract": label_contract(root),
            "table_sha256": compute_sha256(table),
            "scores": detail,
        },
        Path(destination) / "original_label_diagnostic.json",
    )


def check_gender_name_truth_sources(*, grayscale_directory, root=ROOT, **paths):
    """Verify historical runs on original labels before scoring them on new labels."""
    sources, classes, old_spec, evidence = check_gender_grayscale_sources(root=root, **paths)
    directory = Path(grayscale_directory)
    audit = directory / "source_audit.json"
    previous = json.loads(audit.read_text())["identity"]
    if (
        grayscale_source_identity(sources, old_spec, evidence, previous["paths"], root=root)
        != previous
    ):
        raise ValueError("Completed grayscale source audit differs from verified parents")
    registry = pd.read_csv(paths["source_registry_path"], keep_default_na=False)
    canonical = load_splits(Path(root) / "data/processed/splits.csv")
    direct = {}
    for fold, run_id in zip(FOLDS, PARENT_RUN_IDS, strict=True):
        run = inspect_gender_run(
            directory / run_id, registry=registry, splits=canonical, classes=classes, root=root
        )
        if run["fold"] != fold or run["run_id"] != run_id:
            raise ValueError("Completed grayscale parent is assigned to the wrong fold")
        _verify_training_evidence(
            run,
            old_spec,
            DARKENING_RUN_IDS[FOLDS.index(fold)],
            evidence,
            audit_sha256=compute_sha256(audit),
        )
        run["directory"] = str(directory / run_id)
        direct[fold] = run
    sources["Gray10"] = direct
    return sources, classes, name_truth_spec(root), evidence


def _source_identity(sources, spec, evidence, paths, *, root):
    return {
        "spec": spec.to_dict(),
        "baseline_controls": name_truth_config(
            spec, fold=0, device_name="cuda", root=root
        ).to_dict(),
        "paths": {key: str(Path(value).resolve()) for key, value in paths.items()},
        "source_sha256": {
            run["run_id"]: run["sha256"] for group in sources.values() for run in group.values()
        },
        "precision_evidence_sha256": evidence["artifact_sha256"],
        "split_sha256": compute_sha256(Path(root) / "data/processed/splits.csv"),
        "folds": list(FOLDS),
        "rule_version": RULE_VERSION,
        "training_precision": TRAINING_PRECISION,
        "comparison_precision": POLICY,
        "comparison_label_basis": spec.to_dict()["gender_label_variant"],
    }


def require_name_truth_prerequisites(
    path,
    *,
    spec,
    fold,
    parent_run_directory=None,
    root=ROOT,
    device_name="cuda",
):
    name_truth_config(spec, fold=fold, device_name=device_name, root=root)
    if path is None:
        raise ValueError("Name-truth source audit is required before training")
    identity = json.loads(Path(path).read_text())["identity"]
    paths = identity["paths"]
    precision = require_narrow_prerequisites(paths["precision_directory"], root=root)
    sources, _, checked_spec, evidence = check_gender_name_truth_sources(**paths, root=root)
    if (
        checked_spec.to_dict() != spec.to_dict()
        or precision["artifact_sha256"] != evidence["artifact_sha256"]
        or _source_identity(sources, spec, evidence, paths, root=root) != identity
    ):
        raise ValueError("Name-truth prerequisite evidence changed")
    parent = Path(sources["Gray10"][fold]["directory"])
    if (
        parent_run_directory is not None
        and Path(parent_run_directory).resolve() != parent.resolve()
    ):
        raise ValueError("Name-truth requires its verified grayscale parent directory")
    return {
        "precision": precision,
        "parent_directory": parent,
        "prerequisite_sha256": compute_sha256(path),
    }


def run_gender_name_truth_screen(
    *,
    g2_directory,
    e6_directory,
    dropout_directory,
    darkening_directory,
    grayscale_directory,
    source_registry_path,
    precision_directory,
    output_root,
    registry_path,
    registry_mirrors=(),
    root=ROOT,
    device_name="cuda",
):
    """Run a two-fold label sensitivity screen; every comparator uses name-truth labels."""
    root, output_root = Path(root), Path(output_root)
    paths = dict(
        g2_directory=g2_directory,
        e6_directory=e6_directory,
        dropout_directory=dropout_directory,
        darkening_directory=darkening_directory,
        grayscale_directory=grayscale_directory,
        source_registry_path=source_registry_path,
        precision_directory=precision_directory,
    )
    sources, classes, spec, evidence = check_gender_name_truth_sources(**paths, root=root)
    name_truth_config(spec, fold=0, device_name=device_name, root=root)
    require_narrow_prerequisites(precision_directory, root=root)
    destination = output_root / spec.artifact_dir / "gender"
    destination.mkdir(parents=True, exist_ok=True)
    audit = destination / "source_audit.json"
    _save_source_audit(
        audit,
        _source_identity(sources, spec, evidence, paths, root=root),
        registry_path=source_registry_path,
    )
    # Keep the label evidence beside the Drive run, even after the Colab VM expires.
    archive = destination / "label_variant"
    archive.mkdir(exist_ok=True)
    for source in sorted((root / VARIANT_RELATIVE_PATH).iterdir()):
        if source.suffix not in {".csv", ".json"}:
            continue
        target = archive / source.name
        if target.exists() and compute_sha256(target) != compute_sha256(source):
            raise ValueError(f"Archived label evidence changed: {source.name}")
        if not target.exists():
            target.write_bytes(source.read_bytes())
    splits = training_splits(spec, root=root)
    contract = spec.to_dict()["gender_label_variant"]

    def evaluate(run):
        evaluated = evaluate_gender_ieee(
            run,
            splits=splits,
            classes=classes,
            root=root,
            output=destination / "comparison_name_truth_ieee" / run["run_id"],
            label_variant=contract,
        )
        if evaluated["evaluation_manifest"]["identity"].get("label_variant") != contract:
            raise ValueError("Cannot compare runs scored against different gender labels")
        return evaluated

    matched = {name: {} for name in ("G2", "E6", "Gray10")}
    for name in matched:
        for fold in FOLDS:
            matched[name][fold] = evaluate(sources[name][fold])
    child = {}
    for fold in FOLDS:
        result = _reusable_fold(spec, fold, output_root=output_root)
        if result is None:
            result = _train_fold(
                validation_fold=fold,
                output_root=output_root,
                registry_path=registry_path,
                registry_mirrors=registry_mirrors,
                root=root,
                device_name=device_name,
                child_spec=spec,
                parent_run_directory=sources["Gray10"][fold]["directory"],
                prerequisite_path=audit,
            )
        run = inspect_gender_run(
            Path(result["run_dir"]),
            registry=pd.read_csv(registry_path, keep_default_na=False),
            splits=splits,
            classes=classes,
            root=root,
        )
        if run["fold"] != fold:
            raise ValueError("Name-truth candidate is assigned to the wrong fold")
        _verify_training_evidence(
            run,
            spec,
            sources["Gray10"][fold]["run_id"],
            evidence,
            audit_sha256=compute_sha256(audit),
        )
        if run["config"].get("gender_label_variant") != contract:
            raise ValueError("Candidate was not trained on the verified name-truth dataset")
        memory = run["metrics"]["peak_memory_bytes"]
        if not np.isfinite(memory) or not 0 < memory < MEMORY_LIMIT:
            stopped = {
                "status": "fail",
                "phase": "screen",
                "fold": fold,
                "reason": "GPU memory must stay below 3 GB; later folds were not started.",
            }
            _write_json(stopped, destination / "screen_decision.json")
            return stopped
        run["directory"] = result["run_dir"]
        child[fold] = evaluate(run)
    for group in (*matched.values(), child):
        if any(
            run["evaluation_manifest"]["identity"].get("label_variant") != contract
            for run in group.values()
        ):
            raise ValueError("Cannot compare runs scored against different gender labels")
    report = evaluate_gender_narrow_screen(child, matched, classes, experiment_name=NAME)
    report.update(
        registry_and_artifact_integrity=True,
        comparison_label_basis=contract,
        run_ids={str(f): child[f]["run_id"] for f in FOLDS},
        independent_blind_test=False,
        incremental_comparison=compare_with_dropout(child, matched["Gray10"], classes),
        direct_parent_run_ids={str(f): matched["Gray10"][f]["run_id"] for f in FOLDS},
    )
    report["incremental_comparison"]["comparison"] = (
        "candidate minus completed Gray10; both evaluated on name-truth labels"
    )
    write_original_label_diagnostic(
        {**matched, "NameTruth": child},
        splits=splits,
        classes=classes,
        destination=destination,
        root=root,
    )
    _write_json(report, destination / "screen_decision.json")
    _write_json(report["incremental_comparison"], destination / "incremental_comparison.json")
    pd.DataFrame(report["folds"]).to_csv(destination / "clean_gap_comparison.csv", index=False)
    _pool(child).sort_values("id").to_csv(destination / "ieee_oof_predictions.csv", index=False)
    return report
