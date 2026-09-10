"""One frozen, fold-training-only gender/article loss-weighting trial."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.config import ROOT
from fashion.data import get_cv_split, get_samples
from fashion.data.gender_name_truth import load_gender_name_truth_variant
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_g2_audit import inspect_gender_run
from fashion.train.task3_gender_dropout_darkening import _verify_training_evidence
from fashion.train.task3_gender_name_truth import (
    NAME as PARENT_NAME,
)
from fashion.train.task3_gender_name_truth import (
    PARENT_RUN_IDS as GRAYSCALE_RUN_IDS,
)
from fashion.train.task3_gender_name_truth import (
    NameTruthSpec,
    _run_verified_label_screen,
    check_gender_name_truth_sources,
    label_contract,
)
from fashion.train.task3_gender_name_truth import _source_identity as parent_source_identity
from fashion.train.task3_gender_narrow import FOLDS, require_narrow_prerequisites

NAME = "gender_name_truth_article_weight_sqrt_cap3"
STRATEGY = "gender_article_sqrt_cap3_v1"
RULE_VERSION = "gname_article_weight_loss003_gap005_v1"
PARENT_RUN_IDS = (
    "t3_gender_name_truth_dropout_030_grayscale_010_gender_smallcnngem3_f0_s2753_"
    "416e64f58f57_20260905T160634Z0acdc0",
    "t3_gender_name_truth_dropout_030_grayscale_010_gender_smallcnngem3_f4_s2753_"
    "416e64f58f57_20260905T161722Za0a0f7",
)
WEIGHT_COLUMN = "gender_article_weight"
SELECTION_COLUMNS = [
    "id",
    "cv_fold",
    "product_family_group",
    "articleType",
    "gender",
    "gender_article_rows",
    "article_max_gender_rows",
    "raw_group_weight",
    WEIGHT_COLUMN,
]
WEIGHT_POLICY = {
    "version": STRATEGY,
    "fit_scope": "valid_gender_fold_training_rows_only",
    "groups": ["articleType", "gender"],
    "count_unit": "image_rows",
    "raw_weight": "min(3, sqrt(max_gender_count_within_article / gender_article_count))",
    "normalization": "divide by row-weighted mean raw weight within each articleType",
    "raw_weight_cap": 3.0,
    "exponent": 0.5,
    "loss_reduction": "sum(weight * cross_entropy) / sum(weight) per training batch",
    "validation_weighting": "none",
    "clean_training_evaluation_weighting": "none",
    "resampling": False,
}


def add_gender_article_weights(training, *, validation_fold):
    """Keep every training row and each article's total mass; tilt toward minority genders."""
    required = {"id", "partition", "cv_fold", "product_family_group", "articleType", "gender"}
    if validation_fold not in FOLDS or not required.issubset(training):
        raise ValueError("Gender/article weighting requires training metadata and fold 0 or 4")
    folds = pd.to_numeric(training.cv_fold, errors="raise")
    if (
        training.empty
        or training.id.duplicated().any()
        or not training.partition.eq("development").all()
        or not folds.isin(set(range(5)) - {validation_fold}).all()
        or training[list(required)].isna().any().any()
        or not training.gender.isin(["Boys", "Girls", "Men", "Unisex", "Women"]).all()
        or training.articleType.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Weights must use unique valid training rows, never validation or test")
    frame = training.copy()
    counts = frame.groupby(["articleType", "gender"], sort=True).size()
    largest = counts.groupby(level="articleType").max()
    keys = pd.MultiIndex.from_frame(frame[["articleType", "gender"]])
    frame["gender_article_rows"] = counts.reindex(keys).to_numpy(dtype=int)
    frame["article_max_gender_rows"] = frame.articleType.map(largest).astype(int)
    frame["raw_group_weight"] = np.minimum(
        3.0, np.sqrt(frame.article_max_gender_rows / frame.gender_article_rows)
    )
    means = frame.groupby("articleType")["raw_group_weight"].transform("mean")
    frame[WEIGHT_COLUMN] = frame.raw_group_weight / means
    selection = frame[SELECTION_COLUMNS]
    contract = {
        "policy": dict(WEIGHT_POLICY),
        "validation_fold": validation_fold,
        "training_rows": len(frame),
        "observed_groups": len(counts),
        "selection_sha256": hashlib.sha256(selection.to_csv(index=False).encode()).hexdigest(),
        "minimum": float(frame[WEIGHT_COLUMN].min()),
        "maximum": float(frame[WEIGHT_COLUMN].max()),
        "row_mean": float(frame[WEIGHT_COLUMN].mean()),
        "sum": float(frame[WEIGHT_COLUMN].sum()),
    }
    return frame, contract


def preview_gender_article_weights(splits):
    """Describe weights before fitting, using the same function as the trainer."""
    tables = []
    for fold in FOLDS:
        training = get_samples(get_cv_split(splits, fold)[0], target="gender")
        frame, _ = add_gender_article_weights(training, validation_fold=fold)
        table = (
            frame.groupby(["articleType", "gender"], sort=True)
            .agg(
                training_rows=("id", "size"),
                raw_weight=("raw_group_weight", "first"),
                loss_weight=(WEIGHT_COLUMN, "first"),
                weighted_rows=(WEIGHT_COLUMN, "sum"),
            )
            .reset_index()
        )
        table.insert(0, "validation_fold", fold)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


@dataclass(frozen=True)
class GroupWeightSpec(NameTruthSpec):
    """Keep the name-truth parent recipe except for training loss weights."""

    def __getattr__(self, key):
        overrides = {
            "name": NAME,
            "experiment_id": f"t3_{NAME}",
            "hypothesis_id": f"t3_{NAME}",
            "artifact_dir": f"experiments/t3_{NAME}",
            "run_prefix": f"t3_{NAME}",
            "changed_factor": "fold_training_gender_article_cross_entropy_weights",
            "parent_artifact_dir": f"experiments/t3_{PARENT_NAME}",
            "parent_run_ids": PARENT_RUN_IDS,
            "sample_weight_strategy": STRATEGY,
        }
        return overrides[key] if key in overrides else super().__getattr__(key)

    def parent_run_id_for_fold(self, fold):
        if fold not in FOLDS:
            raise ValueError("Gender/article weight screen allows only folds 0 and 4")
        return PARENT_RUN_IDS[FOLDS.index(fold)]

    def to_dict(self):
        payload = super().to_dict()
        payload.update(
            parent_run_ids=list(PARENT_RUN_IDS),
            sample_weight_strategy=STRATEGY,
            sample_weight_policy=dict(WEIGHT_POLICY),
            screen_rule_version=RULE_VERSION,
        )
        return payload


def group_weight_spec(root=ROOT):
    return GroupWeightSpec(json.dumps(label_contract(root), sort_keys=True))


def group_weight_config(spec, *, fold, device_name, root=ROOT):
    if not isinstance(spec, GroupWeightSpec) or spec.to_dict() != group_weight_spec(root).to_dict():
        raise ValueError("Gender/article weights or frozen name-truth recipe changed")
    if fold not in FOLDS or device_name != "cuda":
        raise ValueError("Gender/article weighting requires CUDA and only folds 0 and 4")
    return Task3BaselineConfig(target="gender")


def training_splits(spec, *, root=ROOT):
    if spec.to_dict() != group_weight_spec(root).to_dict():
        raise ValueError("Gender/article training differs from the frozen label contract")
    return load_gender_name_truth_variant(root)


def check_gender_group_weight_sources(*, name_truth_directory, root=ROOT, **paths):
    """Check the completed label trial and its full parent chain before a new fit."""
    sources, classes, old_spec, evidence = check_gender_name_truth_sources(root=root, **paths)
    directory = Path(name_truth_directory)
    audit = directory / "source_audit.json"
    previous = json.loads(audit.read_text())["identity"]
    if (
        parent_source_identity(sources, old_spec, evidence, previous["paths"], root=root)
        != previous
    ):
        raise ValueError("Completed name-truth source audit differs from verified parents")
    splits = load_gender_name_truth_variant(root)
    registry = pd.read_csv(paths["source_registry_path"], keep_default_na=False)
    direct = {}
    for fold, run_id, gray_id in zip(FOLDS, PARENT_RUN_IDS, GRAYSCALE_RUN_IDS, strict=True):
        run = inspect_gender_run(
            directory / run_id, registry=registry, splits=splits, classes=classes, root=root
        )
        if run["fold"] != fold or run["run_id"] != run_id:
            raise ValueError("Completed name-truth parent is assigned to the wrong fold")
        _verify_training_evidence(
            run, old_spec, gray_id, evidence, audit_sha256=compute_sha256(audit)
        )
        if run["config"].get("gender_label_variant") != old_spec.to_dict()["gender_label_variant"]:
            raise ValueError("Direct parent gender labels differ from the verified variant")
        run["directory"] = str(directory / run_id)
        direct[fold] = run
    sources["NameTruth"] = direct
    return sources, classes, group_weight_spec(root), evidence


def _source_identity(sources, spec, evidence, paths, *, root):
    # The old identity builder validates its own recipe; preserve its stable schema here.
    parent_spec = NameTruthSpec(spec.label_contract_json)
    identity = parent_source_identity(sources, parent_spec, evidence, paths, root=root)
    splits = training_splits(spec, root=root)
    contracts = {}
    for fold in FOLDS:
        training = get_samples(get_cv_split(splits, fold)[0], target="gender")
        _, contracts[str(fold)] = add_gender_article_weights(training, validation_fold=fold)
    identity.update(
        spec=spec.to_dict(),
        rule_version=RULE_VERSION,
        training_weight_contracts=contracts,
        weight_implementation_sha256=compute_sha256(
            Path(root) / "src/fashion/train/task3_gender_group_weight.py"
        ),
    )
    return identity


def require_group_weight_prerequisites(
    path,
    *,
    spec,
    fold,
    parent_run_directory=None,
    root=ROOT,
    device_name="cuda",
):
    group_weight_config(spec, fold=fold, device_name=device_name, root=root)
    if path is None:
        raise ValueError("Gender/article source audit is required before training")
    identity = json.loads(Path(path).read_text())["identity"]
    paths = identity["paths"]
    precision = require_narrow_prerequisites(paths["precision_directory"], root=root)
    sources, _, checked, evidence = check_gender_group_weight_sources(**paths, root=root)
    if (
        checked.to_dict() != spec.to_dict()
        or precision["artifact_sha256"] != evidence["artifact_sha256"]
        or _source_identity(sources, spec, evidence, paths, root=root) != identity
    ):
        raise ValueError("Gender/article prerequisite evidence changed")
    parent = Path(sources["NameTruth"][fold]["directory"])
    if (
        parent_run_directory is not None
        and Path(parent_run_directory).resolve() != parent.resolve()
    ):
        raise ValueError("Gender/article trial requires its verified name-truth parent directory")
    return {
        "precision": precision,
        "parent_directory": parent,
        "prerequisite_sha256": compute_sha256(path),
    }


def verify_weight_evidence(run, *, fold, splits, directory):
    """Reject completed/reused runs unless saved row weights match a fresh training-only fit."""
    training = get_samples(get_cv_split(splits, fold)[0], target="gender")
    _, expected = add_gender_article_weights(training, validation_fold=fold)
    config = run["config"]
    selection = config.get("training_selection_contract", {})
    if (
        config.get("sample_weight_strategy") != STRATEGY
        or selection.get("sample_weight_contract") != expected
        or compute_sha256(Path(directory) / "training_selection.csv")
        != expected["selection_sha256"]
        or selection.get("artifact_sha256") != expected["selection_sha256"]
    ):
        raise ValueError("Saved gender/article training weights differ from the frozen fold fit")


def run_gender_group_weight_screen(
    *,
    name_truth_directory,
    output_root,
    registry_path,
    registry_mirrors=(),
    root=ROOT,
    device_name="cuda",
    **paths,
):
    paths = dict(paths, name_truth_directory=name_truth_directory)
    sources, classes, spec, evidence = check_gender_group_weight_sources(**paths, root=root)
    group_weight_config(spec, fold=0, device_name=device_name, root=root)
    require_narrow_prerequisites(paths["precision_directory"], root=root)
    splits = training_splits(spec, root=root)
    destination = Path(output_root) / spec.artifact_dir / "gender"
    destination.mkdir(parents=True, exist_ok=True)
    preview = preview_gender_article_weights(splits).to_csv(index=False)
    preview_path = destination / "training_weight_preview.csv"
    if preview_path.exists() and preview_path.read_text() != preview:
        raise ValueError("Existing training weight preview differs; do not reuse this screen")
    preview_path.write_text(preview)
    return _run_verified_label_screen(
        sources=sources,
        classes=classes,
        spec=spec,
        evidence=evidence,
        identity=_source_identity(sources, spec, evidence, paths, root=root),
        splits=splits,
        source_registry_path=paths["source_registry_path"],
        output_root=output_root,
        registry_path=registry_path,
        registry_mirrors=registry_mirrors,
        root=root,
        device_name=device_name,
        parent_group="NameTruth",
        candidate_group="GroupWeight",
        verify_candidate=verify_weight_evidence,
    )
