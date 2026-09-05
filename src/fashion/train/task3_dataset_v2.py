"""Target-specific, single-factor Task 3 Dataset V2 screens."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from fashion.config import AUDIT_DIR, ROOT, RUNS_CSV, SPLITS_CSV
from fashion.data import load_splits
from fashion.data.hashing import compute_sha256
from fashion.train.config import Task3BaselineConfig, baseline_parameter_count

DatasetV2Name = Literal[
    "gender_v2_foreground_mask",
    "gender_v2_translation",
    "gender_translation_mild_darkening",
    "gender_weight_decay_001",
    "gender_narrow64",
    "gender_dropout_030",
    "gender_dropout_030_mild_darkening",
    "gender_dropout_045_mild_darkening",
    "gender_v2_component_weight",
    "usage_v2_component_weight",
]

SCREEN_FOLDS = (0, 4)
GENDER_G2_CONFIRMATION_FOLDS = (1, 2, 3)
GENDER_G2_ALL_FOLDS = (0, 1, 2, 3, 4)
VISUAL_COMPONENT_STRATEGY = "accepted_visual_component_v1"


@dataclass(frozen=True)
class Task3DatasetV2Spec:
    """A complete child contract compatible with the Task 3 training engine."""

    name: DatasetV2Name
    target: Literal["gender", "usage"]
    experiment_id: str
    hypothesis_id: str
    artifact_dir: str
    run_prefix: str
    changed_factor: str
    parent_artifact_dir: str
    parent_run_ids: tuple[str, ...]
    model_family: str
    run_model_token: str
    training_augmentation: str = "none"
    input_view: Literal["full", "foreground_masked"] = "full"
    sample_weight_strategy: Literal["none", "accepted_visual_component_v1"] = "none"
    loss_name: str = "cross_entropy"
    class_weight_beta: float | None = None
    class_weight_cap: float | None = None
    classifier_dropout: float = 0.0
    label_smoothing: float = 0.0
    focal_gamma: float = 0.0
    checkpoint_policy: Literal["final_epoch"] = "final_epoch"
    early_stopping_min_epoch: int = 0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    training_selection_strategy: Literal["all"] = "all"
    auxiliary_target: Literal["none"] = "none"
    primary_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        count = len(self.parent_folds)
        if (
            len(self.parent_run_ids) != count
            or len(set(self.parent_run_ids)) != count
            or any(not item for item in self.parent_run_ids)
        ):
            raise ValueError(f"a Dataset V2 screen requires {count} distinct completed parent IDs")
        for relative in (self.artifact_dir, self.parent_artifact_dir):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Dataset V2 artifact paths must stay inside the Task 3 output")
        expected = _spec_payload(self.name, self.parent_run_ids)
        actual = asdict(self)
        mismatches = {
            key: {"expected": value, "actual": actual[key]}
            for key, value in expected.items()
            if actual[key] != value
        }
        if mismatches:
            raise ValueError(
                "Dataset V2 changes more than its predeclared factor: "
                + json.dumps(mismatches, sort_keys=True)
            )

    @property
    def parent_folds(self) -> tuple[int, ...]:
        if self.name in {"gender_dropout_030_mild_darkening", "gender_dropout_045_mild_darkening"}:
            return (0, 4)
        return tuple(range(5))

    def parent_run_id_for_fold(self, fold: int) -> str:
        if fold not in self.parent_folds:
            raise ValueError(f"No completed parent for fold {fold}; allowed: {self.parent_folds}")
        return self.parent_run_ids[self.parent_folds.index(fold)]

    @property
    def weight_decay(self) -> float:
        return 0.01 if self.name == "gender_weight_decay_001" else 0.0001

    @property
    def saved_tensors_on_cpu(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["parent_run_ids"] = list(self.parent_run_ids)
        if self.name == "gender_translation_mild_darkening":
            payload["saved_tensors_on_cpu"] = False
            payload["darkening_rng"] = "persistent_worker_initial_seed_xor_0x474431"
            payload["execution_policy"] = "gpu_fp32_batch128_memory_under_3gb_no_speed_cap_v2"
        if self.name == "gender_weight_decay_001":
            payload["weight_decay"] = self.weight_decay
            payload["execution_policy"] = "gpu_fp32_batch128_memory_under_3gb_no_speed_cap_v2"
            payload["screen_rule_version"] = "gwd1_gap_and_validation_v1"
        if self.name == "gender_narrow64":
            payload["channels"] = [32, 64, 128, 64]
            payload["execution_policy"] = "g2_training_defaults_ieee_comparison_under_3gb_v1"
            payload["screen_rule_version"] = "gn64_loss003_gap005_v1"
        if self.name == "gender_dropout_030":
            payload["execution_policy"] = "g2_training_defaults_ieee_comparison_under_3gb_v1"
            payload["screen_rule_version"] = "gdrop030_loss003_gap005_v1"
        if self.name in {"gender_dropout_030_mild_darkening", "gender_dropout_045_mild_darkening"}:
            payload["parent_folds"] = list(self.parent_folds)
            payload["execution_policy"] = "g2_training_defaults_ieee_comparison_under_3gb_v1"
            payload["screen_rule_version"] = (
                "gdrop045dark_loss003_gap005_v1"
                if self.name == "gender_dropout_045_mild_darkening"
                else "gdrop030dark_loss003_gap005_v1"
            )
            payload["darkening_rng"] = "persistent_worker_initial_seed_xor_0x474431"
        return payload


def _spec_payload(name: DatasetV2Name, parent_run_ids: Sequence[str]) -> dict[str, object]:
    common = {
        "parent_run_ids": tuple(parent_run_ids),
        "classifier_dropout": 0.0,
        "label_smoothing": 0.0,
        "focal_gamma": 0.0,
        "checkpoint_policy": "final_epoch",
        "early_stopping_min_epoch": 0,
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "training_selection_strategy": "all",
        "auxiliary_target": "none",
        "primary_loss_weight": 1.0,
        "auxiliary_loss_weight": 0.0,
    }
    if name == "gender_v2_foreground_mask":
        return {
            **common,
            "name": name,
            "target": "gender",
            "experiment_id": "t3_gender_v2_g1_foreground_mask",
            "hypothesis_id": "t3_gender_v2_g1_foreground_mask",
            "artifact_dir": "experiments/t3_gender_v2_g1_foreground_mask",
            "run_prefix": "t3_gender_v2_g1_foreground_mask",
            "changed_factor": "same_canvas_foreground_mask",
            "parent_artifact_dir": "experiments/t3_gender_e6_gem_p3",
            "model_family": "task3_small_cnn_gem_p3",
            "run_model_token": "smallcnngem3",
            "training_augmentation": "none",
            "input_view": "foreground_masked",
            "sample_weight_strategy": "none",
            "loss_name": "cross_entropy",
            "class_weight_beta": None,
            "class_weight_cap": None,
        }
    if name == "gender_dropout_045_mild_darkening":
        payload = _spec_payload("gender_dropout_030_mild_darkening", parent_run_ids)
        payload.update(
            name=name,
            experiment_id="t3_gender_dropout_045_mild_darkening",
            hypothesis_id="t3_gender_dropout_045_mild_darkening",
            artifact_dir="experiments/t3_gender_dropout_045_mild_darkening",
            run_prefix="t3_gender_dropout_045_mild_darkening",
            changed_factor="post_gem_classifier_dropout_030_to_045",
            parent_artifact_dir="experiments/t3_gender_dropout_030_mild_darkening",
            classifier_dropout=0.45,
        )
        return payload
    if name == "gender_dropout_030_mild_darkening":
        payload = _spec_payload("gender_dropout_030", parent_run_ids)
        payload.update(
            name=name,
            experiment_id="t3_gender_dropout_030_mild_darkening",
            hypothesis_id="t3_gender_dropout_030_mild_darkening",
            artifact_dir="experiments/t3_gender_dropout_030_mild_darkening",
            run_prefix="t3_gender_dropout_030_mild_darkening",
            changed_factor="mild_darkening_after_translation_on_dropout_030",
            parent_artifact_dir="experiments/t3_gender_dropout_030",
            training_augmentation="translation_2px_p05_mild_darkening_p025",
        )
        return payload
    if name == "gender_dropout_030":
        payload = _spec_payload("gender_v2_translation", parent_run_ids)
        payload.update(
            name=name,
            experiment_id="t3_gender_dropout_030",
            hypothesis_id="t3_gender_dropout_030",
            artifact_dir="experiments/t3_gender_dropout_030",
            run_prefix="t3_gender_dropout_030",
            changed_factor="post_gem_classifier_dropout_0_to_030",
            parent_artifact_dir="experiments/t3_gender_v2_g2_translation",
            classifier_dropout=0.30,
        )
        return payload
    if name == "gender_narrow64":
        payload = _spec_payload("gender_v2_translation", parent_run_ids)
        payload.update(
            {
                "name": name,
                "experiment_id": "t3_gender_narrow64",
                "hypothesis_id": "t3_gender_narrow64",
                "artifact_dir": "experiments/t3_gender_narrow64",
                "run_prefix": "t3_gender_narrow64",
                "changed_factor": "final_convolution_channels_256_to_64",
                "parent_artifact_dir": "experiments/t3_gender_v2_g2_translation",
                "model_family": "task3_small_cnn_gem_p3_narrow64",
                "run_model_token": "smallcnngem3narrow64",
            }
        )
        return payload
    if name == "gender_weight_decay_001":
        payload = _spec_payload("gender_v2_translation", parent_run_ids)
        payload.update(
            {
                "name": name,
                "experiment_id": "t3_gender_weight_decay_001",
                "hypothesis_id": "t3_gender_weight_decay_001",
                "artifact_dir": "experiments/t3_gender_weight_decay_001",
                "run_prefix": "t3_gender_weight_decay_001",
                "changed_factor": "adamw_weight_decay_0001_to_001",
                "parent_artifact_dir": "experiments/t3_gender_v2_g2_translation",
            }
        )
        return payload
    if name == "gender_translation_mild_darkening":
        payload = _spec_payload("gender_v2_translation", parent_run_ids)
        payload.update(
            {
                "name": name,
                "experiment_id": "t3_gender_translation_mild_darkening",
                "hypothesis_id": "t3_gender_translation_mild_darkening",
                "artifact_dir": "experiments/t3_gender_translation_mild_darkening_gpu_v2",
                "run_prefix": "t3_gender_translation_mild_darkening",
                "changed_factor": "mild_darkening_after_translation_gpu_execution_v2",
                "parent_artifact_dir": "experiments/t3_gender_v2_g2_translation",
                "training_augmentation": "translation_2px_p05_mild_darkening_p025",
            }
        )
        return payload
    if name == "gender_v2_translation":
        return {
            **common,
            "name": name,
            "target": "gender",
            "experiment_id": "t3_gender_v2_g2_translation",
            "hypothesis_id": "t3_gender_v2_g2_translation",
            "artifact_dir": "experiments/t3_gender_v2_g2_translation",
            "run_prefix": "t3_gender_v2_g2_translation",
            "changed_factor": "training_translation",
            "parent_artifact_dir": "experiments/t3_gender_e6_gem_p3",
            "model_family": "task3_small_cnn_gem_p3",
            "run_model_token": "smallcnngem3",
            "training_augmentation": "translation_uniform_2px_p05",
            "input_view": "full",
            "sample_weight_strategy": "none",
            "loss_name": "cross_entropy",
            "class_weight_beta": None,
            "class_weight_cap": None,
        }
    if name == "gender_v2_component_weight":
        return {
            **common,
            "name": name,
            "target": "gender",
            "experiment_id": "t3_gender_v2_g3_component_weight",
            "hypothesis_id": "t3_gender_v2_g3_component_weight",
            "artifact_dir": "experiments/t3_gender_v2_g3_component_weight",
            "run_prefix": "t3_gender_v2_g3_component_weight",
            "changed_factor": "visual_component_weight",
            "parent_artifact_dir": "experiments/t3_gender_e6_gem_p3",
            "model_family": "task3_small_cnn_gem_p3",
            "run_model_token": "smallcnngem3",
            "training_augmentation": "none",
            "input_view": "full",
            "sample_weight_strategy": VISUAL_COMPONENT_STRATEGY,
            "loss_name": "visual_component_cross_entropy",
            "class_weight_beta": None,
            "class_weight_cap": None,
        }
    if name == "usage_v2_component_weight":
        return {
            **common,
            "name": name,
            "target": "usage",
            "experiment_id": "t3_usage_v2_u1_component_weight",
            "hypothesis_id": "t3_usage_v2_u1_component_weight",
            "artifact_dir": "experiments/t3_usage_v2_u1_component_weight",
            "run_prefix": "t3_usage_v2_u1_component_weight",
            "changed_factor": "visual_component_weight",
            "parent_artifact_dir": "experiments/t3_usage_e2_class_balanced_ce",
            "model_family": "task3_small_cnn",
            "run_model_token": "smallcnn",
            "training_augmentation": "none",
            "input_view": "full",
            "sample_weight_strategy": VISUAL_COMPONENT_STRATEGY,
            "loss_name": "effective_number_visual_component_cross_entropy",
            "class_weight_beta": 0.999,
            "class_weight_cap": 5.0,
        }
    raise ValueError(f"unsupported Dataset V2 screen: {name}")


def dataset_v2_spec(name: DatasetV2Name, parent_run_ids: Sequence[str]) -> Task3DatasetV2Spec:
    """Build one frozen Dataset V2 single-factor contract."""
    return Task3DatasetV2Spec(**_spec_payload(name, parent_run_ids))  # type: ignore[arg-type]


class _UnionFind:
    def __init__(self, values: Sequence[int]) -> None:
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            self.parent[high] = low


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def build_visual_component_mapping(
    splits: pd.DataFrame,
    *,
    candidates_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Freeze exact and accepted-near-duplicate components for development rows."""
    development = splits.loc[splits["partition"].eq("development")].copy()
    if development["id"].duplicated().any():
        raise ValueError("development IDs must be unique")
    development["id"] = pd.to_numeric(development["id"], errors="raise").astype(int)
    ids = development["id"].tolist()
    allowed = set(ids)
    union = _UnionFind(ids)
    for _, group in development.groupby("sha256", sort=True):
        members = sorted(group["id"].astype(int).tolist())
        for item_id in members[1:]:
            union.union(members[0], item_id)

    candidates_path = Path(candidates_path)
    if not candidates_path.is_file():
        raise FileNotFoundError(candidates_path)
    candidates = pd.read_csv(candidates_path, keep_default_na=False)
    accepted = candidates.loc[
        _as_bool(candidates["accepted_near_duplicate"])
        & candidates["role_1"].eq("labelled")
        & candidates["role_2"].eq("labelled")
    ]
    for row in accepted.itertuples(index=False):
        left, right = int(row.id_1), int(row.id_2)
        if left in allowed and right in allowed:
            union.union(left, right)

    roots = {item_id: union.find(item_id) for item_id in ids}
    members_by_root: dict[int, list[int]] = {}
    for item_id, root in roots.items():
        members_by_root.setdefault(root, []).append(item_id)
    component_by_id: dict[int, str] = {}
    for members in members_by_root.values():
        ordered = sorted(members)
        digest = hashlib.sha256(",".join(map(str, ordered)).encode("utf-8")).hexdigest()[:16]
        for item_id in ordered:
            component_by_id[item_id] = f"visual_{digest}"

    mapping = development.loc[:, ["id", "cv_fold"]].copy()
    mapping["visual_component_id"] = mapping["id"].map(component_by_id)
    fold_counts = mapping.groupby("visual_component_id")["cv_fold"].nunique()
    if int(fold_counts.max()) != 1:
        raise ValueError("an accepted visual component crosses canonical folds")
    component_sizes = mapping.groupby("visual_component_id")["id"].transform("size")
    mapping["visual_component_rows"] = component_sizes.astype(int)
    digest_rows = mapping.sort_values("id").loc[:, ["id", "visual_component_id"]]
    digest_payload = digest_rows.to_csv(index=False, header=False, lineterminator="\n")
    contract = {
        "strategy": VISUAL_COMPONENT_STRATEGY,
        "development_rows": int(len(mapping)),
        "components": int(mapping["visual_component_id"].nunique()),
        "multirow_components": int(
            mapping.loc[mapping["visual_component_rows"].gt(1), "visual_component_id"].nunique()
        ),
        "multirow_component_rows": int(mapping["visual_component_rows"].gt(1).sum()),
        "largest_component": int(mapping["visual_component_rows"].max()),
        "fold_crossings": 0,
        "mapping_sha256": hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
        "candidates_path": str(candidates_path),
        "candidates_sha256": compute_sha256(candidates_path),
    }
    return mapping.drop(columns="cv_fold"), contract


def add_visual_component_weights(
    training: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    target: str,
    candidates_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Give every accepted visual component total target-valid weight one."""
    mapping, contract = build_visual_component_mapping(
        splits,
        candidates_path=candidates_path,
    )
    weighted = training.merge(mapping, on="id", how="left", validate="one_to_one")
    if weighted["visual_component_id"].isna().any():
        raise ValueError("a fold-training row has no frozen visual component")
    target_component_rows = weighted.groupby("visual_component_id")["id"].transform("size")
    weighted["target_valid_component_rows"] = target_component_rows.astype(int)
    raw_weights = 1.0 / target_component_rows.to_numpy(dtype=np.float64)
    weighted["visual_component_weight"] = raw_weights / raw_weights.mean()
    if not np.isfinite(weighted["visual_component_weight"]).all():
        raise ValueError("visual-component weights must be finite")
    if not np.isclose(float(weighted["visual_component_weight"].mean()), 1.0):
        raise ValueError("visual-component weights must have row mean one")
    contract = {
        **contract,
        "target": target,
        "training_rows": int(len(weighted)),
        "training_components": int(weighted["visual_component_id"].nunique()),
        "weight_formula": "(1 / target_valid_training_rows_in_component) / row_mean",
        "weight_minimum": float(weighted["visual_component_weight"].min()),
        "weight_maximum": float(weighted["visual_component_weight"].max()),
        "weight_mean": float(weighted["visual_component_weight"].mean()),
        "one_row_per_epoch": True,
        "weighted_sampler": False,
    }
    return weighted, contract


def _required_parent_artifacts(spec: Task3DatasetV2Spec, *, output_root: Path) -> None:
    for fold, run_id in zip(spec.parent_folds, spec.parent_run_ids, strict=True):
        run_dir = output_root / spec.parent_artifact_dir / spec.target / run_id
        required = (
            run_dir / "config.json",
            run_dir / "final_epoch.pt",
            run_dir / "metrics.json",
            run_dir / "oof_predictions.csv",
            run_dir / "robustness.csv",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Dataset V2 parent fold {fold} is incomplete: {missing}")


def check_task3_dataset_v2_setup(
    name: DatasetV2Name,
    *,
    parent_run_ids: Sequence[str],
    output_root: str | Path,
    root: str | Path = ROOT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Check one Dataset V2 screen without creating an optimizer or taking a step."""
    if name == "gender_dropout_045_mild_darkening":
        raise ValueError("use check_gender_stronger_dropout_sources for its completed parents")
    if name == "gender_dropout_030_mild_darkening":
        raise ValueError("use check_gender_dropout_darkening_sources for the two-fold parents")
    if name == "gender_dropout_030":
        raise ValueError("use check_gender_dropout_sources for the frozen dropout screen")
    if name == "gender_narrow64":
        raise ValueError(
            "use check_gender_narrow_sources for the frozen width and precision checks"
        )
    import torch

    from fashion.train.model import Task3GeM3CNN
    from fashion.train.task3_baseline import check_task3_baseline_setup

    spec = dataset_v2_spec(name, parent_run_ids)
    output_root = Path(output_root)
    root = Path(root)
    _required_parent_artifacts(spec, output_root=output_root)
    baseline = check_task3_baseline_setup(spec.target, root=root, device_name=device_name)
    config = Task3BaselineConfig(target=spec.target)
    device = torch.device(device_name)
    model = (
        Task3GeM3CNN(config).to(device) if spec.model_family == "task3_small_cnn_gem_p3" else None
    )
    if model is not None:
        with torch.inference_mode():
            output = model(
                torch.zeros(2, 3, config.image_height, config.image_width, device=device)
            )
        if tuple(output.shape) != (2, config.num_classes):
            raise RuntimeError("Dataset V2 model output shape changed")
    component_contract: dict[str, object] | None = None
    if spec.sample_weight_strategy == VISUAL_COMPONENT_STRATEGY:
        splits = load_splits(root / SPLITS_CSV.relative_to(ROOT))
        _, component_contract = build_visual_component_mapping(
            splits,
            candidates_path=root / AUDIT_DIR.relative_to(ROOT) / "near_duplicate_candidates.csv.gz",
        )
    return {
        **baseline,
        "screen_name": name,
        "screen_folds": list(SCREEN_FOLDS),
        "child": spec.to_dict(),
        "input_view": spec.input_view,
        "training_augmentation": spec.training_augmentation,
        "sample_weight_strategy": spec.sample_weight_strategy,
        "visual_component_contract": component_contract,
        "parameter_count": baseline_parameter_count(spec.target),
        "optimizer_steps": 0,
        "ready": True,
    }


def _reusable_fold(
    spec: Task3DatasetV2Spec,
    fold: int,
    *,
    output_root: Path,
) -> dict[str, object] | None:
    target_dir = output_root / spec.artifact_dir / spec.target
    for run_dir in sorted(
        target_dir.glob(f"{spec.run_prefix}_{spec.target}_{spec.run_model_token}_f{fold}_*"),
        reverse=True,
    ):
        required = (
            run_dir / "config.json",
            run_dir / "metrics.json",
            run_dir / "final_epoch.pt",
            run_dir / "oof_predictions.csv",
            run_dir / "robustness.csv",
        )
        if not all(path.is_file() for path in required):
            continue
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if config.get("child_experiment") != spec.to_dict():
            continue
        if int(metrics.get("validation_fold", -1)) != fold:
            continue
        return {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "prediction_path": str(run_dir / "oof_predictions.csv"),
            "metrics_path": str(run_dir / "metrics.json"),
            "robustness_path": str(run_dir / "robustness.csv"),
            "metrics": metrics,
            "reused": True,
        }
    return None


def _write_json(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_task3_dataset_v2_screen(
    name: DatasetV2Name,
    *,
    parent_run_ids: Sequence[str],
    output_root: str | Path,
    folds: Sequence[int] = SCREEN_FOLDS,
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    device_name: str = "cuda",
    reuse_completed: bool = True,
) -> dict[str, object]:
    """Train folds 0 and 4 for one frozen Dataset V2 single-factor screen."""
    if name == "gender_dropout_045_mild_darkening":
        raise ValueError("use run_gender_stronger_dropout_screen for its lineage and IEEE gates")
    if name == "gender_dropout_030_mild_darkening":
        raise ValueError("use run_gender_dropout_darkening_screen for its lineage and IEEE gates")
    if name == "gender_translation_mild_darkening":
        raise ValueError("use run_gender_repair: G-D1 requires diagnostic and memory prerequisites")
    if name == "gender_weight_decay_001":
        raise ValueError("use run_gender_weight_decay_screen for the validation and gap gates")
    if name == "gender_dropout_030":
        raise ValueError("use run_gender_dropout_screen for matched IEEE evaluation and gap gates")
    if name == "gender_narrow64":
        raise ValueError("use run_gender_narrow_screen for matched IEEE evaluation and gap gates")
    from fashion.train.task3_baseline import _aggregate_target, run_task3_baseline_fold

    fold_list = tuple(int(fold) for fold in folds)
    if fold_list != SCREEN_FOLDS:
        raise ValueError("Dataset V2 screens must use folds 0 and 4 in that order")
    spec = dataset_v2_spec(name, parent_run_ids)
    output_root = Path(output_root)
    root = Path(root)
    _required_parent_artifacts(spec, output_root=output_root)
    results: list[dict[str, object]] = []
    for fold in fold_list:
        reusable = _reusable_fold(spec, fold, output_root=output_root) if reuse_completed else None
        if reusable is not None:
            print(f"[task3-v2] reusing completed {name} fold {fold}: {reusable['run_id']}")
            results.append(reusable)
            continue
        results.append(
            run_task3_baseline_fold(
                spec.target,
                fold,
                output_root=output_root,
                registry_path=registry_path,
                registry_mirrors=registry_mirrors,
                root=root,
                device_name=device_name,
                child_spec=spec,  # type: ignore[arg-type]
            )
        )
    aggregate = _aggregate_target(
        spec.target,
        results,
        output_root=output_root,
        root=root,
        artifact_dir=spec.artifact_dir,
        experiment_id=spec.experiment_id,
        hypothesis_id=spec.hypothesis_id,
        model_family=spec.model_family,
        parameter_count=baseline_parameter_count(spec.target),
        architecture_macs=None,
        child_spec=spec,  # type: ignore[arg-type]
    )
    fold_metrics = [dict(result["metrics"]) for result in results]
    aggregate_metrics = dict(aggregate["metrics"])
    aggregate_metrics.update(
        {
            "screen_only": True,
            "validation_folds": list(fold_list),
            "input_view": spec.input_view,
            "training_augmentation": spec.training_augmentation,
            "sample_weight_strategy": spec.sample_weight_strategy,
            "fold_macro_f1": [float(item["macro_f1"]) for item in fold_metrics],
            "fold_macro_f1_sample_sd": float(
                np.std([float(item["macro_f1"]) for item in fold_metrics], ddof=1)
            ),
            "mean_final_train_validation_gap": float(
                np.mean(
                    [float(item["final_train_validation_macro_f1_gap"]) for item in fold_metrics]
                )
            ),
        }
    )
    metrics_path = Path(str(aggregate["metrics_path"]))
    _write_json(aggregate_metrics, metrics_path)
    aggregate["metrics"] = aggregate_metrics
    return aggregate


def check_task3_gender_v2_g2_confirmation_setup(
    *,
    parent_run_ids: Sequence[str],
    output_root: str | Path,
    root: str | Path = ROOT,
    device_name: str = "cuda",
) -> dict[str, object]:
    """Confirm that G2 folds 0/4 exist before fitting only folds 1/2/3."""
    check = check_task3_dataset_v2_setup(
        "gender_v2_translation",
        parent_run_ids=parent_run_ids,
        output_root=output_root,
        root=root,
        device_name=device_name,
    )
    spec = dataset_v2_spec("gender_v2_translation", parent_run_ids)
    completed_screen = {
        fold: _reusable_fold(spec, fold, output_root=Path(output_root)) for fold in SCREEN_FOLDS
    }
    missing = [fold for fold, result in completed_screen.items() if result is None]
    if missing:
        raise FileNotFoundError(
            f"G2 confirmation requires completed screen folds 0 and 4; missing {missing}"
        )
    return {
        **check,
        "completed_screen_fold_run_ids": {
            str(fold): str(result["run_id"])
            for fold, result in completed_screen.items()
            if result is not None
        },
        "confirmation_folds_to_train": list(GENDER_G2_CONFIRMATION_FOLDS),
        "five_fold_order": list(GENDER_G2_ALL_FOLDS),
        "optimizer_steps": 0,
        "ready": True,
    }


def _g2_aggregate(
    spec: Task3DatasetV2Spec,
    results: Sequence[dict[str, object]],
    *,
    output_root: Path,
    root: Path,
    directory_name: str,
    scope: str,
) -> dict[str, object]:
    from fashion.train.task3_baseline import _aggregate_target

    aggregate = _aggregate_target(
        spec.target,
        results,
        output_root=output_root,
        root=root,
        artifact_dir=spec.artifact_dir,
        experiment_id=spec.experiment_id,
        hypothesis_id=spec.hypothesis_id,
        model_family=spec.model_family,
        parameter_count=baseline_parameter_count(spec.target),
        architecture_macs=None,
        child_spec=spec,  # type: ignore[arg-type]
        aggregate_dir_name=directory_name,
    )
    fold_metrics = [dict(result["metrics"]) for result in results]
    metrics = dict(aggregate["metrics"])
    metrics.update(
        {
            "confirmation_scope": scope,
            "validation_folds": [int(item["validation_fold"]) for item in fold_metrics],
            "fold_macro_f1": [float(item["macro_f1"]) for item in fold_metrics],
            "fold_macro_f1_sample_sd": float(
                np.std([float(item["macro_f1"]) for item in fold_metrics], ddof=1)
            ),
            "mean_final_train_validation_gap": float(
                np.mean(
                    [float(item["final_train_validation_macro_f1_gap"]) for item in fold_metrics]
                )
            ),
        }
    )
    _write_json(metrics, Path(str(aggregate["metrics_path"])))
    aggregate["metrics"] = metrics
    return aggregate


def run_task3_gender_v2_g2_confirmation(
    *,
    parent_run_ids: Sequence[str],
    output_root: str | Path,
    folds: Sequence[int] = GENDER_G2_CONFIRMATION_FOLDS,
    registry_path: str | Path = RUNS_CSV,
    registry_mirrors: Sequence[str | Path] = (),
    root: str | Path = ROOT,
    device_name: str = "cuda",
    reuse_completed: bool = True,
) -> dict[str, object]:
    """Train only missing G2 folds 1/2/3, then pool fresh and all-five evidence."""
    from fashion.train.task3_baseline import run_task3_baseline_fold

    fold_list = tuple(int(fold) for fold in folds)
    if fold_list != GENDER_G2_CONFIRMATION_FOLDS:
        raise ValueError("G2 confirmation must train folds 1, 2, and 3 in that order")
    spec = dataset_v2_spec("gender_v2_translation", parent_run_ids)
    output_root = Path(output_root)
    root = Path(root)
    _required_parent_artifacts(spec, output_root=output_root)

    screen_results: dict[int, dict[str, object]] = {}
    for fold in SCREEN_FOLDS:
        result = _reusable_fold(spec, fold, output_root=output_root)
        if result is None:
            raise FileNotFoundError(f"completed G2 screen fold {fold} is required")
        screen_results[fold] = result

    fresh_results: list[dict[str, object]] = []
    for fold in fold_list:
        reusable = _reusable_fold(spec, fold, output_root=output_root) if reuse_completed else None
        if reusable is not None:
            print(f"[task3-v2] reusing completed G2 confirmation fold {fold}: {reusable['run_id']}")
            fresh_results.append(reusable)
        else:
            fresh_results.append(
                run_task3_baseline_fold(
                    spec.target,
                    fold,
                    output_root=output_root,
                    registry_path=registry_path,
                    registry_mirrors=registry_mirrors,
                    root=root,
                    device_name=device_name,
                    child_spec=spec,  # type: ignore[arg-type]
                )
            )

    fresh = _g2_aggregate(
        spec,
        fresh_results,
        output_root=output_root,
        root=root,
        directory_name="aggregate_fresh_folds_1_2_3",
        scope="fresh_confirmation_folds_1_2_3",
    )
    by_fold = {int(result["metrics"]["validation_fold"]): result for result in fresh_results}
    by_fold.update(screen_results)
    all_results = [by_fold[fold] for fold in GENDER_G2_ALL_FOLDS]
    all_five = _g2_aggregate(
        spec,
        all_results,
        output_root=output_root,
        root=root,
        directory_name="aggregate_five_fold",
        scope="all_canonical_folds_0_to_4",
    )
    return {
        "target": "gender",
        "trained_folds": list(fold_list),
        "reused_screen_folds": list(SCREEN_FOLDS),
        "fresh": fresh,
        "all_five": all_five,
        "fold_run_ids": [str(result["run_id"]) for result in all_results],
    }
