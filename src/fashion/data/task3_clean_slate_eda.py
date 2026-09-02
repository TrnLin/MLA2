"""Leakage-safe, Task 3-specific EDA proposed by the clean-slate review.

The functions in this module deliberately separate label-blind image measurements
from target-aware summaries and diagnostic probes.  Protected target values are
never loaded.  Any fitted transform is scoped to an outer-fold training
complement, while the canonical folds in ``data/processed/splits.csv`` remain
unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.ndimage import binary_propagation
from skimage.color import rgb2gray
from skimage.feature import daisy, hog
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from fashion.config import (
    LABEL_MAPS_JSON,
    RANDOM_SEED,
    RESULTS_DIR,
    ROOT,
    RUNS_CSV,
    TEACHER_TRAIN_IMAGE_DIR,
)
from fashion.data.dataset import load_splits
from fashion.data.hashing import compute_sha256, write_deterministic_csv
from fashion.data.splits import cv_assignment_digest
from fashion.train.registry import RunRegistry

TASK3_EDA_DIR = RESULTS_DIR / "evidence/task3/clean_slate_eda"
TASK3_EDA_FIGURE_DIR = RESULTS_DIR / "figures/task3/clean_slate_eda"
AUDIT_SCHEMA_VERSION = "task3_clean_slate_eda_v2"

RARE_USAGE_LABELS = ("NA", "Smart Casual", "Travel", "Party", "Home")
GENDER_REVIEW_CHOICES = ("Boys", "Girls", "Men", "Unisex", "Women", "cannot_tell")
USAGE_REVIEW_CHOICES = (
    "Casual",
    "Ethnic",
    "Formal",
    "Home",
    "NA",
    "Party",
    "Smart Casual",
    "Sports",
    "Travel",
    "cannot_tell",
)
KNOWABILITY_CHOICES = ("clear", "uncertain", "cannot_tell")
CONFIDENCE_CHOICES = ("high", "medium", "low", "cannot_tell")
REVIEW_REQUIRED_COLUMNS = (
    "gender_knowability",
    "gender_judgement",
    "gender_confidence",
    "usage_knowability",
    "usage_judgement",
    "usage_confidence",
)
FOREGROUND_WHITE_THRESHOLD = 245
FOREGROUND_MIN_FRACTION = 0.005
FOREGROUND_MAX_FRACTION = 0.95
FOREGROUND_PADDING_PIXELS = 2
PROBE_VIEWS = (
    "full_rgb_hog",
    "foreground_masked_hog",
    "foreground_letterbox_hog",
    "border_hog",
    "silhouette_hog",
    "grayscale_hog",
    "colour_summary",
    "global_geometry_colour",
)
NEIGHBOUR_REPRESENTATIONS = ("raw_pixels", "hog", "scattering", "fisher")

DIAGNOSTIC_FEATURES = (
    "log_file_size",
    "brightness",
    "contrast",
    "edge_strength",
    "near_white_fraction",
    "border_connected_white_fraction",
    "foreground_fraction",
    "foreground_bbox_area_fraction",
    "foreground_width_fraction",
    "foreground_height_fraction",
    "foreground_center_x",
    "foreground_center_y",
    "red_mean",
    "green_mean",
    "blue_mean",
    "saturation_mean",
    "value_mean",
    "jpeg_quantization_sum",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _stable_token(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError:
        return path.absolute().as_posix()


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _verified_cache_contract(
    path: Path,
    expected: Mapping[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any] | None:
    """Return a cache contract only when its inputs and every artifact still match."""

    if not path.is_file():
        return None
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if any(observed.get(key) != value for key, value in expected.items()):
        return None
    artifacts = observed.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        return None
    for relative, expected_hash in artifacts.items():
        artifact = base_dir / relative
        if not artifact.is_file() or compute_sha256(artifact) != expected_hash:
            return None
    return observed


def _development(splits: pd.DataFrame) -> pd.DataFrame:
    if "partition" not in splits or "cv_fold" not in splits:
        raise ValueError("Task 3 EDA requires the canonical partition and cv_fold columns")
    rows = splits[splits["partition"].eq("development")].copy()
    if rows.empty:
        raise ValueError("Task 3 EDA found no development rows")
    rows["cv_fold"] = pd.to_numeric(rows["cv_fold"], errors="raise").astype(int)
    if sorted(rows["cv_fold"].unique()) != [0, 1, 2, 3, 4]:
        raise ValueError("Task 3 EDA requires the five canonical folds")
    return rows


def _teacher_image_root(root: Path) -> Path:
    configured = (
        TEACHER_TRAIN_IMAGE_DIR
        if root.absolute() == ROOT.absolute()
        else root / "data/raw/teacher/train/images_train"
    )
    if not configured.is_dir():
        raise FileNotFoundError(f"Task 3 teacher image directory is missing: {configured}")
    return configured.resolve(strict=True)


def _teacher_only_development(splits: pd.DataFrame, *, root: str | Path) -> pd.DataFrame:
    """Reject every development image not reached through the teacher train directory."""

    root = Path(root).absolute()
    teacher_root = _teacher_image_root(root)
    rows = _development(splits)
    if "path" not in rows:
        raise ValueError("Task 3 EDA requires a path column")
    checked: list[str] = []
    for raw_path in rows["path"].astype(str):
        path = Path(raw_path)
        if path.is_absolute():
            raise ValueError(f"Task 3 EDA rejects absolute image paths: {path}")
        absolute = (root / path).resolve(strict=True)
        if not absolute.is_file() or not absolute.is_relative_to(teacher_root):
            raise ValueError(f"Task 3 EDA rejects non-teacher image path: {path}")
        checked.append(path.as_posix())
    rows["path"] = checked
    return rows


def _teacher_manifest_digest(development: pd.DataFrame, *, root: Path) -> str:
    digest = hashlib.sha256()
    for row in development.sort_values("id").itertuples(index=False):
        declared = str(getattr(row, "sha256", "")).strip()
        image_hash = compute_sha256(root / str(row.path))
        if declared and declared != image_hash:
            raise ValueError(f"teacher image hash disagrees with splits.csv for ID {int(row.id)}")
        payload = (
            f"{int(row.id)}\x1f{row.path}\x1f{image_hash}\x1f"
            f"{row.product_family_group}\x1f{int(row.cv_fold)}\n"
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def _task3_label_scope_digest(development: pd.DataFrame) -> str:
    columns = [
        "id",
        "partition",
        "cv_fold",
        "product_family_group",
        "duplicate_group",
        "path",
        "gender",
        "usage",
        "has_gender_label",
        "has_usage_label",
    ]
    missing = [column for column in columns if column not in development]
    if missing:
        raise ValueError(f"Task 3 split is missing label-scope columns: {missing}")
    digest = hashlib.sha256()
    for row in development[columns].sort_values("id").itertuples(index=False, name=None):
        digest.update(("\x1f".join(str(value) for value in row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def build_clean_slate_audit_contract(
    splits: pd.DataFrame,
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Return the fail-closed teacher-only provenance contract for every EDA cache."""

    root = Path(root).absolute()
    development = _teacher_only_development(splits, root=root)
    configuration = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "teacher_image_directory": _relative(root / "data/raw/teacher/train/images_train", root),
        "foreground": {
            "white_threshold": FOREGROUND_WHITE_THRESHOLD,
            "minimum_fraction": FOREGROUND_MIN_FRACTION,
            "maximum_fraction": FOREGROUND_MAX_FRACTION,
            "padding_pixels": FOREGROUND_PADDING_PIXELS,
        },
        "probe_views": list(PROBE_VIEWS),
        "neighbour_representations": list(NEIGHBOUR_REPRESENTATIONS),
        "diagnostic_features": list(DIAGNOSTIC_FEATURES),
        "image_canvas": [60, 80],
        "code_sha256": compute_sha256(Path(__file__)),
        "split_digest": cv_assignment_digest(splits),
        "task3_label_scope_digest": _task3_label_scope_digest(development),
        "teacher_manifest_digest": _teacher_manifest_digest(development, root=root),
        "environment": _environment(),
    }
    configuration["audit_contract_hash"] = _stable_token(
        json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    )
    return configuration


def _stable_family_sample(
    rows: pd.DataFrame,
    *,
    target: str,
    labels: Sequence[str],
    per_label: int,
    seed: int,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for label in labels:
        members = rows[rows[target].astype(str).eq(label)].copy()
        members["_order"] = [
            _stable_token(seed, target, label, family, item_id)
            for family, item_id in zip(
                members["product_family_group"], members["id"], strict=True
            )
        ]
        members.sort_values(["_order", "id"], inplace=True)
        members = members.drop_duplicates("product_family_group").head(per_label)
        selected.append(members.drop(columns="_order"))
    if not selected:
        return rows.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)


def _canonical_observability_review(
    development: pd.DataFrame,
    *,
    root: Path,
    sample_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    valid_usage = development[_as_bool(development["has_usage_label"])].copy()
    rare = valid_usage[valid_usage["usage"].astype(str).isin(RARE_USAGE_LABELS)].copy()
    common_labels = [
        label
        for label in valid_usage["usage"].astype(str).value_counts().index
        if label not in RARE_USAGE_LABELS
    ]
    common = _stable_family_sample(
        valid_usage,
        target="usage",
        labels=common_labels,
        per_label=sample_per_class,
        seed=seed,
    )
    valid_gender = development[_as_bool(development["has_gender_label"])].copy()
    gender = _stable_family_sample(
        valid_gender,
        target="gender",
        labels=sorted(valid_gender["gender"].astype(str).unique()),
        per_label=sample_per_class,
        seed=seed,
    )

    reasons: dict[int, set[str]] = defaultdict(set)
    for frame, reason in (
        (rare, "all_ultra_rare_usage"),
        (common, "common_usage_family_sample"),
        (gender, "gender_family_sample"),
    ):
        for item_id in frame["id"].astype(int):
            reasons[item_id].add(reason)

    review = development[development["id"].astype(int).isin(reasons)].copy()
    review["review_token"] = review["id"].map(
        lambda item_id: f"T3R-{_stable_token(seed, int(item_id))[:12]}"
    )
    review["selection_reason"] = review["id"].map(
        lambda item_id: ";".join(sorted(reasons[int(item_id)]))
    )
    review["teacher_image_path"] = review["path"].map(
        lambda value: _relative(root / str(value), root)
    )
    review.sort_values("review_token", inplace=True)
    counts = {
        "ultra_rare_usage_rows": int(len(rare)),
        "common_usage_sample_rows": int(len(common)),
        "gender_sample_rows": int(len(gender)),
    }
    return review, counts


def build_observability_review_pack(
    splits: pd.DataFrame,
    *,
    output_dir: str | Path = TASK3_EDA_DIR / "observability_review",
    root: str | Path = ROOT,
    sample_per_class: int = 25,
    seed: int = RANDOM_SEED,
    audit_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create two label-blind reviewer files and a separately stored answer key.

    All ultra-rare usage rows are included.  The remaining rows are selected one
    product family at a time for common usage and every gender class.  No human
    answer is fabricated.  The pack remains available for later review, but the
    project currently records it as deferred and non-blocking for model screens.
    """

    development = _teacher_only_development(splits, root=root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(root)
    review, counts = _canonical_observability_review(
        development,
        root=root,
        sample_per_class=sample_per_class,
        seed=seed,
    )
    if audit_contract is None:
        audit_contract = build_clean_slate_audit_contract(splits, root=root)

    key_path = output / "answer_key/observability_answer_key.csv"
    stale_exposed_key = output / "observability_answer_key.csv"
    stale_exposed_key.unlink(missing_ok=True)

    blind = review[["review_token", "teacher_image_path"]].copy()
    scope_path = output / "observability_review_scope.csv"
    write_deterministic_csv(blind, scope_path, index=False)
    for column in (
        "gender_knowability",
        "gender_judgement",
        "gender_confidence",
        "gender_alternatives",
        "usage_knowability",
        "usage_judgement",
        "usage_confidence",
        "usage_alternatives",
        "person_or_mannequin",
        "item_count",
        "object_size",
        "clipping_or_visibility_issue",
        "reviewer_notes",
    ):
        blind[column] = ""
    reviewer_paths = []
    review_scope_changed = False
    for reviewer in (1, 2):
        path = output / f"observability_reviewer_{reviewer}.csv"
        preserve = False
        if path.is_file():
            existing = pd.read_csv(path, keep_default_na=False)
            preserve = list(existing.columns) == list(blind.columns) and existing[
                ["review_token", "teacher_image_path"]
            ].astype(str).reset_index(drop=True).equals(
                blind[["review_token", "teacher_image_path"]]
                .astype(str)
                .reset_index(drop=True)
            )
        if not preserve:
            write_deterministic_csv(blind, path, index=False)
            review_scope_changed = True
        reviewer_paths.append(path)

    lock_path = output / "observability_review_lock.json"
    contract_changed = False
    if lock_path.is_file():
        try:
            existing_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            contract_changed = True
        else:
            contract_changed = (
                existing_lock.get("audit_contract_hash")
                != audit_contract["audit_contract_hash"]
                or existing_lock.get("task3_label_scope_digest")
                != audit_contract["task3_label_scope_digest"]
            )
    if review_scope_changed or contract_changed:
        for stale in (
            key_path,
            lock_path,
            output / "observability_agreement_summary.csv",
            output / "observability_agreement_by_fold.csv",
            output / "observability_disagreements.csv",
        ):
            stale.unlink(missing_ok=True)

    instructions = {
        "independence_rule": "reviewers complete files separately before opening answer_key/",
        "required_columns": list(REVIEW_REQUIRED_COLUMNS),
        "knowability_choices": list(KNOWABILITY_CHOICES),
        "confidence_choices": list(CONFIDENCE_CHOICES),
        "gender_judgement_choices": list(GENDER_REVIEW_CHOICES),
        "usage_judgement_choices": list(USAGE_REVIEW_CHOICES),
        "cannot_tell_rule": (
            "when knowability is cannot_tell, judgement and confidence must also be cannot_tell"
        ),
    }
    _write_json(instructions, output / "observability_review_instructions.json")

    summary = pd.DataFrame(
        [
            {
                "review_status": "deferred_non_blocking",
                "unique_images": int(len(review)),
                **counts,
                "reviewers_required": 2,
                "labels_exposed_in_reviewer_files": 0,
            }
        ]
    )
    summary_path = output / "observability_review_summary.csv"
    write_deterministic_csv(summary, summary_path, index=False)
    return {
        "summary": summary,
        "reviewer_paths": reviewer_paths,
        "answer_key_path": key_path,
        "selected": review,
    }


def validate_observability_reviewer(
    reviewer: pd.DataFrame,
    *,
    expected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate one completed blind form without opening the answer key."""

    forbidden = {"id", "gender", "usage", "articleType", "cv_fold"}
    if leaked := forbidden.intersection(reviewer.columns):
        raise ValueError(f"blind reviewer file exposes teacher columns: {sorted(leaked)}")
    identity = {"review_token", "teacher_image_path"}
    required = identity.union(REVIEW_REQUIRED_COLUMNS)
    if missing := required.difference(reviewer.columns):
        raise ValueError(f"reviewer file is missing columns: {sorted(missing)}")
    if reviewer.empty or reviewer["review_token"].duplicated().any():
        raise ValueError("reviewer file must contain one row per unique review token")
    if expected is not None:
        expected_identity = expected[["review_token", "teacher_image_path"]].astype(str)
        observed_identity = reviewer[["review_token", "teacher_image_path"]].astype(str)
        if not observed_identity.reset_index(drop=True).equals(
            expected_identity.reset_index(drop=True)
        ):
            raise ValueError("reviewer rows or teacher image paths differ from the frozen scope")

    for column in REVIEW_REQUIRED_COLUMNS:
        if reviewer[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"reviewer column is incomplete: {column}")
    for target, choices in (
        ("gender", GENDER_REVIEW_CHOICES),
        ("usage", USAGE_REVIEW_CHOICES),
    ):
        knowability = reviewer[f"{target}_knowability"].astype(str).str.strip()
        judgement = reviewer[f"{target}_judgement"].astype(str).str.strip()
        confidence = reviewer[f"{target}_confidence"].astype(str).str.strip()
        if invalid := sorted(set(knowability).difference(KNOWABILITY_CHOICES)):
            raise ValueError(f"invalid {target} knowability values: {invalid}")
        if invalid := sorted(set(judgement).difference(choices)):
            raise ValueError(f"invalid {target} judgement values: {invalid}")
        if invalid := sorted(set(confidence).difference(CONFIDENCE_CHOICES)):
            raise ValueError(f"invalid {target} confidence values: {invalid}")
        any_cannot_tell = (
            knowability.eq("cannot_tell")
            | judgement.eq("cannot_tell")
            | confidence.eq("cannot_tell")
        )
        all_cannot_tell = (
            knowability.eq("cannot_tell")
            & judgement.eq("cannot_tell")
            & confidence.eq("cannot_tell")
        )
        if not any_cannot_tell.eq(all_cannot_tell).all():
            raise ValueError(f"{target} cannot_tell fields are inconsistent")
    return reviewer.copy()


def analyse_observability_reviews(
    splits: pd.DataFrame,
    *,
    output_dir: str | Path = TASK3_EDA_DIR / "observability_review",
    root: str | Path = ROOT,
    seed: int = RANDOM_SEED,
    sample_per_class: int = 25,
) -> dict[str, pd.DataFrame]:
    """Lock two completed forms, then open the answer key and compute agreement."""

    output = Path(output_dir)
    reviewer_paths = [output / f"observability_reviewer_{reviewer}.csv" for reviewer in (1, 2)]
    scope_path = output / "observability_review_scope.csv"
    answer_key_path = output / "answer_key/observability_answer_key.csv"
    lock_path = output / "observability_review_lock.json"
    if not scope_path.is_file():
        raise FileNotFoundError("the frozen observability review scope is missing")
    saved_scope = pd.read_csv(scope_path, keep_default_na=False)
    if list(saved_scope.columns) != ["review_token", "teacher_image_path"]:
        raise ValueError("the frozen observability review scope has an invalid schema")
    root = Path(root)
    development = _teacher_only_development(splits, root=root).copy()
    canonical_review, _ = _canonical_observability_review(
        development,
        root=root,
        sample_per_class=sample_per_class,
        seed=seed,
    )
    expected = canonical_review[["review_token", "teacher_image_path"]]
    if not saved_scope.astype(str).reset_index(drop=True).equals(
        expected.astype(str).reset_index(drop=True)
    ):
        raise ValueError("the saved review scope differs from the canonical Task 3 sample")
    audit_contract = build_clean_slate_audit_contract(splits, root=root)
    reviewers = [
        validate_observability_reviewer(
            pd.read_csv(path, keep_default_na=False), expected=expected
        )
        for path in reviewer_paths
    ]
    first_identity = reviewers[0][["review_token", "teacher_image_path"]].astype(str)
    second_identity = reviewers[1][["review_token", "teacher_image_path"]].astype(str)
    if not first_identity.reset_index(drop=True).equals(second_identity.reset_index(drop=True)):
        raise ValueError("reviewer files do not share the same frozen scope and order")

    form_hashes = {
        "audit_contract_hash": audit_contract["audit_contract_hash"],
        "task3_label_scope_digest": audit_contract["task3_label_scope_digest"],
        "scope_sha256": compute_sha256(scope_path),
        "reviewer_1_sha256": compute_sha256(reviewer_paths[0]),
        "reviewer_2_sha256": compute_sha256(reviewer_paths[1]),
    }
    if lock_path.is_file():
        existing_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing_lock.get("status") != "complete" or any(
            existing_lock.get(key) != value for key, value in form_hashes.items()
        ):
            raise ValueError("observability forms changed after they were locked")
        protected = {
            "answer_key_sha256": answer_key_path,
            "agreement_summary_sha256": output / "observability_agreement_summary.csv",
            "agreement_by_fold_sha256": output / "observability_agreement_by_fold.csv",
            "disagreements_sha256": output / "observability_disagreements.csv",
        }
        if any(
            not path.is_file() or compute_sha256(path) != existing_lock.get(key)
            for key, path in protected.items()
        ):
            raise ValueError("locked observability evidence was changed or removed")
        return {
            "summary": pd.read_csv(protected["agreement_summary_sha256"]),
            "by_fold": pd.read_csv(protected["agreement_by_fold_sha256"]),
            "disagreements": pd.read_csv(
                protected["disagreements_sha256"], keep_default_na=False
            ),
        }

    development["review_token"] = development["id"].map(
        lambda item_id: f"T3R-{_stable_token(seed, int(item_id))[:12]}"
    )
    answer_key = first_identity[["review_token"]].merge(
        development[
            [
                "review_token",
                "id",
                "gender",
                "usage",
                "articleType",
                "cv_fold",
                "product_family_group",
            ]
        ],
        on="review_token",
        how="left",
        validate="one_to_one",
    )
    if answer_key["id"].isna().any():
        raise ValueError("reviewer scope contains tokens outside the canonical teacher split")

    lock = {
        "status": "forms_locked",
        "locked_at_utc": datetime.now(UTC).isoformat(),
        **form_hashes,
        "answer_key_created_after_validation": False,
    }
    _write_json(lock, lock_path)
    write_deterministic_csv(answer_key, answer_key_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    disagreement_rows: list[pd.DataFrame] = []
    keyed = answer_key.set_index("review_token")
    for target in ("gender", "usage"):
        first = reviewers[0].set_index("review_token")
        second = reviewers[1].set_index("review_token")
        joined = pd.DataFrame(
            {
                "teacher_label": keyed[target].astype(str),
                "reviewer_1": first[f"{target}_judgement"].astype(str),
                "reviewer_2": second[f"{target}_judgement"].astype(str),
                "reviewer_1_knowability": first[f"{target}_knowability"].astype(str),
                "reviewer_2_knowability": second[f"{target}_knowability"].astype(str),
                "cv_fold": keyed["cv_fold"].astype(int),
            }
        ).reset_index()
        both_knowable = joined["reviewer_1"].ne("cannot_tell") & joined[
            "reviewer_2"
        ].ne("cannot_tell")

        def metrics(rows: pd.DataFrame) -> dict[str, Any]:
            knowable = rows["reviewer_1"].ne("cannot_tell") & rows["reviewer_2"].ne(
                "cannot_tell"
            )
            first_knowable = rows["reviewer_1"].ne("cannot_tell")
            second_knowable = rows["reviewer_2"].ne("cannot_tell")
            response_labels = set(rows["reviewer_1"]).union(rows["reviewer_2"])
            kappa = (
                float(cohen_kappa_score(rows["reviewer_1"], rows["reviewer_2"]))
                if len(response_labels) > 1
                else float(rows["reviewer_1"].eq(rows["reviewer_2"]).all())
            )
            return {
                "rows": int(len(rows)),
                "reviewer_agreement": float(rows["reviewer_1"].eq(rows["reviewer_2"]).mean()),
                "both_visually_knowable_rate": float(knowable.mean()),
                "reviewer_1_teacher_agreement": float(
                    rows.loc[first_knowable, "reviewer_1"].eq(
                        rows.loc[first_knowable, "teacher_label"]
                    ).mean()
                ),
                "reviewer_2_teacher_agreement": float(
                    rows.loc[second_knowable, "reviewer_2"].eq(
                        rows.loc[second_knowable, "teacher_label"]
                    ).mean()
                ),
                "cohen_kappa_all_responses": kappa,
            }

        summary_rows.append({"target": target, **metrics(joined)})
        for fold, part in joined.groupby("cv_fold", sort=True):
            fold_rows.append({"target": target, "cv_fold": int(fold), **metrics(part)})
        disagreement = joined[
            ~joined["reviewer_1"].eq(joined["reviewer_2"])
            | (both_knowable & ~joined["reviewer_1"].eq(joined["teacher_label"]))
            | (both_knowable & ~joined["reviewer_2"].eq(joined["teacher_label"]))
        ].copy()
        disagreement.insert(0, "target", target)
        disagreement_rows.append(disagreement)

    summary = pd.DataFrame(summary_rows)
    by_fold = pd.DataFrame(fold_rows)
    disagreements = pd.concat(disagreement_rows, ignore_index=True)
    write_deterministic_csv(summary, output / "observability_agreement_summary.csv", index=False)
    write_deterministic_csv(by_fold, output / "observability_agreement_by_fold.csv", index=False)
    write_deterministic_csv(
        disagreements, output / "observability_disagreements.csv", index=False
    )
    lock.update(
        {
            "status": "complete",
            "answer_key_created_after_validation": True,
            "answer_key_sha256": compute_sha256(answer_key_path),
            "agreement_summary_sha256": compute_sha256(
                output / "observability_agreement_summary.csv"
            ),
            "agreement_by_fold_sha256": compute_sha256(
                output / "observability_agreement_by_fold.csv"
            ),
            "disagreements_sha256": compute_sha256(
                output / "observability_disagreements.csv"
            ),
        }
    )
    _write_json(lock, lock_path)
    return {"summary": summary, "by_fold": by_fold, "disagreements": disagreements}


@dataclass(frozen=True)
class ForegroundProposal:
    mask: np.ndarray
    raw_fraction: float
    bbox: tuple[int, int, int, int]
    fallback_reason: str


def foreground_proposal(
    image: Image.Image | np.ndarray,
    *,
    white_threshold: int = FOREGROUND_WHITE_THRESHOLD,
) -> ForegroundProposal:
    """Return a border-connected near-white background proposal and safe fallback."""

    array = (
        np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
        if isinstance(image, Image.Image)
        else np.asarray(image, dtype=np.uint8)
    )
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("foreground proposal expects an RGB image")
    near_white = np.all(array >= white_threshold, axis=2)
    seeds = np.zeros_like(near_white)
    seeds[0, :] = near_white[0, :]
    seeds[-1, :] = near_white[-1, :]
    seeds[:, 0] |= near_white[:, 0]
    seeds[:, -1] |= near_white[:, -1]
    background = binary_propagation(seeds, mask=near_white)
    raw_foreground = ~background
    raw_fraction = float(raw_foreground.mean())

    reason = ""
    if not raw_foreground.any() or raw_fraction < FOREGROUND_MIN_FRACTION:
        reason = "empty_or_tiny_foreground"
    elif raw_fraction > FOREGROUND_MAX_FRACTION:
        reason = "foreground_too_large"
    if reason:
        height, width = raw_foreground.shape
        return ForegroundProposal(
            mask=np.ones_like(raw_foreground),
            raw_fraction=raw_fraction,
            bbox=(0, 0, height, width),
            fallback_reason=reason,
        )

    rows, columns = np.where(raw_foreground)
    top = max(0, int(rows.min()) - FOREGROUND_PADDING_PIXELS)
    bottom = min(array.shape[0], int(rows.max()) + FOREGROUND_PADDING_PIXELS + 1)
    left = max(0, int(columns.min()) - FOREGROUND_PADDING_PIXELS)
    right = min(array.shape[1], int(columns.max()) + FOREGROUND_PADDING_PIXELS + 1)
    return ForegroundProposal(raw_foreground, raw_fraction, (top, left, bottom, right), "")


def foreground_views(image: Image.Image | np.ndarray) -> dict[str, np.ndarray]:
    """Build separate same-canvas masks and aspect-preserving foreground crops."""

    array = (
        np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
        if isinstance(image, Image.Image)
        else np.asarray(image, dtype=np.uint8)
    )
    proposal = foreground_proposal(array)
    mask = proposal.mask
    foreground_masked = np.full_like(array, 255)
    foreground_masked[mask] = array[mask]
    top, left, bottom, right = proposal.bbox
    crop = Image.fromarray(foreground_masked[top:bottom, left:right])
    scale = min(array.shape[1] / crop.width, array.shape[0] / crop.height)
    resized_size = (
        max(1, round(crop.width * scale)),
        max(1, round(crop.height * scale)),
    )
    resized = crop.resize(resized_size, Image.Resampling.LANCZOS)
    letterbox = Image.new("RGB", (array.shape[1], array.shape[0]), (255, 255, 255))
    letterbox.paste(
        resized,
        ((array.shape[1] - resized.width) // 2, (array.shape[0] - resized.height) // 2),
    )

    border = np.full_like(array, 255)
    border_height = max(1, round(array.shape[0] * 0.10))
    border_width = max(1, round(array.shape[1] * 0.10))
    border[:border_height] = array[:border_height]
    border[-border_height:] = array[-border_height:]
    border[:, :border_width] = array[:, :border_width]
    border[:, -border_width:] = array[:, -border_width:]
    gray = np.asarray(Image.fromarray(array).convert("L"), dtype=np.uint8)
    return {
        "full": array,
        "foreground_masked": foreground_masked,
        "foreground_letterbox": np.asarray(letterbox, dtype=np.uint8),
        "border": border,
        "silhouette": (mask.astype(np.uint8) * 255),
        "grayscale": gray,
    }


def _jpeg_quantization_sum(image: Image.Image) -> float:
    tables = getattr(image, "quantization", None)
    if not tables:
        return 0.0
    return float(sum(sum(int(value) for value in table) for table in tables.values()))


def measure_teacher_image(path: str | Path, *, root: str | Path = ROOT) -> dict[str, Any]:
    """Measure one image without reading any target label."""

    root = Path(root)
    path = Path(path)
    absolute = path if path.is_absolute() else root / path
    with Image.open(absolute) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        quantization_sum = _jpeg_quantization_sum(source)
    array = np.asarray(image, dtype=np.uint8)
    proposal = foreground_proposal(array)
    top, left, bottom, right = proposal.bbox
    height, width = array.shape[:2]
    rgb = array.astype(np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    edge_x = float(np.abs(np.diff(gray, axis=1)).mean()) if width > 1 else 0.0
    edge_y = float(np.abs(np.diff(gray, axis=0)).mean()) if height > 1 else 0.0
    near_white = np.all(array >= FOREGROUND_WHITE_THRESHOLD, axis=2)
    bbox_area = max(bottom - top, 0) * max(right - left, 0)
    return {
        "path": _relative(absolute, root),
        "width": width,
        "height": height,
        "log_file_size": float(math.log1p(absolute.stat().st_size)),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "edge_strength": (edge_x + edge_y) / 2,
        "near_white_fraction": float(near_white.mean()),
        "border_connected_white_fraction": float(1.0 - proposal.raw_fraction),
        "foreground_fraction": proposal.raw_fraction,
        "foreground_bbox_area_fraction": float(bbox_area / (height * width)),
        "foreground_width_fraction": float((right - left) / width),
        "foreground_height_fraction": float((bottom - top) / height),
        "foreground_center_x": float(((left + right) / 2) / width),
        "foreground_center_y": float(((top + bottom) / 2) / height),
        "red_mean": float(rgb[..., 0].mean()),
        "green_mean": float(rgb[..., 1].mean()),
        "blue_mean": float(rgb[..., 2].mean()),
        "saturation_mean": float(hsv[..., 1].mean()),
        "value_mean": float(hsv[..., 2].mean()),
        "jpeg_quantization_sum": quantization_sum,
        "foreground_fallback_reason": proposal.fallback_reason,
    }


def build_teacher_image_diagnostics(
    splits: pd.DataFrame,
    *,
    output_path: str | Path = TASK3_EDA_DIR / "teacher_image_diagnostics.csv.gz",
    root: str | Path = ROOT,
    workers: int | None = None,
    reuse: bool = True,
    audit_contract_hash: str | None = None,
) -> pd.DataFrame:
    """Measure every development image once and cache a label-free row table."""

    output = Path(output_path)
    root = Path(root)
    development = _teacher_only_development(splits, root=root)[
        ["id", "path", "cv_fold", "product_family_group"]
    ]
    if audit_contract_hash is None:
        audit_contract_hash = build_clean_slate_audit_contract(
            splits, root=root
        )["audit_contract_hash"]
    expected_columns = [
        "id",
        "cv_fold",
        "product_family_group",
        "path",
        "width",
        "height",
        *DIAGNOSTIC_FEATURES,
        "foreground_fallback_reason",
    ]
    contract_path = output.with_name(f"{output.name}.contract.json")
    contract = {
        "artifact": "teacher_image_diagnostics",
        "audit_contract_hash": audit_contract_hash,
        "columns": expected_columns,
    }
    if (
        reuse
        and output.is_file()
        and _verified_cache_contract(
            contract_path, contract, base_dir=output.parent
        )
        is not None
    ):
        cached = pd.read_csv(output, keep_default_na=False)
        if (
            list(cached.columns) == expected_columns
            and len(cached) == len(development)
            and set(cached["id"]) == set(development["id"])
        ):
            canonical = development.set_index("id")
            observed = cached.set_index("id")
            identity_matches = (
                observed["cv_fold"].astype(int).eq(canonical["cv_fold"].astype(int)).all()
                and observed["product_family_group"]
                .astype(str)
                .eq(canonical["product_family_group"].astype(str))
                .all()
                and observed["path"].astype(str).eq(canonical["path"].astype(str)).all()
            )
            if identity_matches:
                return cached

    def measure(row: Mapping[str, Any]) -> dict[str, Any]:
        return {"id": int(row["id"]), **measure_teacher_image(row["path"], root=root)}

    worker_count = workers or min(16, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        measured = list(executor.map(measure, development.to_dict("records")))
    diagnostics = development[["id", "cv_fold", "product_family_group"]].merge(
        pd.DataFrame(measured), on="id", validate="one_to_one"
    )
    diagnostics.sort_values("id", inplace=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_csv(diagnostics, output, index=False)
    _write_json(
        {**contract, "artifact_sha256": {output.name: compute_sha256(output)}},
        contract_path,
    )
    return diagnostics


def nuisance_association_audit(
    splits: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    targets: Sequence[str] = ("gender", "usage"),
) -> pd.DataFrame:
    """Measure target association before and after removing article-type medians."""

    development = _development(splits)
    merged = development.merge(diagnostics, on="id", validate="one_to_one")
    output: list[dict[str, Any]] = []
    for target in targets:
        rows = merged[_as_bool(merged[f"has_{target}_label"])].copy()
        for feature in DIAGNOSTIC_FEATURES:
            values = pd.to_numeric(rows[feature], errors="raise").astype(float)
            residual = values - rows.groupby("articleType")[feature].transform("median")

            def eta_squared(signal: pd.Series) -> float:
                centered = signal - signal.mean()
                total = float(np.square(centered).sum())
                if total <= 0:
                    return 0.0
                group_mean = signal.groupby(rows[target].astype(str)).transform("mean")
                between = float(np.square(group_mean - signal.mean()).sum())
                return min(max(between / total, 0.0), 1.0)

            output.append(
                {
                    "target": target,
                    "feature": feature,
                    "global_eta_squared": eta_squared(values),
                    "within_article_type_eta_squared": eta_squared(residual),
                    "absolute_median_shift": float(
                        rows.assign(_value=values).groupby(target)["_value"].median().max()
                        - rows.assign(_value=values).groupby(target)["_value"].median().min()
                    ),
                }
            )
    return pd.DataFrame(output).sort_values(
        ["target", "within_article_type_eta_squared"], ascending=[True, False]
    )


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / max(first.sum(), 1.0)
    second = second / max(second.sum(), 1.0)
    midpoint = (first + second) / 2

    def divergence(values: np.ndarray) -> float:
        valid = values > 0
        return float(np.sum(values[valid] * np.log2(values[valid] / midpoint[valid])))

    return (divergence(first) + divergence(second)) / 2


def fold_artifact_audit(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Compare label-free image-property distributions across the fixed folds."""

    rows: list[dict[str, Any]] = []
    for feature in DIAGNOSTIC_FEATURES:
        values = pd.to_numeric(diagnostics[feature], errors="raise").astype(float)
        unique = values.nunique()
        if unique < 2:
            continue
        quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if len(quantiles) < 3:
            quantiles = np.linspace(values.min(), values.max() + 1e-12, 3)
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        scale = float(values.std(ddof=0)) or 1.0
        for fold in range(5):
            in_fold = diagnostics["cv_fold"].astype(int).eq(fold)
            first = np.histogram(values[in_fold], bins=quantiles)[0]
            second = np.histogram(values[~in_fold], bins=quantiles)[0]
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "jensen_shannon_divergence": _jensen_shannon(first, second),
                    "standardized_mean_shift": float(
                        (values[in_fold].mean() - values[~in_fold].mean()) / scale
                    ),
                    "fold_median": float(values[in_fold].median()),
                    "other_folds_median": float(values[~in_fold].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["jensen_shannon_divergence", "feature", "fold"], ascending=[False, True, True]
    )


def family_and_component_audit(
    splits: pd.DataFrame,
    *,
    anchor_oof_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, pd.DataFrame]:
    """Check group boundaries, component weights, and optional anchor-error slices."""

    development = _development(splits)
    boundary = pd.DataFrame(
        [
            {
                "unit": "product_family_group",
                "development_rows": int(len(development)),
                "unique_units": int(development["product_family_group"].nunique()),
                "fold_crossings": int(
                    development.groupby("product_family_group")["cv_fold"].nunique().gt(1).sum()
                ),
                "largest_unit": int(
                    development.groupby("product_family_group")["id"].size().max()
                ),
            },
            {
                "unit": "duplicate_group",
                "development_rows": int(len(development)),
                "unique_units": int(development["duplicate_group"].nunique()),
                "fold_crossings": int(
                    development.groupby("duplicate_group")["cv_fold"].nunique().gt(1).sum()
                ),
                "largest_unit": int(development.groupby("duplicate_group")["id"].size().max()),
            },
        ]
    )

    duplicate_size = development.groupby("duplicate_group")["id"].transform("size")
    weighted = development.assign(component_weight=1.0 / duplicate_size)
    weight_rows: list[dict[str, Any]] = []
    for target in ("gender", "usage"):
        valid = weighted[_as_bool(weighted[f"has_{target}_label"])].copy()
        total_weight = float(valid["component_weight"].sum())
        for label, part in valid.groupby(target, sort=True):
            weight_rows.append(
                {
                    "target": target,
                    "class_name": str(label),
                    "row_count": int(len(part)),
                    "row_share": float(len(part) / len(valid)),
                    "component_weighted_support": float(part["component_weight"].sum()),
                    "component_weighted_share": float(
                        part["component_weight"].sum() / total_weight
                    ),
                }
            )
    component_weights = pd.DataFrame(weight_rows)

    family_sizes = development.groupby("product_family_group")["id"].size()
    profile = pd.DataFrame(
        {
            "family_size": family_sizes,
        }
    ).reset_index()
    profile["size_band"] = pd.cut(
        profile["family_size"],
        bins=[0, 1, 2, 5, np.inf],
        labels=["1", "2", "3-5", "6+"],
    ).astype(str)
    family_profile = (
        profile.groupby("size_band", as_index=False, observed=False)
        .agg(families=("product_family_group", "size"), products=("family_size", "sum"))
        .sort_values("size_band")
    )

    error_rows: list[dict[str, Any]] = []
    if anchor_oof_paths:
        lookup = family_sizes.to_dict()
        id_lookup = development.set_index("id")
        for target, path in anchor_oof_paths.items():
            oof = pd.read_csv(path, keep_default_na=False)
            required = {"id", "true_label", "predicted_label", "confidence"}
            if missing := required.difference(oof.columns):
                raise ValueError(f"{target} anchor OOF is missing columns: {sorted(missing)}")
            rows = oof.merge(
                id_lookup[["product_family_group", target]],
                left_on="id",
                right_index=True,
                validate="one_to_one",
                suffixes=("_oof", "_split"),
            )
            if "product_family_group_oof" in rows:
                matches = rows["product_family_group_oof"].astype(str).eq(
                    rows["product_family_group_split"].astype(str)
                )
                if not matches.all():
                    mismatched = rows.loc[~matches, "id"].astype(int).tolist()[:10]
                    raise ValueError(
                        f"{target} anchor OOF family IDs disagree with splits for: "
                        f"{mismatched}"
                    )
                rows["product_family_group"] = rows["product_family_group_split"]
            else:
                rows = rows.rename(
                    columns={"product_family_group_split": "product_family_group"}
                )
            canonical_truth = rows[target].astype(str)
            imported_truth = rows["true_label"].astype(str).str.strip()
            imported_prediction = rows["predicted_label"].astype(str).str.strip()
            if target == "usage":
                imported_truth = imported_truth.mask(imported_truth.eq(""), "NA")
                imported_prediction = imported_prediction.mask(
                    imported_prediction.eq(""), "NA"
                )
            if not imported_truth.eq(canonical_truth).all():
                mismatched = rows.loc[~imported_truth.eq(canonical_truth), "id"].astype(int)
                raise ValueError(
                    f"{target} anchor OOF labels disagree with the canonical split for: "
                    f"{mismatched.tolist()[:10]}"
                )
            rows["true_label"] = canonical_truth
            rows["predicted_label"] = imported_prediction
            rows["family_size"] = rows["product_family_group"].map(lookup).astype(int)
            rows["size_band"] = pd.cut(
                rows["family_size"],
                bins=[0, 1, 2, 5, np.inf],
                labels=["1", "2", "3-5", "6+"],
            ).astype(str)
            rows["correct"] = rows["true_label"].astype(str).eq(
                rows["predicted_label"].astype(str)
            )
            labels = sorted(rows["true_label"].astype(str).unique())
            for band, part in rows.groupby("size_band", sort=False):
                error_rows.append(
                    {
                        "target": target,
                        "size_band": str(band),
                        "rows": int(len(part)),
                        "accuracy": float(part["correct"].mean()),
                        "macro_f1": float(
                            f1_score(
                                part["true_label"].astype(str),
                                part["predicted_label"].astype(str),
                                labels=labels,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        "mean_confidence": float(pd.to_numeric(part["confidence"]).mean()),
                        "wrong_mean_confidence": float(
                            pd.to_numeric(part.loc[~part["correct"], "confidence"]).mean()
                        ),
                    }
                )
    return {
        "boundary": boundary,
        "component_weights": component_weights,
        "family_profile": family_profile,
        "anchor_error_slices": pd.DataFrame(error_rows),
    }


def _select_probe_rows(
    development: pd.DataFrame,
    target: str,
    *,
    per_class_fold: int,
    seed: int,
) -> pd.DataFrame:
    valid = development[_as_bool(development[f"has_{target}_label"])].copy()
    selected: list[pd.DataFrame] = []
    for (fold, label), part in valid.groupby(["cv_fold", target], sort=True):
        part = part.copy()
        part["_order"] = [
            _stable_token(seed, "probe", target, fold, label, family, item_id)
            for family, item_id in zip(part["product_family_group"], part["id"], strict=True)
        ]
        part.sort_values(["_order", "id"], inplace=True)
        selected.append(part.drop_duplicates("product_family_group").head(per_class_fold))
    rows = pd.concat(selected, ignore_index=True).drop(columns="_order")
    return rows.sort_values(["cv_fold", target, "id"]).reset_index(drop=True)


def _colour_summary(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(array)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    features: list[float] = []
    for channel in range(3):
        histogram, _ = np.histogram(hsv[..., channel], bins=8, range=(0, 1), density=True)
        features.extend(histogram.astype(float))
    small = np.asarray(image.resize((4, 3), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    features.extend(small.ravel().astype(float))
    return np.asarray(features, dtype=np.float32)


def _hog(array: np.ndarray, *, colour: bool) -> np.ndarray:
    values = array.astype(np.float32) / 255.0
    return hog(
        values if colour else rgb2gray(values),
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
        channel_axis=-1 if colour else None,
    ).astype(np.float32)


def _probe_feature(path: Path, view: str, diagnostics: Mapping[str, Any]) -> np.ndarray:
    if view == "global_geometry_colour":
        return np.asarray(
            [float(diagnostics[column]) for column in DIAGNOSTIC_FEATURES], dtype=np.float32
        )
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB").resize(
            (60, 80), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.uint8)
    views = foreground_views(array)
    if view == "full_rgb_hog":
        return _hog(views["full"], colour=True)
    if view == "foreground_masked_hog":
        return _hog(views["foreground_masked"], colour=True)
    if view == "foreground_letterbox_hog":
        return _hog(views["foreground_letterbox"], colour=True)
    if view == "border_hog":
        return _hog(views["border"], colour=True)
    if view == "silhouette_hog":
        silhouette = np.repeat(views["silhouette"][..., None], 3, axis=2)
        return _hog(silhouette, colour=False)
    if view == "grayscale_hog":
        return _hog(views["full"], colour=False)
    if view == "colour_summary":
        return _colour_summary(views["full"])
    raise ValueError(f"unsupported diagnostic view: {view}")


def _feature_matrix(
    rows: pd.DataFrame,
    diagnostics: pd.DataFrame,
    view: str,
    *,
    root: Path,
    cache_dir: Path,
    workers: int,
    audit_contract_hash: str,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids = rows["id"].astype(int).tolist()
    digest = _stable_token(audit_contract_hash, view, *ids)[:16]
    path = cache_dir / f"{view}_{digest}.npy"
    contract_path = cache_dir / f"{view}_{digest}.contract.json"
    contract = {
        "artifact": "diagnostic_feature_matrix",
        "audit_contract_hash": audit_contract_hash,
        "view": view,
        "ordered_ids_sha256": _stable_token(*ids),
        "rows": len(ids),
        "dtype": "float32",
    }
    if (
        path.is_file()
        and _verified_cache_contract(contract_path, contract, base_dir=cache_dir)
        is not None
    ):
        cached = np.load(path, mmap_mode="r", allow_pickle=False)
        if cached.ndim == 2 and cached.shape[0] == len(ids) and cached.dtype == np.float32:
            return cached
    diagnostic_lookup = diagnostics.set_index("id").to_dict("index")

    def extract(row: Mapping[str, Any]) -> np.ndarray:
        return _probe_feature(
            root / str(row["path"]), view, diagnostic_lookup[int(row["id"])]
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        features = list(executor.map(extract, rows.to_dict("records")))
    matrix = np.stack(features).astype(np.float32)
    np.save(path, matrix)
    _write_json(
        {
            **contract,
            "columns": int(matrix.shape[1]),
            "artifact_sha256": {path.name: compute_sha256(path)},
        },
        contract_path,
    )
    return np.load(path, mmap_mode="r")


def _environment() -> dict[str, str]:
    import scipy
    import skimage
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }


def run_diagnostic_view_probes(
    splits: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    output_dir: str | Path = TASK3_EDA_DIR / "view_probes",
    registry_path: str | Path = RUNS_CSV,
    root: str | Path = ROOT,
    targets: Sequence[str] = ("gender", "usage"),
    views: Sequence[str] = PROBE_VIEWS,
    per_class_fold: int = 250,
    seed: int = RANDOM_SEED,
    workers: int | None = None,
    reuse: bool = True,
    audit_contract_hash: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run small, registered outer-fold probes to compare diagnostic image views.

    These deliberately balanced samples are shortcut probes, not candidate model
    scores.  Each fitted fold is registered as debug and submission-ineligible.
    """

    output = Path(output_dir)
    summary_path = output / "view_probe_summary.csv"
    fold_path = output / "view_probe_folds.csv"
    root = Path(root)
    development = _teacher_only_development(splits, root=root)
    if audit_contract_hash is None:
        audit_contract_hash = build_clean_slate_audit_contract(
            splits, root=root
        )["audit_contract_hash"]
    batch_contract = {
        "artifact": "view_probes",
        "audit_contract_hash": audit_contract_hash,
        "targets": list(targets),
        "views": list(views),
        "per_class_fold": per_class_fold,
        "seed": seed,
        "classifier": "StandardScaler+SGDClassifier(log_loss,class_weight=balanced)",
    }
    batch_id = _stable_token(json.dumps(batch_contract, sort_keys=True))[:16]
    batch_contract["batch_id"] = batch_id
    contract_path = output / "view_probe_contract.json"
    if (
        reuse
        and summary_path.is_file()
        and fold_path.is_file()
        and _verified_cache_contract(
            contract_path, batch_contract, base_dir=output
        )
        is not None
    ):
        summary = pd.read_csv(summary_path, keep_default_na=False)
        folds = pd.read_csv(fold_path, keep_default_na=False)
        expected_pairs = {(target, view) for target in targets for view in views}
        observed_pairs = set(zip(summary["target"], summary["view"], strict=True))
        if observed_pairs == expected_pairs and len(folds) == len(expected_pairs) * 5:
            return summary, folds
    output.mkdir(parents=True, exist_ok=True)
    worker_count = workers or min(12, (os.cpu_count() or 4) * 2)
    registry = RunRegistry(registry_path)
    split_digest = cv_assignment_digest(splits)
    label_digest = compute_sha256(LABEL_MAPS_JSON)
    fold_rows: list[dict[str, Any]] = []
    all_oof: list[pd.DataFrame] = []

    for target in targets:
        scope = _select_probe_rows(
            development, target, per_class_fold=per_class_fold, seed=seed
        )
        labels = sorted(scope[target].astype(str).unique())
        scope_path = output / f"{target}_probe_scope.csv"
        write_deterministic_csv(
            scope[["id", "cv_fold", "product_family_group", target]], scope_path, index=False
        )
        for view in views:
            matrix = _feature_matrix(
                scope,
                diagnostics,
                view,
                root=root,
                cache_dir=output / "feature_cache" / target,
                workers=worker_count,
                audit_contract_hash=audit_contract_hash,
            )
            view_oof: list[pd.DataFrame] = []
            for fold in range(5):
                validation = scope["cv_fold"].astype(int).eq(fold).to_numpy()
                training = ~validation
                config = {
                    "purpose": "eda_shortcut_probe_not_candidate_model",
                    "target": target,
                    "view": view,
                    "fold": fold,
                    "per_class_fold_cap": per_class_fold,
                    "seed": seed,
                    "audit_contract_hash": audit_contract_hash,
                    "batch_id": batch_id,
                    "classifier": "StandardScaler+SGDClassifier(log_loss,class_weight=balanced)",
                }
                config_hash = _stable_token(json.dumps(config, sort_keys=True))
                timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                run_id = (
                    f"t3_eda_{target}_{view}_f{fold}_s{seed}_"
                    f"{config_hash[:10]}_{timestamp}{uuid.uuid4().hex[:6]}"
                )
                run_dir = output / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                config_path = _write_json(config, run_dir / "config.json")
                started = time.perf_counter()
                registry.start(
                    {
                        "run_id": run_id,
                        "experiment_id": "t3_clean_slate_eda_view_probe",
                        "hypothesis_id": "diagnose_foreground_background_nuisance_signal",
                        "task": "task3_eda",
                        "target": target,
                        "validation_fold": fold,
                        "seed": seed,
                        "debug": True,
                        "scratch": True,
                        "submission_eligible": False,
                        "config_hash": config_hash,
                        "config_path": _relative(config_path, root),
                        "split_digest": split_digest,
                        "label_map_digest": label_digest,
                        "training_product_count": int(training.sum()),
                        "validation_product_count": int(validation.sum()),
                        "training_family_count": int(
                            scope.loc[training, "product_family_group"].nunique()
                        ),
                        "validation_family_count": int(
                            scope.loc[validation, "product_family_group"].nunique()
                        ),
                        "model_family": f"eda_linear_probe_{view}",
                        "environment_json": _environment(),
                    }
                )
                try:
                    classifier = make_pipeline(
                        StandardScaler(),
                        SGDClassifier(
                            loss="log_loss",
                            alpha=1e-4,
                            max_iter=500,
                            tol=1e-3,
                            class_weight="balanced",
                            random_state=seed + fold,
                            average=True,
                        ),
                    )
                    classifier.fit(np.asarray(matrix[training]), scope.loc[training, target])
                    predicted = classifier.predict(np.asarray(matrix[validation])).astype(str)
                    probabilities = classifier.predict_proba(np.asarray(matrix[validation]))
                    confidence = probabilities.max(axis=1)
                    truth = scope.loc[validation, target].astype(str).to_numpy()
                    fold_macro = float(
                        f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
                    )
                    fold_accuracy = float(accuracy_score(truth, predicted))
                    elapsed = time.perf_counter() - started
                    predictions = scope.loc[
                        validation, ["id", "cv_fold", "product_family_group"]
                    ].copy()
                    predictions["target"] = target
                    predictions["view"] = view
                    predictions["true_label"] = truth
                    predictions["predicted_label"] = predicted
                    predictions["confidence"] = confidence
                    prediction_path = run_dir / "oof_predictions.csv"
                    write_deterministic_csv(predictions, prediction_path, index=False)
                    metrics = {
                        "macro_f1": fold_macro,
                        "accuracy": fold_accuracy,
                        "support": int(validation.sum()),
                        "scope": "balanced diagnostic sample; not a candidate score",
                    }
                    registry.complete(
                        run_id,
                        {
                            "parameter_count": int(
                                classifier[-1].coef_.size + classifier[-1].intercept_.size
                            ),
                            "prediction_path": _relative(prediction_path, root),
                            "prediction_sha256": compute_sha256(prediction_path),
                            "metrics_json": metrics,
                            "train_seconds": elapsed,
                            "last_completed_stage": "diagnostic_probe_complete",
                        },
                    )
                    fold_rows.append(
                        {
                            "target": target,
                            "view": view,
                            "fold": fold,
                            "rows": int(validation.sum()),
                            "macro_f1": fold_macro,
                            "accuracy": fold_accuracy,
                            "train_seconds": elapsed,
                            "run_id": run_id,
                            "batch_id": batch_id,
                        }
                    )
                    view_oof.append(predictions)
                except (Exception, KeyboardInterrupt) as error:
                    registry.fail(run_id, error, last_completed_stage="registry_started")
                    raise
            combined = pd.concat(view_oof, ignore_index=True)
            combined_path = output / f"{target}_{view}_oof.csv"
            write_deterministic_csv(combined, combined_path, index=False)
            all_oof.append(combined)

    folds = pd.DataFrame(fold_rows)
    summary_rows = []
    for (target, view), part in pd.concat(all_oof).groupby(["target", "view"], sort=True):
        labels = sorted(part["true_label"].astype(str).unique())
        matching_folds = folds[folds["target"].eq(target) & folds["view"].eq(view)]
        summary_rows.append(
            {
                "target": target,
                "view": view,
                "sample_rows": int(len(part)),
                "pooled_macro_f1": float(
                    f1_score(
                        part["true_label"].astype(str),
                        part["predicted_label"].astype(str),
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "accuracy": float(
                    accuracy_score(
                        part["true_label"].astype(str), part["predicted_label"].astype(str)
                    )
                ),
                "fold_macro_f1_sd": float(matching_folds["macro_f1"].std(ddof=1)),
                "interpretation": (
                    "relative shortcut probe on balanced sample; not candidate result"
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["target", "pooled_macro_f1"], ascending=[True, False]
    )
    write_deterministic_csv(folds, fold_path, index=False)
    write_deterministic_csv(summary, summary_path, index=False)
    probe_artifacts = [
        summary_path,
        fold_path,
        *(output / f"{target}_probe_scope.csv" for target in targets),
        *(
            output / f"{target}_{view}_oof.csv"
            for target in targets
            for view in views
        ),
    ]
    _write_json(
        {
            **batch_contract,
            "artifact_sha256": {
                path.relative_to(output).as_posix(): compute_sha256(path)
                for path in probe_artifacts
            },
        },
        contract_path,
    )
    return summary, folds


def _representation_scope(
    development: pd.DataFrame,
    target: str,
    *,
    query_fold: int,
    reference_per_class: int,
    query_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = development[_as_bool(development[f"has_{target}_label"])].copy()

    def sample(rows: pd.DataFrame, cap: int, role: str) -> pd.DataFrame:
        output = []
        for label, part in rows.groupby(target, sort=True):
            part = part.copy()
            part["_order"] = [
                _stable_token(seed, "neighbour", role, target, label, family, item_id)
                for family, item_id in zip(
                    part["product_family_group"], part["id"], strict=True
                )
            ]
            part.sort_values(["_order", "id"], inplace=True)
            output.append(part.drop_duplicates("product_family_group").head(cap))
        return pd.concat(output, ignore_index=True).drop(columns="_order")

    query = sample(valid[valid["cv_fold"].astype(int).eq(query_fold)], query_per_class, "query")
    reference = sample(
        valid[valid["cv_fold"].astype(int).ne(query_fold)], reference_per_class, "reference"
    )
    if set(query["product_family_group"]).intersection(reference["product_family_group"]):
        raise ValueError("neighbourhood scope crossed product families")
    return reference.reset_index(drop=True), query.reset_index(drop=True)


def _raw_descriptor(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        gray = ImageOps.exif_transpose(opened).convert("L").resize(
            (24, 32), Image.Resampling.BILINEAR
        )
    return (np.asarray(gray, dtype=np.float32) / 255.0).ravel()


def _hog_descriptor(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB").resize(
            (60, 80), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.uint8)
    return _hog(array, colour=False)


def _scattering_descriptors(paths: Sequence[Path], *, batch_size: int = 64) -> np.ndarray:
    from kymatio.numpy import Scattering2D

    scattering = Scattering2D(J=2, shape=(80, 60), L=4, max_order=2)
    output = []
    for start in range(0, len(paths), batch_size):
        arrays = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("L").resize(
                    (60, 80), Image.Resampling.BILINEAR
                )
            arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
        coefficients = scattering(np.stack(arrays))
        pooled = np.concatenate(
            [coefficients.mean(axis=(-2, -1)), coefficients.std(axis=(-2, -1))], axis=1
        )
        output.append(pooled.astype(np.float32))
    return np.concatenate(output, axis=0)


def _daisy_descriptors(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("L").resize(
            (60, 80), Image.Resampling.BILINEAR
        )
    values = np.asarray(image, dtype=np.float32) / 255.0
    descriptors = daisy(
        values,
        step=8,
        radius=8,
        rings=2,
        histograms=4,
        orientations=8,
        normalization="l2",
    )
    return descriptors.reshape(-1, descriptors.shape[-1]).astype(np.float32)


def _fisher_encode(descriptors: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    posterior = mixture.predict_proba(descriptors)
    weights = np.maximum(mixture.weights_, 1e-12)
    standard = np.sqrt(np.maximum(mixture.covariances_, 1e-12))
    difference = (descriptors[:, None, :] - mixture.means_[None, :, :]) / standard[None, :, :]
    first = (posterior[..., None] * difference).sum(axis=0)
    first /= max(len(descriptors), 1) * np.sqrt(weights)[:, None]
    second = (posterior[..., None] * (np.square(difference) - 1)).sum(axis=0)
    second /= max(len(descriptors), 1) * np.sqrt(2 * weights)[:, None]
    vector = np.concatenate([first.ravel(), second.ravel()]).astype(np.float32)
    vector = np.sign(vector) * np.sqrt(np.abs(vector))
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _representation_features(
    reference: pd.DataFrame,
    query: pd.DataFrame,
    representation: str,
    *,
    root: Path,
    workers: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference_paths = [root / str(path) for path in reference["path"]]
    query_paths = [root / str(path) for path in query["path"]]
    paths = [*reference_paths, *query_paths]
    if representation in {"raw_pixels", "hog"}:
        function = _raw_descriptor if representation == "raw_pixels" else _hog_descriptor
        with ThreadPoolExecutor(max_workers=workers) as executor:
            combined = np.stack(list(executor.map(function, paths))).astype(np.float32)
        provenance = {
            "representation": representation,
            "configuration": (
                "grayscale_24x32_flattened"
                if representation == "raw_pixels"
                else "grayscale_hog_9orient_8x8cell_2x2block_l2hys_60x80"
            ),
        }
    elif representation == "scattering":
        combined = _scattering_descriptors(paths)
        provenance = {
            "representation": representation,
            "configuration": "Scattering2D_J2_L4_order2_60x80_mean_std_pool",
        }
    elif representation == "fisher":
        with ThreadPoolExecutor(max_workers=workers) as executor:
            reference_local = list(executor.map(_daisy_descriptors, reference_paths))
            query_local = list(executor.map(_daisy_descriptors, query_paths))
        rng = np.random.default_rng(seed)
        codebook_rows = []
        for descriptors in reference_local:
            if len(descriptors) > 12:
                chosen = rng.choice(len(descriptors), size=12, replace=False)
                descriptors = descriptors[chosen]
            codebook_rows.append(descriptors)
        codebook = np.concatenate(codebook_rows, axis=0)
        if len(codebook) > 20_000:
            codebook = codebook[rng.choice(len(codebook), size=20_000, replace=False)]
        mixture = GaussianMixture(
            n_components=8,
            covariance_type="diag",
            max_iter=100,
            reg_covar=1e-5,
            random_state=seed,
        ).fit(codebook)
        mixture_digest = hashlib.sha256()
        for values in (mixture.weights_, mixture.means_, mixture.covariances_):
            mixture_digest.update(np.ascontiguousarray(values).tobytes())
        combined = np.stack(
            [_fisher_encode(item, mixture) for item in [*reference_local, *query_local]]
        )
        provenance = {
            "representation": representation,
            "configuration": "DAISY_step8_radius8_rings2_hist4_orient8_GMM8_diag_Fisher",
            "reference_codebook_rows": int(len(codebook)),
            "gmm_sha256": mixture_digest.hexdigest(),
        }
    else:
        raise ValueError(f"unsupported representation: {representation}")
    combined = normalize(combined, norm="l2")
    provenance["output_features"] = int(combined.shape[1])
    return combined[: len(reference)], combined[len(reference) :], provenance


def representation_neighbourhood_audit(
    splits: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    output_dir: str | Path = TASK3_EDA_DIR / "representation_neighbourhoods",
    root: str | Path = ROOT,
    targets: Sequence[str] = ("gender", "usage"),
    representations: Sequence[str] = NEIGHBOUR_REPRESENTATIONS,
    query_fold: int = 0,
    reference_per_class: int = 250,
    query_per_class: int = 100,
    seed: int = RANDOM_SEED,
    workers: int | None = None,
    reuse: bool = True,
    audit_contract_hash: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure non-family nearest-neighbour purity for four distinct representations."""

    output = Path(output_dir)
    summary_path = output / "neighbourhood_summary.csv"
    slices_path = output / "neighbourhood_slices.csv"
    neighbours_path = output / "neighbourhood_neighbours.csv.gz"
    contract_path = output / "neighbourhood_contract.json"
    root = Path(root)
    development = _teacher_only_development(splits, root=root)
    if audit_contract_hash is None:
        audit_contract_hash = build_clean_slate_audit_contract(
            splits, root=root
        )["audit_contract_hash"]
    contract = {
        "artifact": "representation_neighbourhoods",
        "audit_contract_hash": audit_contract_hash,
        "targets": list(targets),
        "representations": list(representations),
        "query_fold": query_fold,
        "reference_per_class": reference_per_class,
        "query_per_class": query_per_class,
        "seed": seed,
        "distance": "cosine",
        "neighbours": 5,
    }
    if (
        reuse
        and summary_path.is_file()
        and slices_path.is_file()
        and neighbours_path.is_file()
        and _verified_cache_contract(contract_path, contract, base_dir=output)
        is not None
    ):
        summary = pd.read_csv(summary_path, keep_default_na=False)
        slices = pd.read_csv(slices_path, keep_default_na=False)
        expected = {(target, item) for target in targets for item in representations}
        observed = set(zip(summary["target"], summary["representation"], strict=True))
        if observed == expected:
            return summary, slices
    output.mkdir(parents=True, exist_ok=True)
    diagnostics_lookup = diagnostics.set_index("id")
    worker_count = workers or min(12, (os.cpu_count() or 4) * 2)
    summary_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    neighbour_rows: list[dict[str, Any]] = []
    representation_provenance: list[dict[str, Any]] = []

    for target in targets:
        reference, query = _representation_scope(
            development,
            target,
            query_fold=query_fold,
            reference_per_class=reference_per_class,
            query_per_class=query_per_class,
            seed=seed,
        )
        areas = query["id"].map(diagnostics_lookup["foreground_bbox_area_fraction"])
        query["object_size_band"] = pd.cut(
            areas.astype(float),
            bins=[-np.inf, 0.25, 0.60, np.inf],
            labels=["small", "medium", "large"],
        ).astype(str)
        write_deterministic_csv(
            reference[["id", "cv_fold", "product_family_group", target]],
            output / f"{target}_reference_scope.csv",
            index=False,
        )
        write_deterministic_csv(
            query[["id", "cv_fold", "product_family_group", target, "object_size_band"]],
            output / f"{target}_query_scope.csv",
            index=False,
        )
        for representation in representations:
            reference_features, query_features, provenance = _representation_features(
                reference,
                query,
                representation,
                root=root,
                workers=worker_count,
                seed=seed,
            )
            representation_provenance.append({"target": target, **provenance})
            neighbour_count = min(5, len(reference))
            model = NearestNeighbors(
                n_neighbors=neighbour_count, metric="cosine", algorithm="brute"
            )
            model.fit(reference_features)
            distances, indices = model.kneighbors(query_features)
            reference_labels = reference[target].astype(str).to_numpy()
            query_labels = query[target].astype(str).to_numpy()
            neighbour_labels = reference_labels[indices]
            top1 = neighbour_labels[:, 0] == query_labels
            topk = (neighbour_labels == query_labels[:, None]).mean(axis=1)
            for query_position, query_row in query.reset_index(drop=True).iterrows():
                for rank, (distance, reference_position) in enumerate(
                    zip(distances[query_position], indices[query_position], strict=True),
                    start=1,
                ):
                    reference_row = reference.iloc[int(reference_position)]
                    neighbour_rows.append(
                        {
                            "target": target,
                            "representation": representation,
                            "query_id": int(query_row["id"]),
                            "query_family": str(query_row["product_family_group"]),
                            "query_label": str(query_row[target]),
                            "rank": rank,
                            "reference_id": int(reference_row["id"]),
                            "reference_family": str(
                                reference_row["product_family_group"]
                            ),
                            "reference_label": str(reference_row[target]),
                            "cosine_distance": float(distance),
                            "label_match": bool(
                                str(reference_row[target]) == str(query_row[target])
                            ),
                            "family_match": bool(
                                str(reference_row["product_family_group"])
                                == str(query_row["product_family_group"])
                            ),
                        }
                    )
            summary_rows.append(
                {
                    "target": target,
                    "representation": representation,
                    "query_fold": query_fold,
                    "reference_rows": int(len(reference)),
                    "query_rows": int(len(query)),
                    "top1_label_purity": float(top1.mean()),
                    "top5_label_purity": float(topk.mean()),
                    "same_family_neighbours": 0,
                }
            )
            audit = query[[target, "object_size_band"]].copy()
            audit["top1"] = top1
            audit["top5"] = topk
            for slice_name, values in (
                ("class", audit[target].astype(str)),
                ("object_size", audit["object_size_band"].astype(str)),
            ):
                for value, part in audit.groupby(values, sort=True):
                    slice_rows.append(
                        {
                            "target": target,
                            "representation": representation,
                            "slice_type": slice_name,
                            "slice_value": str(value),
                            "rows": int(len(part)),
                            "top1_label_purity": float(part["top1"].mean()),
                            "top5_label_purity": float(part["top5"].mean()),
                        }
                    )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["target", "top5_label_purity"], ascending=[True, False]
    )
    slices = pd.DataFrame(slice_rows)
    write_deterministic_csv(summary, summary_path, index=False)
    write_deterministic_csv(slices, slices_path, index=False)
    write_deterministic_csv(pd.DataFrame(neighbour_rows), neighbours_path, index=False)
    provenance_path = output / "representation_provenance.json"
    _write_json(
        {"representations": representation_provenance},
        provenance_path,
    )
    neighbourhood_artifacts = [
        summary_path,
        slices_path,
        neighbours_path,
        provenance_path,
        *(
            output / f"{target}_{role}_scope.csv"
            for target in targets
            for role in ("reference", "query")
        ),
    ]
    _write_json(
        {
            **contract,
            "artifact_sha256": {
                path.relative_to(output).as_posix(): compute_sha256(path)
                for path in neighbourhood_artifacts
            },
        },
        contract_path,
    )
    return summary, slices


def _verified_human_review(
    output: Path,
    audit_contract: Mapping[str, Any],
) -> tuple[bool, list[Path]]:
    review_dir = output / "observability_review"
    artifacts = [
        review_dir / "observability_review_scope.csv",
        review_dir / "observability_reviewer_1.csv",
        review_dir / "observability_reviewer_2.csv",
        review_dir / "answer_key/observability_answer_key.csv",
        review_dir / "observability_agreement_summary.csv",
        review_dir / "observability_agreement_by_fold.csv",
        review_dir / "observability_disagreements.csv",
        review_dir / "observability_review_lock.json",
    ]
    lock_path = artifacts[-1]
    if not lock_path.is_file() or any(not path.is_file() for path in artifacts):
        return False, artifacts
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, artifacts
    expected = {
        "scope_sha256": artifacts[0],
        "reviewer_1_sha256": artifacts[1],
        "reviewer_2_sha256": artifacts[2],
        "answer_key_sha256": artifacts[3],
        "agreement_summary_sha256": artifacts[4],
        "agreement_by_fold_sha256": artifacts[5],
        "disagreements_sha256": artifacts[6],
    }
    valid = (
        lock.get("status") == "complete"
        and lock.get("audit_contract_hash") == audit_contract["audit_contract_hash"]
        and lock.get("task3_label_scope_digest")
        == audit_contract["task3_label_scope_digest"]
        and all(compute_sha256(path) == lock.get(key) for key, path in expected.items())
    )
    return valid, artifacts


def write_clean_slate_eda_tables(
    *,
    splits: pd.DataFrame | None = None,
    output_dir: str | Path = TASK3_EDA_DIR,
    root: str | Path = ROOT,
    anchor_oof_paths: Mapping[str, str | Path] | None = None,
    run_probes: bool = True,
    run_neighbourhoods: bool = True,
) -> dict[str, Any]:
    """Run all automatable clean-slate audits and write their evidence tables."""

    splits = load_splits() if splits is None else splits
    root = Path(root).absolute()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit_contract = build_clean_slate_audit_contract(splits, root=root)
    audit_contract_hash = str(audit_contract["audit_contract_hash"])
    _write_json(audit_contract, output / "audit_contract.json")
    review = build_observability_review_pack(
        splits,
        output_dir=output / "observability_review",
        root=root,
        audit_contract=audit_contract,
    )
    diagnostics = build_teacher_image_diagnostics(
        splits,
        output_path=output / "teacher_image_diagnostics.csv.gz",
        root=root,
        audit_contract_hash=audit_contract_hash,
    )
    nuisance = nuisance_association_audit(splits, diagnostics)
    write_deterministic_csv(nuisance, output / "nuisance_association.csv", index=False)
    fold_shifts = fold_artifact_audit(diagnostics)
    write_deterministic_csv(fold_shifts, output / "fold_artifact_shifts.csv", index=False)
    family = family_and_component_audit(splits, anchor_oof_paths=anchor_oof_paths)
    for name, table in family.items():
        write_deterministic_csv(table, output / f"{name}.csv", index=False)

    probe_summary = probe_folds = None
    if run_probes:
        probe_summary, probe_folds = run_diagnostic_view_probes(
            splits,
            diagnostics,
            output_dir=output / "view_probes",
            root=root,
            audit_contract_hash=audit_contract_hash,
        )
    neighbourhood_summary = neighbourhood_slices = None
    if run_neighbourhoods:
        neighbourhood_summary, neighbourhood_slices = representation_neighbourhood_audit(
            splits,
            diagnostics,
            output_dir=output / "representation_neighbourhoods",
            root=root,
            audit_contract_hash=audit_contract_hash,
        )

    human_complete, human_outputs = _verified_human_review(output, audit_contract)
    artifact_paths = [
        output / "audit_contract.json",
        output / "teacher_image_diagnostics.csv.gz",
        output / "teacher_image_diagnostics.csv.gz.contract.json",
        output / "nuisance_association.csv",
        output / "fold_artifact_shifts.csv",
        *(output / f"{name}.csv" for name in family),
        output / "observability_review/observability_review_instructions.json",
        output / "observability_review/observability_review_summary.csv",
        output / "observability_review/observability_review_scope.csv",
        *(
            output / f"observability_review/observability_reviewer_{reviewer}.csv"
            for reviewer in (1, 2)
        ),
    ]
    if run_probes:
        artifact_paths.extend(
            [
                output / "view_probes/view_probe_summary.csv",
                output / "view_probes/view_probe_folds.csv",
                output / "view_probes/view_probe_contract.json",
                *(
                    output / f"view_probes/{target}_probe_scope.csv"
                    for target in ("gender", "usage")
                ),
                *(
                    output / f"view_probes/{target}_{view}_oof.csv"
                    for target in ("gender", "usage")
                    for view in PROBE_VIEWS
                ),
            ]
        )
    if run_neighbourhoods:
        artifact_paths.extend(
            [
                output / "representation_neighbourhoods/neighbourhood_summary.csv",
                output / "representation_neighbourhoods/neighbourhood_slices.csv",
                output / "representation_neighbourhoods/neighbourhood_neighbours.csv.gz",
                output / "representation_neighbourhoods/neighbourhood_contract.json",
                output / "representation_neighbourhoods/representation_provenance.json",
                *(
                    output / f"representation_neighbourhoods/{target}_{role}_scope.csv"
                    for target in ("gender", "usage")
                    for role in ("reference", "query")
                ),
            ]
        )
    if human_complete:
        artifact_paths.extend(human_outputs)
    missing_artifacts = [path for path in artifact_paths if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError(
            "Task 3 EDA did not create required artifacts: "
            f"{[path.as_posix() for path in missing_artifacts]}"
        )
    manifest_rows = []
    for artifact in sorted(set(artifact_paths)):
        relative = artifact.relative_to(output)
        manifest_rows.append(
            {
                "path": relative.as_posix(),
                "sha256": compute_sha256(artifact),
                "bytes": artifact.stat().st_size,
            }
        )
    artifact_manifest = pd.DataFrame(manifest_rows)
    manifest_path = output / "artifact_manifest.csv"
    write_deterministic_csv(artifact_manifest, manifest_path, index=False)
    completion = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split_digest": cv_assignment_digest(splits),
        "audit_contract_hash": audit_contract_hash,
        "artifact_manifest": manifest_path.name,
        "artifact_manifest_sha256": compute_sha256(manifest_path),
        "artifact_count": int(len(artifact_manifest)),
        "automated_audits_complete": True,
        "training_screen_ready": True,
        "training_blockers": [],
        "human_observability_review_complete": human_complete,
        "human_observability_review_status": (
            "complete" if human_complete else "deferred_non_blocking"
        ),
        "deferred_items": (
            [] if human_complete else ["two_independent_human_observability_reviews"]
        ),
        "holdout_targets_read": False,
        "image_scope": "teacher_development_images_only",
        "probe_scope": "balanced development sample; diagnostic only",
        "neighbourhood_scope": "fold_0_queries_against_other_fold_references",
    }
    _write_json(completion, output / "completion.json")
    return {
        "review": review,
        "diagnostics": diagnostics,
        "nuisance": nuisance,
        "fold_shifts": fold_shifts,
        "family": family,
        "probe_summary": probe_summary,
        "probe_folds": probe_folds,
        "neighbourhood_summary": neighbourhood_summary,
        "neighbourhood_slices": neighbourhood_slices,
        "completion": completion,
        "audit_contract": audit_contract,
        "artifact_manifest": artifact_manifest,
    }
