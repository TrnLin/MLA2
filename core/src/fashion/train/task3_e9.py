"""Fold-safe E9 audits and data-selection contracts for Task 3.

This module deliberately has no PyTorch import.  It can build and validate every
E9 pre-run artifact on a small local machine without creating an optimiser.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fashion.data import load_splits

GENDER_CLASSES = ("Boys", "Girls", "Men", "Unisex", "Women")
USAGE_CLASSES = (
    "Casual",
    "Ethnic",
    "Formal",
    "Home",
    "NA",
    "Party",
    "Smart Casual",
    "Sports",
    "Travel",
)

GENDER_SEMANTIC_RULE_VERSION = "gender_semantic_conflicts_v1"
USAGE_EXCEPTION_RULE_VERSION = "usage_article_type_exception_v1"
GENDER_VISUAL_AUDIT_VERSION = "gender_qualitative_review_v2"
E9_AUDIT_SEED = 2753

GENDER_E9_EXPECTED_CONFLICT_ROWS = 305
GENDER_E9_EXPECTED_SOURCE_CLASS_COUNTS = {"Men": 168, "Women": 137}
GENDER_E9_EXPECTED_VALIDATION_FOLD_COUNTS = {
    "0": 63,
    "1": 81,
    "2": 46,
    "3": 42,
    "4": 73,
}
GENDER_E9_EXPECTED_TRAINING_REMOVALS = {
    "0": 242,
    "1": 224,
    "2": 259,
    "3": 263,
    "4": 232,
}
GENDER_E9_EXPECTED_CONFLICT_IDS_HASH = (
    "42faeef3794267065daa3e1aa61aeb5bbc62364f9bcb405013d96d29e96ccdb6"
)
GENDER_E9_EXPECTED_DISTRIBUTION_HASHES = {
    "source_class": "e26928537ce3d63cc90b75f3c1804152064ad1c00d2894060d7463a03ae3c8b4",
    "validation_fold": "2def30fdf7fc2c95adc27727fa5e56a250ee940f39726991ba764f9cce6dd4dd",
    "family": "540b2ef2f282410f7b88919285540934764fc0459df462728c8ecdd19a2a66ae",
    "article_type": "132ced1a27c06d8aa2757104875bb52a349c9ccfd1c10c42a3d3d7c7a2a3bb0f",
}

_STRONG_AGE_PATTERN = re.compile(
    r"\b(?:kid|kids|kidswear|child|children|infants?|bab(?:y|ies)|"
    r"toddlers?|teens?|juniors?|jr)\b",
    flags=re.IGNORECASE,
)
_ADULT_PATTERN = re.compile(
    r"\b(?:men|mens|men's|man|women|womens|women's|lady|ladies|adults?)\b",
    flags=re.IGNORECASE,
)


def stable_payload_hash(payload: object, *, length: int | None = None) -> str:
    """Hash one JSON-compatible payload with stable separators and key order."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest if length is None else digest[:length]


def stable_frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash selected rows after stable column selection and string conversion."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"cannot hash missing columns: {missing}")
    records = (
        frame.loc[:, list(columns)]
        .sort_values(list(columns), kind="mergesort")
        .astype("string")
        .fillna("")
        .to_dict(orient="records")
    )
    return stable_payload_hash(records)


def _normalise_product_name(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .fillna("")
        .str.lower()
        .str.replace("’", "'", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _child_cue(names: pd.Series, cue: str) -> pd.Series:
    direct = names.str.contains(rf"\b(?:{cue}s|{cue}'s)\b", regex=True)
    singular = names.str.contains(rf"\b{cue}\b", regex=True)
    strong_age = names.str.contains(_STRONG_AGE_PATTERN, regex=True)
    return direct | (singular & strong_age)


def _any_child_cue(names: pd.Series, cue: str) -> pd.Series:
    return names.str.contains(rf"\b(?:{cue}|{cue}s|{cue}'s)\b", regex=True)


def gender_semantic_conflicts(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate the conservative Men/Women-versus-child-name conflict rule.

    A direct possessive/plural child cue is sufficient.  A singular ``boy`` or
    ``girl`` needs a second strong age cue.  Opposite-child and explicit-adult
    cues veto the rule.  In particular, ``little`` alone is intentionally not a
    strong age cue because it is common marketing language.
    """
    required = {"gender", "productDisplayName"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"gender conflict rule needs columns: {missing}")
    names = _normalise_product_name(frame["productDisplayName"])
    adult = names.str.contains(_ADULT_PATTERN, regex=True)
    boy = _child_cue(names, "boy")
    girl = _child_cue(names, "girl")
    any_boy = _any_child_cue(names, "boy")
    any_girl = _any_child_cue(names, "girl")
    men_conflict = frame["gender"].eq("Men") & boy & ~any_girl & ~adult
    women_conflict = frame["gender"].eq("Women") & girl & ~any_boy & ~adult

    result = frame.copy()
    result["e9_semantic_conflict"] = men_conflict | women_conflict
    result["e9_name_implied_gender"] = np.select(
        [men_conflict, women_conflict], ["Boys", "Girls"], default=""
    )
    result["e9_semantic_conflict_kind"] = np.select(
        [men_conflict, women_conflict],
        ["Men_with_boy_cue", "Women_with_girl_cue"],
        default="",
    )
    result["e9_semantic_rule_version"] = GENDER_SEMANTIC_RULE_VERSION
    return result


def gender_clean_child_evidence_mask(frame: pd.DataFrame) -> pd.Series:
    """Select official child rows whose names give one clean matching child cue."""
    required = {"gender", "productDisplayName"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"clean child-evidence rule needs columns: {missing}")
    names = _normalise_product_name(frame["productDisplayName"])
    adult = names.str.contains(_ADULT_PATTERN, regex=True)
    boy = _child_cue(names, "boy")
    girl = _child_cue(names, "girl")
    any_boy = _any_child_cue(names, "boy")
    any_girl = _any_child_cue(names, "girl")
    clean_boys = frame["gender"].eq("Boys") & boy & ~any_girl & ~adult
    clean_girls = frame["gender"].eq("Girls") & girl & ~any_boy & ~adult
    return clean_boys | clean_girls


def prepare_gender_e9_training(
    training: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Remove only locked semantic conflicts from one fold-training complement."""
    annotated = gender_semantic_conflicts(training)
    excluded = annotated.loc[annotated["e9_semantic_conflict"]].copy()
    selected = annotated.loc[~annotated["e9_semantic_conflict"]].copy()
    if selected.empty:
        raise ValueError("the gender semantic filter removed every training row")
    id_column = "id" if "id" in excluded else excluded.columns[0]
    excluded_ids = sorted(int(value) for value in excluded[id_column].tolist())
    metadata: dict[str, object] = {
        "rule_version": GENDER_SEMANTIC_RULE_VERSION,
        "rule_hash": stable_payload_hash(gender_semantic_rule_contract()),
        "before_rows": int(len(training)),
        "after_rows": int(len(selected)),
        "excluded_rows": int(len(excluded)),
        "excluded_ids": excluded_ids,
        "excluded_ids_hash": stable_payload_hash(excluded_ids),
        "validation_unchanged": True,
    }
    return selected.reset_index(drop=True), excluded.reset_index(drop=True), metadata


def gender_semantic_rule_contract() -> dict[str, object]:
    """Return the human-readable, hashable rule declaration."""
    return {
        "version": GENDER_SEMANTIC_RULE_VERSION,
        "eligible_official_labels": ["Men", "Women"],
        "direct_child_cues": ["boys", "boy's", "girls", "girl's"],
        "singular_child_cues_require_strong_age_term": ["boy", "girl"],
        "strong_age_terms": [
            "kid(s)",
            "kidswear",
            "child(ren)",
            "infant(s)",
            "baby/babies",
            "toddler(s)",
            "teen(s)",
            "junior(s)",
            "jr",
        ],
        "vetoes": ["opposite child cue", "explicit adult cue"],
        "little_alone_is_strong": False,
        "model_inputs_unchanged": True,
        "validation_rows_unchanged": True,
    }


def gender_conflict_audit(splits: pd.DataFrame) -> dict[str, object]:
    """Build the fixed-development audit behind the gender E9 proposal."""
    development = splits.loc[
        splits["partition"].eq("development") & splits["has_gender_label"]
    ].copy()
    annotated = gender_semantic_conflicts(development)
    conflicts = annotated.loc[annotated["e9_semantic_conflict"]].copy()
    fold_counts = (
        conflicts.groupby("cv_fold", dropna=False).size().reindex(range(5), fill_value=0)
    )
    removals = pd.DataFrame(
        {
            "validation_fold": range(5),
            "conflicts_in_validation": [int(fold_counts.loc[fold]) for fold in range(5)],
            "conflicts_removed_from_training": [
                int(len(conflicts) - fold_counts.loc[fold]) for fold in range(5)
            ],
        }
    )
    distributions = {
        "source_class": conflicts.groupby("gender", dropna=False)
        .size()
        .rename("rows")
        .reset_index(),
        "validation_fold": conflicts.groupby("cv_fold", dropna=False)
        .size()
        .rename("rows")
        .reset_index(),
        "family": conflicts.groupby("product_family_group", dropna=False)
        .size()
        .rename("rows")
        .reset_index(),
        "article_type": conflicts.groupby("articleType", dropna=False)
        .size()
        .rename("rows")
        .reset_index(),
    }
    distribution_columns = {
        "source_class": ["gender", "rows"],
        "validation_fold": ["cv_fold", "rows"],
        "family": ["product_family_group", "rows"],
        "article_type": ["articleType", "rows"],
    }
    conflict_ids = sorted(int(value) for value in conflicts["id"])
    summary = {
        "rule": gender_semantic_rule_contract(),
        "development_rows": int(len(development)),
        "conflict_rows": int(len(conflicts)),
        "source_class_counts": {
            str(key): int(value) for key, value in conflicts.groupby("gender").size().items()
        },
        "validation_fold_counts": {
            str(int(key)): int(value) for key, value in fold_counts.items()
        },
        "training_removals": {
            str(int(row.validation_fold)): int(row.conflicts_removed_from_training)
            for row in removals.itertuples(index=False)
        },
        "unique_families": int(conflicts["product_family_group"].nunique()),
        "contract_hash": stable_payload_hash(gender_semantic_rule_contract()),
        "conflict_ids_hash": stable_payload_hash(conflict_ids),
        "distribution_hashes": {
            name: stable_frame_hash(frame, distribution_columns[name])
            for name, frame in distributions.items()
        },
        "conflict_rows_hash": stable_frame_hash(
            conflicts, ["id", "gender", "e9_name_implied_gender", "cv_fold"]
        ),
    }
    return {
        "annotated": annotated,
        "conflicts": conflicts,
        "removals": removals,
        "summary": summary,
        "conflict_ids": conflict_ids,
        "distributions": distributions,
    }


def verify_gender_e9_deterministic_audit(
    audit: Mapping[str, object],
) -> dict[str, object]:
    """Require the frozen 305-row audit before any E9G training call."""
    summary = audit.get("summary", audit)
    if not isinstance(summary, Mapping):
        raise TypeError("gender deterministic audit must contain a summary mapping")
    rule = summary.get("rule", {})
    checks = {
        "rule_version": isinstance(rule, Mapping)
        and rule.get("version") == GENDER_SEMANTIC_RULE_VERSION,
        "conflict_rows": summary.get("conflict_rows")
        == GENDER_E9_EXPECTED_CONFLICT_ROWS,
        "source_class_counts": summary.get("source_class_counts")
        == GENDER_E9_EXPECTED_SOURCE_CLASS_COUNTS,
        "validation_fold_counts": summary.get("validation_fold_counts")
        == GENDER_E9_EXPECTED_VALIDATION_FOLD_COUNTS,
        "training_removals": summary.get("training_removals")
        == GENDER_E9_EXPECTED_TRAINING_REMOVALS,
        "unique_families": summary.get("unique_families") == 200,
        "conflict_ids_hash": summary.get("conflict_ids_hash")
        == GENDER_E9_EXPECTED_CONFLICT_IDS_HASH,
        "distribution_hashes": summary.get("distribution_hashes")
        == GENDER_E9_EXPECTED_DISTRIBUTION_HASHES,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Gender E9 deterministic audit changed: {failed}")
    return {
        "verified": True,
        "checks": checks,
        "rule_version": GENDER_SEMANTIC_RULE_VERSION,
        "conflict_rows": GENDER_E9_EXPECTED_CONFLICT_ROWS,
        "conflict_ids_hash": GENDER_E9_EXPECTED_CONFLICT_IDS_HASH,
        "training_removals": GENDER_E9_EXPECTED_TRAINING_REMOVALS,
        "human_rating_gate_required": False,
    }


def _target_development(splits: pd.DataFrame, target: str) -> pd.DataFrame:
    validity = f"has_{target}_label"
    required = {"partition", "cv_fold", target, validity, "articleType", "id"}
    missing = sorted(required.difference(splits.columns))
    if missing:
        raise ValueError(f"split table lacks columns for {target}: {missing}")
    result = splits.loc[splits["partition"].eq("development") & splits[validity]].copy()
    if result[target].isna().any() or result["cv_fold"].isna().any():
        raise ValueError(f"valid development {target} rows contain missing target/fold values")
    return result


def build_usage_article_type_contract(
    training: pd.DataFrame,
    *,
    classes: Sequence[str] = USAGE_CLASSES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Derive fold-training ArticleType usual/exception groups and mean-one factors."""
    required = {"id", "usage", "articleType"}
    missing = sorted(required.difference(training.columns))
    if missing:
        raise ValueError(f"usage group contract needs columns: {missing}")
    if training["articleType"].isna().any():
        raise ValueError("usage training rows contain an unsupported ArticleType")
    class_order = tuple(str(value) for value in classes)
    unknown = sorted(set(training["usage"]).difference(class_order))
    if unknown:
        raise ValueError(f"usage rows contain labels outside the fixed map: {unknown}")

    counts = (
        training.groupby(["articleType", "usage"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=class_order, fill_value=0)
        .sort_index()
    )
    mapping_rows: list[dict[str, object]] = []
    for article_type, row in counts.iterrows():
        values = row.to_numpy(dtype=np.int64)
        support = int(values.sum())
        maximum = int(values.max())
        tied = [class_order[index] for index in np.flatnonzero(values == maximum)]
        usual = tied[0]
        mapping_rows.append(
            {
                "articleType": str(article_type),
                "usual_usage": usual,
                "support": support,
                "usual_count": maximum,
                "exception_count": support - maximum,
                "usual_share": maximum / support,
                "is_tie": len(tied) > 1,
                "tied_usages": "|".join(tied),
                "tie_break": "fixed_label_map_order",
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    usual_lookup = mapping.set_index("articleType")["usual_usage"]
    annotated = training.copy()
    annotated["e9_usual_usage"] = annotated["articleType"].map(usual_lookup)
    if annotated["e9_usual_usage"].isna().any():
        raise AssertionError("a training ArticleType did not map to its own usual label")
    annotated["e9_usage_group"] = np.where(
        annotated["usage"].eq(annotated["e9_usual_usage"]), "usual", "exception"
    )
    group_counts = annotated["e9_usage_group"].value_counts().reindex(
        ["usual", "exception"], fill_value=0
    )
    if (group_counts == 0).any():
        raise ValueError("both usage groups need positive fold-training support")
    factors = len(annotated) / (2.0 * group_counts.astype(float))
    annotated["e9_group_factor"] = annotated["e9_usage_group"].map(factors).astype(float)
    weighted_mean = float(annotated["e9_group_factor"].mean())
    if not math.isclose(weighted_mean, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("usage group factors do not have a row-weighted mean of one")
    metadata: dict[str, object] = {
        "rule_version": USAGE_EXCEPTION_RULE_VERSION,
        "class_tie_break_order": list(class_order),
        "group_counts": {str(key): int(value) for key, value in group_counts.items()},
        "group_factors": {str(key): float(value) for key, value in factors.items()},
        "row_weighted_group_factor_mean": weighted_mean,
        "mapping_hash": stable_frame_hash(
            mapping,
            [
                "articleType",
                "usual_usage",
                "support",
                "usual_count",
                "exception_count",
                "is_tie",
                "tied_usages",
            ],
        ),
        "mapping_rows": int(len(mapping)),
        "tie_rows": int(mapping["is_tie"].sum()),
        "model_inputs_unchanged": True,
        "validation_rows_unchanged": True,
    }
    return annotated, mapping, metadata


def annotate_usage_validation(
    validation: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    """Apply one fold-training usage map without inventing zero-support labels."""
    required = {"articleType", "usual_usage"}
    if not required.issubset(mapping.columns):
        raise ValueError("usage mapping lacks articleType/usual_usage")
    lookup = mapping.set_index("articleType")["usual_usage"]
    result = validation.copy()
    result["e9_usual_usage"] = result["articleType"].map(lookup)
    result["e9_usage_group"] = np.select(
        [
            result["e9_usual_usage"].isna(),
            result["usage"].eq(result["e9_usual_usage"]),
        ],
        ["unsupported", "usual"],
        default="exception",
    )
    return result


def usage_exception_diagnostics(
    splits: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    classes: Sequence[str] = USAGE_CLASSES,
) -> dict[str, object]:
    """Measure E2/E9 usage errors with a fold-cross-fitted ArticleType map."""
    development = _target_development(splits, "usage")
    prediction_columns = {"id", "predicted_index"}
    missing = sorted(prediction_columns.difference(predictions.columns))
    if missing:
        raise ValueError(f"usage predictions lack columns: {missing}")
    predicted = predictions.loc[:, ["id", "predicted_index"]].copy()
    if predicted["id"].duplicated().any():
        raise ValueError("usage predictions contain duplicate IDs")
    if set(predicted["id"]) != set(development["id"]):
        raise ValueError("usage OOF IDs do not exactly match valid development rows")
    predicted_indices = pd.to_numeric(predicted["predicted_index"], errors="raise").astype(int)
    if not predicted_indices.between(0, len(classes) - 1).all():
        raise ValueError("usage predictions contain an out-of-range class index")
    predicted["predicted_label"] = predicted_indices.map(dict(enumerate(classes)))

    annotated_parts: list[pd.DataFrame] = []
    mapping_parts: list[pd.DataFrame] = []
    contract_rows: list[dict[str, object]] = []
    for fold in range(5):
        training = development.loc[development["cv_fold"].ne(fold)].copy()
        validation = development.loc[development["cv_fold"].eq(fold)].copy()
        _, mapping, metadata = build_usage_article_type_contract(training, classes=classes)
        mapped = annotate_usage_validation(validation, mapping)
        mapped["validation_fold"] = fold
        mapping = mapping.copy()
        mapping["validation_fold"] = fold
        annotated_parts.append(mapped)
        mapping_parts.append(mapping)
        contract_rows.append(
            {
                "validation_fold": fold,
                "training_rows": len(training),
                "usual_training_rows": metadata["group_counts"]["usual"],
                "exception_training_rows": metadata["group_counts"]["exception"],
                "usual_group_factor": metadata["group_factors"]["usual"],
                "exception_group_factor": metadata["group_factors"]["exception"],
                "mapping_hash": metadata["mapping_hash"],
                "tie_article_types": metadata["tie_rows"],
            }
        )
    annotated = pd.concat(annotated_parts, ignore_index=True).merge(
        predicted, on="id", validate="one_to_one"
    )
    annotated["is_error"] = annotated["predicted_label"].ne(annotated["usage"])
    annotated["predicted_usual_shortcut"] = annotated["predicted_label"].eq(
        annotated["e9_usual_usage"]
    )
    group_summary = (
        annotated.groupby("e9_usage_group", observed=True)
        .agg(rows=("id", "size"), errors=("is_error", "sum"))
        .reset_index()
    )
    group_summary["error_rate"] = group_summary["errors"] / group_summary["rows"]
    error_exception = annotated.loc[
        annotated["e9_usage_group"].eq("exception") & annotated["is_error"]
    ]
    shortcut_rate = float(error_exception["predicted_usual_shortcut"].mean())

    family_label_counts = annotated.groupby("product_family_group")["usage"].transform(
        "nunique"
    )
    mixed = family_label_counts > 1
    mixed_accuracy = float((~annotated.loc[mixed, "is_error"]).mean())
    group_lookup = group_summary.set_index("e9_usage_group")
    summary = {
        "rule_version": USAGE_EXCEPTION_RULE_VERSION,
        "usual_rows": int(group_lookup.loc["usual", "rows"]),
        "usual_errors": int(group_lookup.loc["usual", "errors"]),
        "usual_error_rate": float(group_lookup.loc["usual", "error_rate"]),
        "exception_rows": int(group_lookup.loc["exception", "rows"]),
        "exception_errors": int(group_lookup.loc["exception", "errors"]),
        "exception_error_rate": float(group_lookup.loc["exception", "error_rate"]),
        "wrong_exception_shortcut_count": int(
            error_exception["predicted_usual_shortcut"].sum()
        ),
        "wrong_exception_shortcut_rate": shortcut_rate,
        "unsupported_rows": int(group_lookup.loc["unsupported", "rows"]),
        "unsupported_errors": int(group_lookup.loc["unsupported", "errors"]),
        "mixed_label_family_rows": int(mixed.sum()),
        "mixed_label_family_accuracy": mixed_accuracy,
        "tie_break": "fixed_label_map_order",
        "validation_rows_unchanged": True,
    }
    return {
        "annotated": annotated,
        "mappings": pd.concat(mapping_parts, ignore_index=True),
        "fold_contracts": pd.DataFrame(contract_rows),
        "group_summary": group_summary,
        "summary": summary,
    }


def _gender_article_type_majorities(
    development: pd.DataFrame,
    classes: Sequence[str] = GENDER_CLASSES,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in range(5):
        training = development.loc[development["cv_fold"].ne(fold)]
        validation = development.loc[development["cv_fold"].eq(fold)].copy()
        counts = (
            training.groupby(["articleType", "gender"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=classes, fill_value=0)
        )
        majority = counts.apply(lambda row: classes[int(row.to_numpy().argmax())], axis=1)
        validation["article_type_majority_gender"] = validation["articleType"].map(majority)
        validation["article_type_agrees_official"] = validation["gender"].eq(
            validation["article_type_majority_gender"]
        )
        parts.append(validation)
    return pd.concat(parts, ignore_index=True)


def _prediction_frame(value: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.read_csv(value, keep_default_na=False)


def _priority(seed: int, *parts: object) -> str:
    value = ":".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unique_balanced_pick(
    candidates: pd.DataFrame,
    *,
    count: int,
    seed: int,
    stratum: str,
    class_name: str,
    balance_columns: Sequence[str],
    used_families: set[str],
) -> pd.DataFrame:
    available = candidates.loc[
        ~candidates["product_family_group"].astype(str).isin(used_families)
    ].copy()
    available["_priority"] = [
        _priority(seed, stratum, class_name, value) for value in available["id"]
    ]
    available = available.sort_values("_priority", kind="mergesort")
    if len(available.drop_duplicates("product_family_group")) < count:
        raise ValueError(f"not enough unique families for {stratum}/{class_name}")
    if balance_columns:
        available["_balance_bucket"] = available.loc[:, list(balance_columns)].astype(str).agg(
            "|".join, axis=1
        )
    else:
        available["_balance_bucket"] = "all"
    buckets = sorted(available["_balance_bucket"].unique())
    base_quota, extra = divmod(count, len(buckets))
    selected_indices: list[int] = []
    selected_families: set[str] = set()
    for position, bucket in enumerate(buckets):
        quota = base_quota + int(position < extra)
        rows = available.loc[available["_balance_bucket"].eq(bucket)]
        for index, row in rows.iterrows():
            family = str(row["product_family_group"])
            if family in selected_families:
                continue
            selected_indices.append(index)
            selected_families.add(family)
            if sum(
                available.loc[selected_indices, "_balance_bucket"].eq(bucket)
            ) >= quota:
                break
    if len(selected_indices) < count:
        for index, row in available.iterrows():
            family = str(row["product_family_group"])
            if index in selected_indices or family in selected_families:
                continue
            selected_indices.append(index)
            selected_families.add(family)
            if len(selected_indices) == count:
                break
    if len(selected_indices) != count:
        raise ValueError(f"balanced selection failed for {stratum}/{class_name}")
    selected = available.loc[selected_indices].copy()
    used_families.update(selected["product_family_group"].astype(str))
    return selected.drop(columns=["_priority", "_balance_bucket"])


def select_gender_visual_audit(
    splits: pd.DataFrame,
    model_predictions: Mapping[str, str | Path | pd.DataFrame],
    *,
    seed: int = E9_AUDIT_SEED,
) -> pd.DataFrame:
    """Select 450 blinded rows: 30 per official class in each of three strata."""
    required_models = ("e1", "e4", "e5", "e6", "e7", "e8")
    if tuple(model_predictions) != required_models:
        raise ValueError(f"gender audit models must be ordered exactly as {required_models}")
    development = _target_development(splits, "gender")
    audit = _gender_article_type_majorities(development)
    for model_name, value in model_predictions.items():
        predictions = _prediction_frame(value)
        required = {"id", "true_label", "predicted_label"}
        missing = sorted(required.difference(predictions.columns))
        if missing:
            raise ValueError(f"{model_name} predictions lack columns: {missing}")
        if predictions["id"].duplicated().any():
            raise ValueError(f"{model_name} predictions contain duplicate IDs")
        if set(predictions["id"]) != set(development["id"]):
            raise ValueError(f"{model_name} OOF IDs do not match gender development rows")
        truth = predictions.set_index("id")["true_label"]
        expected = audit.set_index("id")["gender"]
        if not truth.reindex(expected.index).equals(expected):
            raise ValueError(f"{model_name} OOF labels disagree with the split table")
        audit = audit.merge(
            predictions.loc[:, ["id", "predicted_label"]].rename(
                columns={"predicted_label": f"prediction_{model_name}"}
            ),
            on="id",
            validate="one_to_one",
        )
    prediction_columns = [f"prediction_{name}" for name in required_models]
    correct = audit[prediction_columns].eq(audit["gender"], axis=0)
    audit["all_models_correct"] = correct.all(axis=1)
    audit["all_models_wrong"] = (~correct).all(axis=1)
    audit["model_sensitive"] = correct.any(axis=1) & (~correct).any(axis=1)
    audit["persistent_error_pattern"] = np.where(
        audit[prediction_columns].nunique(axis=1).eq(1),
        "stable_same_error",
        "changing_error",
    )
    audit["audit_stratum"] = np.select(
        [audit["all_models_wrong"], audit["model_sensitive"], audit["all_models_correct"]],
        ["persistent_error", "model_sensitive_error", "stable_control"],
        default="unused",
    )

    used_families: set[str] = set()
    selections: list[pd.DataFrame] = []
    for stratum in ("persistent_error", "model_sensitive_error", "stable_control"):
        for class_name in GENDER_CLASSES:
            candidates = audit.loc[
                audit["audit_stratum"].eq(stratum) & audit["gender"].eq(class_name)
            ]
            balance = (
                ["persistent_error_pattern", "article_type_agrees_official"]
                if stratum == "persistent_error"
                else ["article_type_agrees_official"]
            )
            selections.append(
                _unique_balanced_pick(
                    candidates,
                    count=30,
                    seed=seed,
                    stratum=stratum,
                    class_name=class_name,
                    balance_columns=balance,
                    used_families=used_families,
                )
            )
    selected = pd.concat(selections, ignore_index=True)
    if len(selected) != 450 or selected["id"].duplicated().any():
        raise AssertionError("gender visual audit is not exactly 450 unique products")
    if selected["product_family_group"].duplicated().any():
        raise AssertionError("gender visual audit contains a repeated product family")
    counts = selected.groupby(["audit_stratum", "gender"]).size()
    if not (counts == 30).all() or len(counts) != 15:
        raise AssertionError("gender audit class/stratum cells are not all size 30")
    selected["_shuffle"] = [_priority(seed, "blind", value) for value in selected["id"]]
    selected = selected.sort_values("_shuffle", kind="mergesort").reset_index(drop=True)
    selected["audit_code"] = [f"GVA-{index:04d}" for index in range(1, len(selected) + 1)]
    return selected.drop(columns="_shuffle")


def nominal_krippendorff_alpha(ratings: pd.DataFrame) -> float:
    """Compute nominal Krippendorff alpha from item/rater/category rows."""
    required = {"audit_code", "rater_id", "top1_choice"}
    missing = sorted(required.difference(ratings.columns))
    if missing:
        raise ValueError(f"ratings lack columns for alpha: {missing}")
    categories = sorted(ratings["top1_choice"].unique())
    category_index = {category: index for index, category in enumerate(categories)}
    coincidence = np.zeros((len(categories), len(categories)), dtype=float)
    for _, group in ratings.groupby("audit_code", sort=False):
        values = group["top1_choice"].tolist()
        if len(values) < 2:
            continue
        weight = 1.0 / (len(values) - 1)
        for left_index, left in enumerate(values):
            for right_index, right in enumerate(values):
                if left_index != right_index:
                    coincidence[category_index[left], category_index[right]] += weight
    total = float(coincidence.sum())
    if total <= 1:
        raise ValueError("not enough ratings to compute Krippendorff alpha")
    observed_disagreement = float(coincidence.sum() - np.trace(coincidence)) / total
    marginals = coincidence.sum(axis=0)
    expected_disagreement = float(total**2 - np.square(marginals).sum()) / (
        total * (total - 1)
    )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def _parse_boolean(value: object) -> bool:
    normalised = str(value).strip().lower()
    if normalised in {"true", "1", "yes", "y"}:
        return True
    if normalised in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid insufficient_evidence value: {value!r}")


def validate_gender_visual_ratings(
    ratings: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Reject blank, duplicate, incomplete, or out-of-vocabulary audit ratings."""
    required = {
        "audit_code",
        "rater_id",
        "top1_choice",
        "insufficient_evidence",
        "confidence",
    }
    missing = sorted(required.difference(ratings.columns))
    if missing:
        raise ValueError(f"ratings lack columns: {missing}")
    clean = ratings.loc[:, list(required)].copy()
    for column in ("audit_code", "rater_id", "top1_choice", "insufficient_evidence", "confidence"):
        clean[column] = clean[column].astype("string").fillna("").str.strip()
    if (clean == "").any().any():
        raise ValueError("gender visual ratings are still blank or incomplete")
    if clean.duplicated(["audit_code", "rater_id"]).any():
        raise ValueError("one rater rated the same audit item more than once")
    expected_codes = set(manifest["audit_code"])
    if set(clean["audit_code"]) != expected_codes:
        raise ValueError("rated audit codes do not exactly match the locked manifest")
    unknown = sorted(set(clean["top1_choice"]).difference(GENDER_CLASSES))
    if unknown:
        raise ValueError(f"ratings contain unknown top-1 choices: {unknown}")
    clean["insufficient_evidence"] = clean["insufficient_evidence"].map(_parse_boolean)
    clean["confidence"] = pd.to_numeric(clean["confidence"], errors="raise").astype(int)
    if not clean["confidence"].between(1, 5).all():
        raise ValueError("rating confidence must be an integer from 1 to 5")
    rater_counts = clean.groupby("audit_code")["rater_id"].nunique()
    if (rater_counts < 3).any():
        raise ValueError("every audit item needs at least three distinct raters")
    return clean.sort_values(["audit_code", "rater_id"], kind="mergesort").reset_index(drop=True)


def _majority_choice(values: pd.Series) -> str:
    counts = values.value_counts()
    maximum = counts.max()
    tied = set(counts.index[counts.eq(maximum)])
    return next(value for value in GENDER_CLASSES if value in tied)


def _cluster_interval(
    items: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    seed: int,
    repetitions: int,
) -> dict[str, float]:
    families = np.asarray(sorted(items["product_family_group"].astype(str).unique()))
    if len(families) < 2:
        raise ValueError("family bootstrap needs at least two product families")
    groups = {
        family: items.loc[items["product_family_group"].astype(str).eq(family)]
        for family in families
    }
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        sampled = generator.choice(families, size=len(families), replace=True)
        draw = pd.concat([groups[str(family)] for family in sampled], ignore_index=True)
        value = float(statistic(draw))
        if np.isfinite(value):
            values.append(value)
    if len(values) < repetitions * 0.9:
        raise ValueError("too many invalid family-bootstrap draws")
    return {
        "point": float(statistic(items)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "repetitions": int(repetitions),
        "families": int(len(families)),
    }


def analyze_gender_visual_audit(
    ratings: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    seed: int = E9_AUDIT_SEED,
    bootstrap_repetitions: int = 2000,
) -> dict[str, object]:
    """Summarize optional research ratings; this never controls E9 training."""
    clean = validate_gender_visual_ratings(ratings, manifest)
    item_rows: list[dict[str, object]] = []
    for audit_code, group in clean.groupby("audit_code", sort=False):
        counts = group["top1_choice"].value_counts()
        item_rows.append(
            {
                "audit_code": audit_code,
                "majority_choice": _majority_choice(group["top1_choice"]),
                "majority_votes": int(counts.max()),
                "majority_tie": int((counts == counts.max()).sum()) > 1,
                "insufficient_votes": int(group["insufficient_evidence"].sum()),
                "mean_confidence": float(group["confidence"].mean()),
                "rater_count": int(group["rater_id"].nunique()),
            }
        )
    items = manifest.merge(pd.DataFrame(item_rows), on="audit_code", validate="one_to_one")
    items["official_agreement"] = items["majority_choice"].eq(items["gender"])
    items["insufficient_two_plus"] = items["insufficient_votes"].ge(2)
    alpha = nominal_krippendorff_alpha(clean)

    slice_summary = (
        items.groupby(["audit_stratum", "gender"], observed=True)
        .agg(
            rows=("audit_code", "size"),
            official_agreement=("official_agreement", "mean"),
            insufficient_two_plus=("insufficient_two_plus", "mean"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .reset_index()
    )
    persistent = items.loc[items["audit_stratum"].eq("persistent_error")]
    control = items.loc[items["audit_stratum"].eq("stable_control")]
    persistent_by_class = (
        persistent.groupby("gender")["official_agreement"]
        .mean()
        .reindex(GENDER_CLASSES)
        .to_dict()
    )
    article_summary = (
        persistent.groupby("article_type_agrees_official", observed=True)["official_agreement"]
        .agg(["size", "mean"])
        .reset_index()
        .rename(columns={"size": "rows", "mean": "official_agreement"})
    )

    stable_minus_persistent = _cluster_interval(
        items,
        lambda frame: float(
            frame.loc[frame["audit_stratum"].eq("stable_control"), "official_agreement"].mean()
            - frame.loc[
                frame["audit_stratum"].eq("persistent_error"), "official_agreement"
            ].mean()
        ),
        seed=seed + 1,
        repetitions=bootstrap_repetitions,
    )
    article_agreement_minus_disagreement = _cluster_interval(
        persistent,
        lambda frame: float(
            frame.loc[frame["article_type_agrees_official"], "official_agreement"].mean()
            - frame.loc[~frame["article_type_agrees_official"], "official_agreement"].mean()
        ),
        seed=seed + 2,
        repetitions=bootstrap_repetitions,
    )
    persistent_interval = _cluster_interval(
        persistent,
        lambda frame: float(frame["official_agreement"].mean()),
        seed=seed + 3,
        repetitions=bootstrap_repetitions,
    )

    persistent_rate = float(persistent["official_agreement"].mean())
    control_rate = float(control["official_agreement"].mean())
    insufficient_rate = float(items["insufficient_two_plus"].mean())
    article_gap = float(article_agreement_minus_disagreement["point"])
    supported_checks = {
        "persistent_official_agreement_at_most_0_65": persistent_rate <= 0.65,
        "girls_or_unisex_at_most_0_55": (
            float(persistent_by_class["Girls"]) <= 0.55
            or float(persistent_by_class["Unisex"]) <= 0.55
        ),
        "stable_control_gap_at_least_0_15": stable_minus_persistent["point"] >= 0.15,
        "stable_control_gap_lower_95_above_zero": stable_minus_persistent["lower_95"] > 0.0,
        "low_alpha_or_high_insufficient": alpha < 0.67 or insufficient_rate >= 0.25,
        "article_disagreement_at_least_0_15_worse": article_gap >= 0.15,
    }
    falsified_checks = {
        "persistent_official_agreement_above_0_80": persistent_rate > 0.80,
        "alpha_above_0_80": alpha > 0.80,
        "every_persistent_class_above_0_70": all(
            float(value) > 0.70 for value in persistent_by_class.values()
        ),
        "insufficient_below_0_10": insufficient_rate < 0.10,
        "article_gap_below_0_10": article_gap < 0.10,
    }
    if all(supported_checks.values()):
        decision = "supported"
    elif all(falsified_checks.values()):
        decision = "falsified"
    else:
        decision = "inconclusive"

    confusion = pd.crosstab(
        items["gender"], items["majority_choice"], dropna=False
    ).reindex(index=GENDER_CLASSES, columns=GENDER_CLASSES, fill_value=0)
    child_adult_confusions = {
        "Boys_to_Men": int(confusion.loc["Boys", "Men"]),
        "Girls_to_Women": int(confusion.loc["Girls", "Women"]),
        "Men_to_Boys": int(confusion.loc["Men", "Boys"]),
        "Women_to_Girls": int(confusion.loc["Women", "Girls"]),
    }
    return {
        "version": GENDER_VISUAL_AUDIT_VERSION,
        "decision": decision,
        "items": int(len(items)),
        "ratings": int(len(clean)),
        "distinct_raters": int(clean["rater_id"].nunique()),
        "krippendorff_alpha_nominal": float(alpha),
        "persistent_official_agreement": persistent_rate,
        "stable_control_official_agreement": control_rate,
        "persistent_by_class": {
            str(key): float(value) for key, value in persistent_by_class.items()
        },
        "insufficient_two_plus_item_rate": insufficient_rate,
        "majority_tie_rate": float(items["majority_tie"].mean()),
        "persistent_interval": persistent_interval,
        "stable_minus_persistent_interval": stable_minus_persistent,
        "article_agreement_minus_disagreement_interval": article_agreement_minus_disagreement,
        "supported_checks": supported_checks,
        "falsified_checks": falsified_checks,
        "child_adult_confusions": child_adult_confusions,
        "confusion_matrix": confusion.to_dict(orient="index"),
        "slice_summary": slice_summary.to_dict(orient="records"),
        "persistent_article_type_summary": article_summary.to_dict(orient="records"),
    }


def write_task3_e9_prerun_evidence(
    *,
    splits_path: str | Path,
    usage_prediction_path: str | Path | pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, object]:
    """Write every deterministic E9 audit artifact without starting training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(splits_path)

    gender = gender_conflict_audit(splits)
    gender_verification = verify_gender_e9_deterministic_audit(gender)
    gender_dir = output_dir / "gender_semantic_filter"
    gender_dir.mkdir(parents=True, exist_ok=True)
    gender["conflicts"].sort_values("id").to_csv(
        gender_dir / "conflict_rows.csv", index=False
    )
    gender["removals"].to_csv(gender_dir / "fold_training_removals.csv", index=False)
    for distribution_name, name in (
        ("source_class", "by_source_class.csv"),
        ("validation_fold", "by_validation_fold.csv"),
        ("family", "by_family.csv"),
        ("article_type", "by_article_type.csv"),
    ):
        gender["distributions"][distribution_name].to_csv(gender_dir / name, index=False)
    (gender_dir / "excluded_ids.json").write_text(
        json.dumps(
            {
                "rule_version": GENDER_SEMANTIC_RULE_VERSION,
                "count": len(gender["conflict_ids"]),
                "ids": gender["conflict_ids"],
                "ids_hash": gender["summary"]["conflict_ids_hash"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (gender_dir / "deterministic_contract_verification.json").write_text(
        json.dumps(gender_verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (gender_dir / "summary.json").write_text(
        json.dumps(gender["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    usage_predictions = _prediction_frame(usage_prediction_path)
    usage = usage_exception_diagnostics(splits, usage_predictions)
    usage_dir = output_dir / "usage_exception_balance"
    usage_dir.mkdir(parents=True, exist_ok=True)
    usage["annotated"].sort_values("id").to_csv(
        usage_dir / "cross_fitted_rows.csv", index=False
    )
    usage["mappings"].to_csv(usage_dir / "fold_article_type_mappings.csv", index=False)
    usage["mappings"].loc[usage["mappings"]["is_tie"]].to_csv(
        usage_dir / "tie_article_types.csv", index=False
    )
    usage["annotated"].loc[
        usage["annotated"]["e9_usage_group"].eq("unsupported")
    ].to_csv(usage_dir / "unsupported_validation_rows.csv", index=False)
    usage["fold_contracts"].to_csv(usage_dir / "fold_training_contracts.csv", index=False)
    usage["group_summary"].to_csv(usage_dir / "group_error_summary.csv", index=False)
    (usage_dir / "summary.json").write_text(
        json.dumps(usage["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = {
        "gender_semantic_filter": gender["summary"],
        "gender_deterministic_contract": gender_verification,
        "usage_exception_balance": usage["summary"],
        "optimizer_steps": 0,
        "training_started": False,
    }
    (output_dir / "e9_prerun_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
