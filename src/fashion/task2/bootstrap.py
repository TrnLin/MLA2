"""Paired product-family bootstrap analysis for frozen Task 2 finalists."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from fashion.config import TASK2_CONFIG_DIR
from fashion.task2.slices import CandidateOOFPack
from fashion.task2.stability import (
    C2_PRIMARY_EXPERIMENT_ID,
    C2_STABILITY_EXPERIMENT_ID,
    G5_SEED,
    I2_PRIMARY_EXPERIMENT_ID,
    I2_STABILITY_EXPERIMENT_ID,
)
from fashion.train.artifacts import atomic_write_bytes, canonical_sha256
from fashion.train.metrics import SEASON_LABELS, paired_group_bootstrap

PAIRED_BOOTSTRAP_CONFIG_PATH = TASK2_CONFIG_DIR / "g6_paired_group_bootstrap.json"
EXPECTED_PAIR_IDENTITIES = (
    (
        "primary_interval",
        2753,
        C2_PRIMARY_EXPERIMENT_ID,
        I2_PRIMARY_EXPERIMENT_ID,
    ),
    (
        "stability_sensitivity",
        G5_SEED,
        C2_STABILITY_EXPERIMENT_ID,
        I2_STABILITY_EXPERIMENT_ID,
    ),
)


@dataclass(frozen=True)
class BootstrapCandidatePair:
    """One fixed C2/I2 fitted-model comparison."""

    role: str
    seed: int
    c2_experiment_id: str
    i2_experiment_id: str


@dataclass(frozen=True)
class PairedBootstrapSpec:
    """Strict declaration for the G6 paired family bootstrap."""

    analysis_id: str
    expected_row_count: int
    expected_group_count: int
    group_column: str
    group_semantics: str
    pairs: tuple[BootstrapCandidatePair, ...]
    replicates: int
    bootstrap_seed: int
    batch_size: int
    confidence_level: float
    practical_tie_threshold: float
    interval_method: str
    quantile_method: str
    current_candidate: str
    confidence_language_if_interval_contains_zero: str
    confidence_language_if_lower_bound_above_zero: str


@dataclass(frozen=True)
class PairedBootstrapTables:
    """Measured outputs from one deterministic paired bootstrap run."""

    observed_metrics: pd.DataFrame
    interval_summary: pd.DataFrame
    group_audit: pd.DataFrame
    draws: pd.DataFrame


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], scope: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        raise ValueError(f"{scope} fields changed; missing={missing}, unknown={unknown}")


def _exact_integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _exact_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def load_paired_bootstrap_spec(
    path: str | Path = PAIRED_BOOTSTRAP_CONFIG_PATH,
) -> PairedBootstrapSpec:
    """Load the frozen G6 declaration without coercing malformed values."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("paired bootstrap config must be a JSON object")
    _require_exact_keys(
        payload,
        {
            "analysis_id",
            "bootstrap",
            "candidate_pairs",
            "decision_boundary",
            "expected_group_count",
            "expected_row_count",
            "group_column",
            "group_semantics",
            "labels",
            "method_references",
            "metrics",
            "schema_version",
            "stage",
            "target",
            "warnings",
        },
        "paired bootstrap config",
    )
    identity = {
        "schema_version": "1.0.0",
        "analysis_id": "g6-paired-group-bootstrap",
        "stage": "g6_paired_group_bootstrap_uncertainty",
        "target": "season",
        "group_column": "product_family_group",
        "group_semantics": "conservative_dependency_block_not_verified_sku",
    }
    mismatches = [name for name, expected in identity.items() if payload[name] != expected]
    if mismatches:
        raise ValueError(f"paired bootstrap identity changed: {mismatches}")
    if tuple(payload["labels"]) != tuple(SEASON_LABELS):
        raise ValueError("paired bootstrap changed the canonical Season order")

    expected_row_count = _exact_integer(
        payload["expected_row_count"],
        name="expected_row_count",
        minimum=1,
    )
    expected_group_count = _exact_integer(
        payload["expected_group_count"],
        name="expected_group_count",
        minimum=2,
    )
    if expected_row_count != 32_753 or expected_group_count != 22_885:
        raise ValueError("paired bootstrap changed the canonical row/group counts")

    raw_pairs = payload["candidate_pairs"]
    if not isinstance(raw_pairs, list) or len(raw_pairs) != 2:
        raise ValueError("paired bootstrap requires exactly two candidate pairs")
    pairs: list[BootstrapCandidatePair] = []
    for index, raw_pair in enumerate(raw_pairs):
        if not isinstance(raw_pair, Mapping):
            raise ValueError("paired bootstrap candidate pairs must be objects")
        _require_exact_keys(
            raw_pair,
            {"c2_experiment_id", "i2_experiment_id", "role", "seed"},
            f"candidate_pairs[{index}]",
        )
        pair = BootstrapCandidatePair(
            role=str(raw_pair["role"]),
            seed=_exact_integer(raw_pair["seed"], name=f"pair {index} seed", minimum=0),
            c2_experiment_id=str(raw_pair["c2_experiment_id"]),
            i2_experiment_id=str(raw_pair["i2_experiment_id"]),
        )
        pairs.append(pair)
    observed_pairs = tuple(
        (pair.role, pair.seed, pair.c2_experiment_id, pair.i2_experiment_id) for pair in pairs
    )
    if observed_pairs != EXPECTED_PAIR_IDENTITIES:
        raise ValueError("paired bootstrap changed the frozen C2/I2 pair identities")

    bootstrap = payload["bootstrap"]
    if not isinstance(bootstrap, Mapping):
        raise ValueError("paired bootstrap protocol must be an object")
    _require_exact_keys(
        bootstrap,
        {
            "batch_size",
            "bootstrap_seed",
            "cluster_draw_rule",
            "confidence_level",
            "group_order",
            "interval_method",
            "pairing_rule",
            "quantile_method",
            "replicates_per_seed_pair",
            "retain_cluster_rows",
            "rng",
            "same_group_draws_across_seed_pairs",
        },
        "bootstrap protocol",
    )
    fixed_protocol = {
        "cluster_draw_rule": "sample_observed_group_count_with_replacement",
        "group_order": "ascending_product_family_group",
        "interval_method": "percentile",
        "pairing_rule": "same_group_multiplicity_for_c2_and_i2",
        "quantile_method": "linear",
        "retain_cluster_rows": "all_rows_with_sampled_group_multiplicity",
        "rng": "numpy_pcg64",
        "same_group_draws_across_seed_pairs": True,
    }
    protocol_mismatches = [
        name for name, expected in fixed_protocol.items() if bootstrap[name] != expected
    ]
    if protocol_mismatches:
        raise ValueError(f"paired bootstrap protocol changed: {protocol_mismatches}")
    replicates = _exact_integer(
        bootstrap["replicates_per_seed_pair"],
        name="replicates_per_seed_pair",
        minimum=1,
    )
    bootstrap_seed = _exact_integer(
        bootstrap["bootstrap_seed"],
        name="bootstrap_seed",
        minimum=0,
    )
    batch_size = _exact_integer(bootstrap["batch_size"], name="batch_size", minimum=1)
    confidence_level = _exact_real(
        bootstrap["confidence_level"],
        name="confidence_level",
    )
    if (
        replicates != 10_000
        or bootstrap_seed != 2753
        or batch_size != 64
        or confidence_level != 0.95
    ):
        raise ValueError("paired bootstrap numeric protocol changed")

    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("paired bootstrap metrics must be an object")
    _require_exact_keys(metrics, {"primary", "secondary", "zero_division"}, "metrics")
    if (
        metrics["primary"] != "i2_minus_c2_pooled_oof_macro_f1"
        or tuple(metrics["secondary"]) != ("i2_minus_c2_accuracy", "i2_minus_c2_per_class_f1")
        or _exact_integer(metrics["zero_division"], name="zero_division", minimum=0) != 0
    ):
        raise ValueError("paired bootstrap metric contract changed")

    boundary = payload["decision_boundary"]
    if not isinstance(boundary, Mapping):
        raise ValueError("paired bootstrap decision boundary must be an object")
    _require_exact_keys(
        boundary,
        {
            "confidence_language_if_interval_contains_zero",
            "confidence_language_if_lower_bound_above_zero",
            "current_candidate",
            "new_candidates_allowed",
            "practical_tie_threshold_macro_f1",
            "random_seed_generalisability_claim_allowed",
            "ultimate_winner_frozen_by_this_gate",
        },
        "decision boundary",
    )
    tie_threshold = _exact_real(
        boundary["practical_tie_threshold_macro_f1"],
        name="practical_tie_threshold_macro_f1",
    )
    contains_zero_language = "cluster_resampling_does_not_resolve_the_fitted_pair"
    positive_language = (
        "cluster_resampling_supports_a_positive_i2_minus_c2_difference_for_this_fitted_pair"
    )
    if (
        boundary["current_candidate"] != "I2"
        or boundary["new_candidates_allowed"] is not False
        or boundary["random_seed_generalisability_claim_allowed"] is not False
        or boundary["ultimate_winner_frozen_by_this_gate"] is not False
        or tie_threshold != 0.005
        or boundary["confidence_language_if_interval_contains_zero"] != contains_zero_language
        or boundary["confidence_language_if_lower_bound_above_zero"] != positive_language
    ):
        raise ValueError("paired bootstrap decision safety boundary changed")

    warnings = payload["warnings"]
    expected_warnings = {
        "calibrated_probabilities_are_not_needed_for_label_metrics",
        "family_groups_are_conservative_proxies",
        "holdout_is_forbidden",
        "independent_row_bootstrap_is_forbidden",
        "two_fixed_training_seeds_do_not_measure_all_training_randomness",
    }
    if not isinstance(warnings, Mapping):
        raise ValueError("paired bootstrap warnings must be an object")
    _require_exact_keys(warnings, expected_warnings, "warnings")
    if any(warnings[name] is not True for name in expected_warnings):
        raise ValueError("paired bootstrap safety warnings must stay enabled")

    expected_references = [
        {
            "citation": "Field and Welsh (2007), Bootstrapping clustered data",
            "doi": "10.1111/j.1467-9868.2007.00593.x",
            "purpose": "cluster_resampling_boundary",
        },
        {
            "citation": "Rifkin and Klautau (2004), In Defense of One-Vs-All Classification",
            "url": "https://www.jmlr.org/papers/v5/rifkin04a.html",
            "purpose": "paired_classifier_bootstrap_precedent",
        },
    ]
    if payload["method_references"] != expected_references:
        raise ValueError("paired bootstrap method references changed")

    return PairedBootstrapSpec(
        analysis_id=str(payload["analysis_id"]),
        expected_row_count=expected_row_count,
        expected_group_count=expected_group_count,
        group_column=str(payload["group_column"]),
        group_semantics=str(payload["group_semantics"]),
        pairs=tuple(pairs),
        replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        batch_size=batch_size,
        confidence_level=confidence_level,
        practical_tie_threshold=tie_threshold,
        interval_method=str(bootstrap["interval_method"]),
        quantile_method=str(bootstrap["quantile_method"]),
        current_candidate=str(boundary["current_candidate"]),
        confidence_language_if_interval_contains_zero=contains_zero_language,
        confidence_language_if_lower_bound_above_zero=positive_language,
    )


def _integer_ids(values: pd.Series, scope: str) -> np.ndarray:
    identifiers: list[int] = []
    limits = np.iinfo(np.int64)
    for value in values.to_numpy(dtype=object):
        if isinstance(value, bool):
            raise ValueError(f"{scope} must contain finite integer IDs")
        if isinstance(value, Integral):
            identifier = int(value)
        elif isinstance(value, str):
            token = value.strip()
            digits = token[1:] if token[:1] in {"+", "-"} else token
            if not digits or not digits.isdecimal():
                raise ValueError(f"{scope} must contain finite integer IDs")
            identifier = int(token)
        else:
            raise ValueError(f"{scope} must contain finite integer IDs")
        if not limits.min <= identifier <= limits.max:
            raise ValueError(f"{scope} must fit signed 64-bit integers")
        identifiers.append(identifier)
    return np.asarray(identifiers, dtype=np.int64)


def _fixed_label_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    true_indices = np.fromiter(
        (label_to_index[str(value)] for value in y_true),
        dtype=np.int64,
        count=len(y_true),
    )
    predicted_indices = np.fromiter(
        (label_to_index[str(value)] for value in y_pred),
        dtype=np.int64,
        count=len(y_pred),
    )
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(confusion, (true_indices, predicted_indices), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=1) + confusion.sum(axis=0)
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return {
        "macro_f1": float(per_class_f1.mean()),
        "accuracy": float(true_positive.sum() / len(y_true)),
        "per_class_f1": {label: float(per_class_f1[index]) for index, label in enumerate(labels)},
    }


def _canonical_development(
    development: pd.DataFrame,
    spec: PairedBootstrapSpec,
) -> pd.DataFrame:
    required = {"id", "season", spec.group_column}
    missing = sorted(required - set(development))
    if missing:
        raise ValueError(f"canonical development rows are missing columns: {missing}")
    canonical = development.loc[:, ["id", "season", spec.group_column]].copy()
    if len(canonical) != spec.expected_row_count:
        raise ValueError("paired bootstrap development row count changed")
    canonical["id"] = _integer_ids(canonical["id"], "canonical development IDs")
    if canonical["id"].duplicated().any():
        raise ValueError("canonical development IDs must be unique")
    canonical["season"] = canonical["season"].astype(str)
    unknown_targets = sorted(set(canonical["season"]) - set(SEASON_LABELS))
    if unknown_targets:
        raise ValueError(f"canonical development contains unknown labels: {unknown_targets}")
    group_values = canonical[spec.group_column].to_numpy(dtype=object)
    if any(not isinstance(value, str) or not value.strip() for value in group_values):
        raise ValueError("canonical product-family groups must be non-empty strings")
    if canonical[spec.group_column].nunique() != spec.expected_group_count:
        raise ValueError("paired bootstrap product-family group count changed")
    return canonical.sort_values("id", kind="stable").reset_index(drop=True)


def _align_pack(
    pack: CandidateOOFPack,
    canonical: pd.DataFrame,
    spec: PairedBootstrapSpec,
) -> pd.DataFrame:
    required = {"id", "y_true", "y_pred"}
    missing = sorted(required - set(pack.oof))
    if missing:
        raise ValueError(f"{pack.experiment_id} OOF is missing columns: {missing}")
    predictions = pack.oof.loc[:, ["id", "y_true", "y_pred"]].copy()
    if len(predictions) != spec.expected_row_count:
        raise ValueError(f"{pack.experiment_id} OOF row count changed")
    predictions["id"] = _integer_ids(predictions["id"], f"{pack.experiment_id} OOF IDs")
    if predictions["id"].duplicated().any():
        raise ValueError(f"{pack.experiment_id} OOF IDs must be unique")
    if set(predictions["id"]) != set(canonical["id"]):
        raise ValueError(f"{pack.experiment_id} OOF ID coverage changed")
    predictions["y_true"] = predictions["y_true"].astype(str)
    predictions["y_pred"] = predictions["y_pred"].astype(str)
    unknown_predictions = sorted(set(predictions["y_pred"]) - set(SEASON_LABELS))
    if unknown_predictions:
        raise ValueError(f"{pack.experiment_id} OOF has unknown predictions")
    aligned = canonical.merge(
        predictions,
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if not np.array_equal(
        aligned["season"].to_numpy(dtype=str),
        aligned["y_true"].to_numpy(dtype=str),
    ):
        raise ValueError(f"{pack.experiment_id} OOF truth differs from canonical Season")
    return aligned


def _summarise_intervals(
    draws: pd.DataFrame,
    observed: pd.DataFrame,
    spec: PairedBootstrapSpec,
) -> pd.DataFrame:
    alpha = (1.0 - spec.confidence_level) / 2.0
    metric_definitions = (
        ("overall", "macro_f1", "", "macro_f1"),
        ("overall", "accuracy", "", "accuracy"),
        *(("per_class", "f1", label, f"f1_{label.lower()}") for label in SEASON_LABELS),
    )
    rows: list[dict[str, Any]] = []
    for pair in spec.pairs:
        comparison_draws = draws.loc[draws["comparison_id"].eq(pair.role)]
        observed_row = observed.loc[observed["comparison_id"].eq(pair.role)]
        if len(comparison_draws) != spec.replicates or len(observed_row) != 1:
            raise ValueError(f"paired bootstrap output is incomplete for {pair.role}")
        observed_values = observed_row.iloc[0]
        for scope, metric, label, suffix in metric_definitions:
            delta_column = f"i2_minus_c2_{suffix}"
            values = comparison_draws[delta_column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"paired bootstrap produced non-finite {delta_column}")
            lower, upper = np.quantile(
                values,
                [alpha, 1.0 - alpha],
                method=spec.quantile_method,
            )
            observed_delta = float(observed_values[delta_column])
            is_primary = suffix == "macro_f1"
            rows.append(
                {
                    "comparison_id": pair.role,
                    "role": pair.role,
                    "seed": pair.seed,
                    "c2_experiment_id": pair.c2_experiment_id,
                    "i2_experiment_id": pair.i2_experiment_id,
                    "metric_scope": scope,
                    "metric": metric,
                    "label": label,
                    "observed_delta": observed_delta,
                    "bootstrap_mean_delta": float(values.mean()),
                    "bootstrap_median_delta": float(np.median(values)),
                    "bootstrap_sd": float(values.std(ddof=1)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "confidence_level": spec.confidence_level,
                    "interval_contains_zero": bool(lower <= 0.0 <= upper),
                    "fraction_replicates_above_zero": float(np.mean(values > 0.0)),
                    "replicates": spec.replicates,
                    "interval_method": spec.interval_method,
                    "quantile_method": spec.quantile_method,
                    "practical_tie_threshold": (
                        spec.practical_tie_threshold if is_primary else np.nan
                    ),
                    "observed_within_practical_tie": (
                        bool(abs(observed_delta) < spec.practical_tie_threshold)
                        if is_primary
                        else False
                    ),
                }
            )
    return pd.DataFrame(rows)


def analyse_paired_bootstrap_packs(
    packs: Sequence[CandidateOOFPack],
    development: pd.DataFrame,
    spec: PairedBootstrapSpec,
) -> PairedBootstrapTables:
    """Join four audited OOF packs by ID, then run one shared family bootstrap."""
    canonical = _canonical_development(development, spec)
    expected_identities = {("C2", pair.c2_experiment_id, pair.seed) for pair in spec.pairs} | {
        ("I2", pair.i2_experiment_id, pair.seed) for pair in spec.pairs
    }
    observed_identities = {(pack.candidate, pack.experiment_id, pack.seed) for pack in packs}
    if len(packs) != 4 or observed_identities != expected_identities:
        raise ValueError("paired bootstrap requires the exact four frozen OOF packs")
    by_identity = {(pack.candidate, pack.experiment_id, pack.seed): pack for pack in packs}

    comparisons: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    observed_rows: list[dict[str, Any]] = []
    true = canonical["season"].to_numpy(dtype=object)
    for pair in spec.pairs:
        c2 = _align_pack(
            by_identity[("C2", pair.c2_experiment_id, pair.seed)],
            canonical,
            spec,
        )
        i2 = _align_pack(
            by_identity[("I2", pair.i2_experiment_id, pair.seed)],
            canonical,
            spec,
        )
        c2_predictions = c2["y_pred"].to_numpy(dtype=object)
        i2_predictions = i2["y_pred"].to_numpy(dtype=object)
        comparisons[pair.role] = (c2_predictions, i2_predictions)
        c2_scores = _fixed_label_scores(true, c2_predictions, tuple(SEASON_LABELS))
        i2_scores = _fixed_label_scores(true, i2_predictions, tuple(SEASON_LABELS))
        row: dict[str, Any] = {
            "comparison_id": pair.role,
            "role": pair.role,
            "seed": pair.seed,
            "c2_experiment_id": pair.c2_experiment_id,
            "i2_experiment_id": pair.i2_experiment_id,
            "row_count": len(canonical),
            "group_count": canonical[spec.group_column].nunique(),
            "c2_macro_f1": c2_scores["macro_f1"],
            "i2_macro_f1": i2_scores["macro_f1"],
            "i2_minus_c2_macro_f1": i2_scores["macro_f1"] - c2_scores["macro_f1"],
            "c2_accuracy": c2_scores["accuracy"],
            "i2_accuracy": i2_scores["accuracy"],
            "i2_minus_c2_accuracy": i2_scores["accuracy"] - c2_scores["accuracy"],
        }
        for label in SEASON_LABELS:
            slug = label.lower()
            c2_f1 = c2_scores["per_class_f1"][label]
            i2_f1 = i2_scores["per_class_f1"][label]
            row[f"c2_f1_{slug}"] = c2_f1
            row[f"i2_f1_{slug}"] = i2_f1
            row[f"i2_minus_c2_f1_{slug}"] = i2_f1 - c2_f1
        observed_rows.append(row)
    observed = pd.DataFrame(observed_rows)

    draws = paired_group_bootstrap(
        true,
        canonical[spec.group_column].to_numpy(dtype=object),
        comparisons,
        labels=SEASON_LABELS,
        replicates=spec.replicates,
        random_seed=spec.bootstrap_seed,
        batch_size=spec.batch_size,
    ).rename(
        columns=lambda column: (
            column.replace("model_a_", "c2_")
            .replace("model_b_", "i2_")
            .replace("b_minus_a_", "i2_minus_c2_")
        )
    )
    pair_metadata = {pair.role: pair for pair in spec.pairs}
    draws.insert(1, "role", draws["comparison_id"])
    draws.insert(2, "seed", draws["comparison_id"].map(lambda role: pair_metadata[role].seed))
    draws.insert(
        3,
        "c2_experiment_id",
        draws["comparison_id"].map(lambda role: pair_metadata[role].c2_experiment_id),
    )
    draws.insert(
        4,
        "i2_experiment_id",
        draws["comparison_id"].map(lambda role: pair_metadata[role].i2_experiment_id),
    )
    counts = draws.groupby("comparison_id", sort=True)["replicate"].count().to_dict()
    if counts != {pair.role: spec.replicates for pair in spec.pairs}:
        raise ValueError("paired bootstrap replicate counts changed")
    sampled_rows = draws.pivot(
        index="replicate",
        columns="comparison_id",
        values="sampled_row_count",
    )
    if not sampled_rows.nunique(axis=1).eq(1).all():
        raise ValueError("paired bootstrap did not reuse group draws across seed pairs")
    if not draws["sampled_group_count"].eq(spec.expected_group_count).all():
        raise ValueError("paired bootstrap sampled group count changed")

    intervals = _summarise_intervals(draws, observed, spec)
    group_sizes = canonical.groupby(spec.group_column, sort=True)["id"].count()
    group_audit = pd.DataFrame(
        [
            {
                "row_count": len(canonical),
                "unique_group_count": len(group_sizes),
                "singleton_group_count": int(group_sizes.eq(1).sum()),
                "multirow_group_count": int(group_sizes.gt(1).sum()),
                "minimum_group_size": int(group_sizes.min()),
                "median_group_size": float(group_sizes.median()),
                "mean_group_size": float(group_sizes.mean()),
                "maximum_group_size": int(group_sizes.max()),
                "sampled_group_count_per_replicate": spec.expected_group_count,
                "minimum_sampled_row_count": int(draws["sampled_row_count"].min()),
                "mean_sampled_row_count": float(draws["sampled_row_count"].mean()),
                "maximum_sampled_row_count": int(draws["sampled_row_count"].max()),
                "group_semantics": spec.group_semantics,
                "group_assignment_sha256": canonical_sha256(
                    canonical.loc[:, ["id", spec.group_column]].to_dict(orient="records")
                ),
                "group_size_sha256": canonical_sha256(
                    [
                        {"group": str(group), "size": int(size)}
                        for group, size in group_sizes.items()
                    ]
                ),
            }
        ]
    )
    return PairedBootstrapTables(
        observed_metrics=observed,
        interval_summary=intervals,
        group_audit=group_audit,
        draws=draws,
    )


def build_paired_bootstrap_decision(
    tables: PairedBootstrapTables,
    spec: PairedBootstrapSpec,
) -> dict[str, Any]:
    """Translate the frozen primary interval into bounded, non-causal wording."""
    outcomes: dict[str, Any] = {}
    for pair in spec.pairs:
        rows = tables.interval_summary.loc[
            tables.interval_summary["comparison_id"].eq(pair.role)
            & tables.interval_summary["metric_scope"].eq("overall")
            & tables.interval_summary["metric"].eq("macro_f1")
        ]
        if len(rows) != 1:
            raise ValueError(f"paired bootstrap decision lacks {pair.role} macro-F1 interval")
        row = rows.iloc[0]
        if (
            str(row["role"]) != pair.role
            or _exact_integer(row["seed"], name=f"{pair.role} seed", minimum=0) != pair.seed
            or str(row["c2_experiment_id"]) != pair.c2_experiment_id
            or str(row["i2_experiment_id"]) != pair.i2_experiment_id
            or _exact_integer(
                row["replicates"],
                name=f"{pair.role} replicates",
                minimum=1,
            )
            != spec.replicates
            or float(row["confidence_level"]) != spec.confidence_level
            or str(row["interval_method"]) != spec.interval_method
            or str(row["quantile_method"]) != spec.quantile_method
        ):
            raise ValueError(f"paired bootstrap decision contract changed for {pair.role}")
        observed_delta = float(row["observed_delta"])
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        if (
            not np.isfinite((observed_delta, lower, upper)).all()
            or not -1.0 <= observed_delta <= 1.0
            or not -1.0 <= lower <= upper <= 1.0
        ):
            raise ValueError(f"paired bootstrap decision interval is invalid for {pair.role}")
        contains_zero = lower <= 0.0 <= upper
        recorded_contains_zero = row["interval_contains_zero"]
        recorded_tie = row["observed_within_practical_tie"]
        expected_tie = abs(observed_delta) < spec.practical_tie_threshold
        if (
            not isinstance(recorded_contains_zero, (bool, np.bool_))
            or bool(recorded_contains_zero) != contains_zero
            or not isinstance(recorded_tie, (bool, np.bool_))
            or bool(recorded_tie) != expected_tie
        ):
            raise ValueError(f"paired bootstrap decision flags changed for {pair.role}")
        if lower > 0.0:
            confidence_language = spec.confidence_language_if_lower_bound_above_zero
        elif upper < 0.0:
            confidence_language = (
                "cluster_resampling_supports_a_negative_i2_minus_c2_difference_for_this_fitted_pair"
            )
        else:
            confidence_language = spec.confidence_language_if_interval_contains_zero
        outcomes[pair.role] = {
            "seed": pair.seed,
            "observed_i2_minus_c2_macro_f1": observed_delta,
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence_level": spec.confidence_level,
            "interval_contains_zero": contains_zero,
            "observed_within_practical_tie": expected_tie,
            "confidence_language": confidence_language,
        }
    return {
        "schema_version": "1.0.0",
        "gate": "G6-PAIRED-BOOTSTRAP",
        "decision_status": "closed",
        "analysis_role": "development_oof_fitted_pair_uncertainty_only",
        "current_candidate": spec.current_candidate,
        "candidate_selection_affected": False,
        "new_candidates_allowed": False,
        "ultimate_winner_frozen": False,
        "holdout_opened": False,
        "random_seed_generalisability_claim_allowed": False,
        "practical_tie_threshold_macro_f1": spec.practical_tie_threshold,
        "pair_outcomes": outcomes,
        "interpretation_rule": (
            "Intervals compare I2 minus C2 for two already-fitted seed pairs. They quantify "
            "product-family resampling uncertainty, not all training randomness."
        ),
        "limitations": [
            "product_family_group is a conservative dependency block, not a verified SKU",
            "two fixed training seeds cannot measure all optimisation randomness",
            "percentile intervals describe these fitted development OOF pairs only",
            "holdout remains sealed and the ultimate winner is not frozen by this gate",
        ],
        "next_question": "Review deterministic Grad-CAM examples and failure taxonomy.",
    }


def plot_paired_bootstrap_intervals(
    tables: PairedBootstrapTables,
    output_path: str | Path,
) -> Path:
    """Plot macro-F1 draw distributions and all paired 95% intervals."""
    expected_roles = {role for role, *_ in EXPECTED_PAIR_IDENTITIES}
    if set(tables.draws["comparison_id"]) != expected_roles:
        raise ValueError("paired bootstrap plot requires both frozen comparisons")
    required_metrics = {"macro_f1", "accuracy", "f1"}
    if set(tables.interval_summary["metric"]) != required_metrics:
        raise ValueError("paired bootstrap plot interval metrics changed")

    colors = {"primary_interval": "#2563eb", "stability_sensitivity": "#f97316"}
    labels = {"primary_interval": "Seed 2753", "stability_sensitivity": "Seed 2026"}
    figure = Figure(figsize=(14, 5.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    distribution_axis, interval_axis = figure.subplots(1, 2)
    for role in ("primary_interval", "stability_sensitivity"):
        values = tables.draws.loc[
            tables.draws["comparison_id"].eq(role), "i2_minus_c2_macro_f1"
        ].to_numpy(dtype=float)
        distribution_axis.hist(
            values * 100.0,
            bins=45,
            density=True,
            alpha=0.42,
            color=colors[role],
            label=labels[role],
        )
    distribution_axis.axvline(0.0, color="#111827", linewidth=1.4)
    distribution_axis.axvspan(-0.5, 0.5, color="#9ca3af", alpha=0.14, label="±0.5 pp tie band")
    distribution_axis.set_title("Paired family-bootstrap macro-F1 differences")
    distribution_axis.set_xlabel("I2 minus C2 (percentage points)")
    distribution_axis.set_ylabel("Density")
    distribution_axis.grid(alpha=0.2)
    distribution_axis.legend(fontsize=8)

    metric_order = (
        ("overall", "macro_f1", "", "Macro-F1"),
        ("overall", "accuracy", "", "Accuracy"),
        *(("per_class", "f1", label, f"{label} F1") for label in SEASON_LABELS),
    )
    base_y = np.arange(len(metric_order), dtype=float)
    offsets = {"primary_interval": -0.12, "stability_sensitivity": 0.12}
    for role in ("primary_interval", "stability_sensitivity"):
        observed_values = []
        lower_bounds = []
        upper_bounds = []
        for scope, metric, label, _ in metric_order:
            row = tables.interval_summary.loc[
                tables.interval_summary["comparison_id"].eq(role)
                & tables.interval_summary["metric_scope"].eq(scope)
                & tables.interval_summary["metric"].eq(metric)
                & tables.interval_summary["label"].eq(label)
            ]
            if len(row) != 1:
                raise ValueError(f"paired bootstrap plot lacks one {role}/{metric}/{label} row")
            values = row.iloc[0]
            observed = float(values["observed_delta"]) * 100.0
            lower = float(values["ci_lower"]) * 100.0
            upper = float(values["ci_upper"]) * 100.0
            observed_values.append(observed)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        y_values = base_y + offsets[role]
        interval_axis.hlines(
            y_values,
            lower_bounds,
            upper_bounds,
            color=colors[role],
            linewidth=2,
        )
        interval_axis.scatter(
            observed_values,
            y_values,
            color=colors[role],
            marker="o",
            s=30,
            label=labels[role],
        )
    interval_axis.axvline(0.0, color="#111827", linewidth=1.4)
    interval_axis.set_yticks(base_y, [name for *_, name in metric_order])
    interval_axis.invert_yaxis()
    interval_axis.set_title("Observed differences with 95% percentile intervals")
    interval_axis.set_xlabel("I2 minus C2 (percentage points)")
    interval_axis.grid(axis="x", alpha=0.2)
    interval_axis.legend(fontsize=8)

    destination = Path(output_path)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    atomic_write_bytes(destination, buffer.getvalue())
    figure.clear()
    return destination


__all__ = [
    "analyse_paired_bootstrap_packs",
    "BootstrapCandidatePair",
    "build_paired_bootstrap_decision",
    "EXPECTED_PAIR_IDENTITIES",
    "load_paired_bootstrap_spec",
    "PAIRED_BOOTSTRAP_CONFIG_PATH",
    "PairedBootstrapSpec",
    "PairedBootstrapTables",
    "plot_paired_bootstrap_intervals",
]
