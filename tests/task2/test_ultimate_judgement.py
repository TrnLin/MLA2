from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

import fashion.task2.ultimate_judgement as judgement_module
from fashion.config import ROOT
from fashion.task2.ultimate_judgement import (
    apply_ultimate_judgement,
    build_candidate_scorecard,
    build_rejected_alternatives,
    build_ultimate_judgement_evidence,
    load_ultimate_judgement_spec,
    load_verified_selection_freeze,
    load_verified_ultimate_judgement_manifest,
)


def _read(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


def _measured_tables() -> dict[str, pd.DataFrame]:
    return {
        "stability": _read("results/evidence/task2/seed_stability/seed_stability.csv"),
        "slice_deltas": _read(
            "results/evidence/task2/shortcut_error_slices/candidate_slice_deltas.csv"
        ),
        "robustness": _read("results/evidence/task2/robustness_cost/candidate_comparison.csv"),
        "deployment_cost": _read("results/evidence/task2/robustness_cost/deployment_cost.csv"),
        "calibration": _read("results/evidence/task2/calibration/calibration_summary.csv"),
        "interval_summary": _read("results/evidence/task2/paired_bootstrap/interval_summary.csv"),
    }


def _scorecard() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    tables = _measured_tables()
    scorecard = build_candidate_scorecard(
        spec=load_ultimate_judgement_spec(),
        stability=tables["stability"],
        slice_deltas=tables["slice_deltas"],
        robustness=tables["robustness"],
        deployment_cost=tables["deployment_cost"],
        calibration=tables["calibration"],
    )
    return scorecard, tables


def _applied_scorecard() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, object]]:
    scorecard, tables = _scorecard()
    selected, decision = apply_ultimate_judgement(
        scorecard,
        interval_summary=tables["interval_summary"],
        robustness=tables["robustness"],
        spec=load_ultimate_judgement_spec(),
    )
    return selected, tables, decision


def test_ultimate_judgement_config_keeps_two_scratch_candidate_roles() -> None:
    spec = load_ultimate_judgement_spec()

    assert [(candidate.candidate, candidate.role) for candidate in spec.candidates] == [
        ("C2", "reference"),
        ("I2", "challenger"),
    ]
    assert spec.primary_metric == "pooled_five_fold_oof_macro_f1"
    assert spec.rules.practical_tie_threshold_macro_f1 == 0.005
    assert spec.refit_epoch_rule == "median_primary_seed_cv_best_epoch"
    assert spec.refit_seed == 2753


def test_measured_scorecard_joins_quality_shortcut_stress_calibration_and_cost() -> None:
    scorecard, _ = _scorecard()
    indexed = scorecard.set_index("candidate")

    assert set(indexed.index) == {"C2", "I2"}
    assert indexed.loc["I2", "primary_macro_f1"] == pytest.approx(0.7526869559580971)
    assert indexed.loc["I2", "stability_macro_f1"] == pytest.approx(0.7447434490683583)
    assert indexed.loc["I2", "article_type_conflict_macro_f1_seed_2753"] == pytest.approx(
        0.43021100203044355
    )
    assert indexed.loc["I2", "worst_stress_condition"] == "brightness_0_85"
    assert indexed.loc["I2", "worst_stress_spring_recall"] == pytest.approx(0.0030097817908201654)
    assert indexed.loc["I2", "parameter_count"] == 1_206_112
    assert (
        indexed.loc["I2", "cpu_end_to_end_median_ms"]
        < indexed.loc["C2", "cpu_end_to_end_median_ms"]
    )


def test_frozen_scorecard_selects_i2_without_using_the_cost_tie_break() -> None:
    scorecard, tables = _scorecard()

    selected, decision = apply_ultimate_judgement(
        scorecard,
        interval_summary=tables["interval_summary"],
        robustness=tables["robustness"],
        spec=load_ultimate_judgement_spec(),
    )

    assert selected.loc[selected["selected"], "candidate"].item() == "I2"
    assert decision["selected_experiment_id"] == "g4-i2-article-type-lambda-0-3-c1"
    assert all(decision["selection_checks"].values())
    assert decision["direct_selection_rule_passed"] is True
    assert decision["near_tie_rule_triggered"] is False
    assert decision["cost_tie_break_used"] is False
    assert decision["ultimate_winner_frozen"] is True
    assert decision["holdout_opened"] is False
    assert decision["holdout_metrics_present"] is False


def test_scorecard_falls_back_to_c2_when_a_required_bootstrap_interval_fails() -> None:
    scorecard, tables = _scorecard()
    intervals = tables["interval_summary"].copy()
    mask = (
        intervals["seed"].eq(2026)
        & intervals["metric_scope"].eq("overall")
        & intervals["metric"].eq("macro_f1")
    )
    intervals.loc[mask, "ci_lower"] = -0.001
    intervals.loc[mask, "interval_contains_zero"] = True

    selected, decision = apply_ultimate_judgement(
        scorecard,
        interval_summary=intervals,
        robustness=tables["robustness"],
        spec=load_ultimate_judgement_spec(),
    )

    assert selected.loc[selected["selected"], "candidate"].item() == "C2"
    assert decision["selection_checks"]["both_grouped_bootstrap_intervals_above_zero"] is False
    assert decision["direct_selection_rule_passed"] is False


def test_cost_tie_break_cannot_bypass_stability_shortcut_or_jpeg_guards() -> None:
    scorecard, tables = _scorecard()
    c2 = scorecard["candidate"].eq("C2")
    i2 = scorecard["candidate"].eq("I2")
    scorecard.loc[i2, "primary_macro_f1"] = (
        float(scorecard.loc[c2, "primary_macro_f1"].item()) + 0.001
    )
    scorecard.loc[i2, "stability_macro_f1"] = (
        float(scorecard.loc[c2, "stability_macro_f1"].item()) - 0.5
    )
    for seed in (2753, 2026):
        field = f"article_type_conflict_macro_f1_seed_{seed}"
        scorecard.loc[i2, field] = float(scorecard.loc[c2, field].item()) - 0.5
    scorecard.loc[i2, "jpeg_drop_vs_clean"] = (
        float(scorecard.loc[c2, "jpeg_drop_vs_clean"].item()) + 0.5
    )
    intervals = tables["interval_summary"].copy()
    primary = (
        intervals["seed"].eq(2753)
        & intervals["metric_scope"].eq("overall")
        & intervals["metric"].eq("macro_f1")
    )
    intervals.loc[primary, "ci_lower"] = -0.001
    intervals.loc[primary, "interval_contains_zero"] = True

    selected, decision = apply_ultimate_judgement(
        scorecard,
        interval_summary=intervals,
        robustness=tables["robustness"],
        spec=load_ultimate_judgement_spec(),
    )

    assert decision["near_tie_rule_triggered"] is True
    assert decision["cost_tie_break_safety_passed"] is False
    assert decision["cost_tie_break_used"] is False
    assert selected.loc[selected["selected"], "candidate"].item() == "C2"


def test_safe_practical_tie_can_choose_the_smaller_or_faster_challenger() -> None:
    scorecard, tables = _scorecard()
    c2 = scorecard["candidate"].eq("C2")
    i2 = scorecard["candidate"].eq("I2")
    scorecard.loc[i2, "primary_macro_f1"] = (
        float(scorecard.loc[c2, "primary_macro_f1"].item()) + 0.001
    )
    intervals = tables["interval_summary"].copy()
    primary = (
        intervals["seed"].eq(2753)
        & intervals["metric_scope"].eq("overall")
        & intervals["metric"].eq("macro_f1")
    )
    intervals.loc[primary, "ci_lower"] = -0.001
    intervals.loc[primary, "interval_contains_zero"] = True

    selected, decision = apply_ultimate_judgement(
        scorecard,
        interval_summary=intervals,
        robustness=tables["robustness"],
        spec=load_ultimate_judgement_spec(),
    )

    assert decision["direct_selection_rule_passed"] is False
    assert decision["cost_tie_break_safety_passed"] is True
    assert decision["cost_tie_break_used"] is True
    assert selected.loc[selected["selected"], "candidate"].item() == "I2"


def test_rejected_alternatives_trace_every_model_role_to_a_manifest() -> None:
    scorecard, _, _ = _applied_scorecard()

    rejected = build_rejected_alternatives(project_root=ROOT, scorecard=scorecard)

    assert len(rejected) == 12
    assert rejected["source_manifest_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert {
        "comparison_anchor",
        "serious_classical_baseline",
        "full_budget_finalist",
        "minority_class_intervention",
        "multitask_weight_ablation",
        "pretraining_benchmark",
        "final_reference",
    } <= set(rejected["role"])
    pstar = rejected.loc[rejected["alternative"].str.startswith("P* pretrained")].iloc[0]
    assert bool(pstar["final_eligible"]) is False
    assert "scratch" in str(pstar["reason"])
    assert not rejected["alternative"].str.startswith("I2 ArticleType auxiliary lambda 0.3").any()


def test_rejected_alternatives_follow_a_c2_fallback() -> None:
    scorecard, tables = _scorecard()
    intervals = tables["interval_summary"].copy()
    mask = (
        intervals["seed"].eq(2026)
        & intervals["metric_scope"].eq("overall")
        & intervals["metric"].eq("macro_f1")
    )
    intervals.loc[mask, "ci_lower"] = -0.001
    intervals.loc[mask, "interval_contains_zero"] = True
    scorecard, _ = apply_ultimate_judgement(
        scorecard,
        interval_summary=intervals,
        robustness=tables["robustness"],
        spec=load_ultimate_judgement_spec(),
    )

    rejected = build_rejected_alternatives(project_root=ROOT, scorecard=scorecard)

    assert not rejected["alternative"].str.startswith("C2 small-stem").any()
    assert rejected["alternative"].str.startswith("I2 ArticleType auxiliary lambda 0.3").any()


def test_selection_freeze_allows_identical_retry_and_rejects_changed_choice(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection_freeze.json"
    payload = {"selected_candidate": "I2", "holdout_opened": False}

    judgement_module._write_immutable_freeze(path, payload)
    first_bytes = path.read_bytes()
    judgement_module._write_immutable_freeze(path, dict(payload))

    assert path.read_bytes() == first_bytes
    with pytest.raises(ValueError, match="different content"):
        judgement_module._write_immutable_freeze(
            path,
            {"selected_candidate": "C2", "holdout_opened": False},
        )
    assert json.loads(path.read_text(encoding="utf-8"))["selected_candidate"] == "I2"


def test_ultimate_judgement_config_rejects_a_holdout_refit_boundary(tmp_path: Path) -> None:
    payload = json.loads(
        judgement_module.ULTIMATE_JUDGEMENT_CONFIG_PATH.read_text(encoding="utf-8")
    )
    payload["refit"]["dataset"] = "development_and_holdout_rows"
    path = tmp_path / "unsafe_refit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="development-only refit contract"):
        load_ultimate_judgement_spec(path)


def test_evidence_build_is_recoverable_immutable_and_strictly_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        judgement_module,
        "capture_git_state",
        lambda _root: {"commit": "a" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        judgement_module,
        "verify_implementation_at_head",
        lambda *paths, root: tuple(str(path) for path in paths),
    )
    parent = ROOT / "tmp/task2/test-ultimate-judgement"
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=parent) as temporary:
        workspace = Path(temporary)
        evidence = workspace / "evidence"
        freeze = workspace / "selection_freeze.json"
        timestamp = "2026-08-30T10:00:00Z"

        first = build_ultimate_judgement_evidence(
            project_root=ROOT,
            evidence_directory=evidence,
            freeze_path=freeze,
            recorded_at_utc=timestamp,
        )
        manifest = evidence / "manifest.json"
        first_manifest_bytes = manifest.read_bytes()
        first_freeze_bytes = freeze.read_bytes()

        manifest.unlink()
        recovered = build_ultimate_judgement_evidence(
            project_root=ROOT,
            evidence_directory=evidence,
            freeze_path=freeze,
            recorded_at_utc=timestamp,
        )
        retried = build_ultimate_judgement_evidence(
            project_root=ROOT,
            evidence_directory=evidence,
            freeze_path=freeze,
            recorded_at_utc=timestamp,
        )

        assert manifest.read_bytes() == first_manifest_bytes
        assert freeze.read_bytes() == first_freeze_bytes
        assert recovered["manifest_sha256"] == first["manifest_sha256"]
        assert retried["selection_freeze_sha256"] == first["selection_freeze_sha256"]
        verified_freeze, verified_freeze_path = load_verified_selection_freeze(
            freeze,
            project_root=ROOT,
        )
        verified_manifest, verified_manifest_path, artifacts = (
            load_verified_ultimate_judgement_manifest(
                manifest,
                project_root=ROOT,
            )
        )
        assert verified_freeze["selected_model"]["candidate"] == "I2"
        assert verified_freeze["refit_rule"]["epochs"] == 24
        assert verified_freeze["holdout_opened"] is False
        assert verified_freeze_path == freeze
        assert verified_manifest["selected_candidate"] == "I2"
        assert verified_manifest_path == manifest
        assert artifacts["selection_freeze"] == freeze

        with pytest.raises(ValueError, match="recorded decision timestamp"):
            build_ultimate_judgement_evidence(
                project_root=ROOT,
                evidence_directory=evidence,
                freeze_path=freeze,
                recorded_at_utc="2026-08-30T11:00:00Z",
            )
        copied_config = workspace / "g7_ultimate_judgement.json"
        shutil.copyfile(judgement_module.ULTIMATE_JUDGEMENT_CONFIG_PATH, copied_config)
        with pytest.raises(ValueError, match="config or Grad-CAM"):
            build_ultimate_judgement_evidence(
                project_root=ROOT,
                analysis_config_path=copied_config,
                evidence_directory=evidence,
                freeze_path=freeze,
                recorded_at_utc=timestamp,
            )
        copied_gradcam = workspace / "gradcam_manifest.json"
        shutil.copyfile(ROOT / judgement_module.DEFAULT_GRADCAM_MANIFEST, copied_gradcam)
        with pytest.raises(ValueError, match="config or Grad-CAM"):
            build_ultimate_judgement_evidence(
                project_root=ROOT,
                gradcam_manifest_path=copied_gradcam,
                evidence_directory=evidence,
                freeze_path=freeze,
                recorded_at_utc=timestamp,
            )
        assert manifest.read_bytes() == first_manifest_bytes
        assert freeze.read_bytes() == first_freeze_bytes

        original_payload = json.loads(freeze.read_text(encoding="utf-8"))
        bad_boolean = dict(original_payload)
        bad_boolean["selection_checks"] = dict(original_payload["selection_checks"])
        bad_boolean["selection_checks"]["jpeg_drop_guard"] = "false"
        bad_boolean_path = workspace / "bad_boolean_freeze.json"
        bad_boolean_path.write_text(json.dumps(bad_boolean), encoding="utf-8")
        with pytest.raises(ValueError, match="strict booleans"):
            load_verified_selection_freeze(bad_boolean_path, project_root=ROOT)

        missing_check = dict(original_payload)
        missing_check["selection_checks"] = dict(original_payload["selection_checks"])
        del missing_check["selection_checks"]["jpeg_drop_guard"]
        missing_check_path = workspace / "missing_check_freeze.json"
        missing_check_path.write_text(json.dumps(missing_check), encoding="utf-8")
        with pytest.raises(ValueError, match="selection_checks fields changed"):
            load_verified_selection_freeze(missing_check_path, project_root=ROOT)

        unsafe_refit = dict(original_payload)
        unsafe_refit["refit_rule"] = dict(original_payload["refit_rule"])
        unsafe_refit["refit_rule"]["dataset"] = "development_and_holdout_rows"
        unsafe_refit_path = workspace / "unsafe_refit_freeze.json"
        unsafe_refit_path.write_text(json.dumps(unsafe_refit), encoding="utf-8")
        with pytest.raises(ValueError, match="development-only refit rule"):
            load_verified_selection_freeze(unsafe_refit_path, project_root=ROOT)
