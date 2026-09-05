"""Occasional grayscale preserves the 04aa recipe, RNG streams and evaluation rules."""

import copy
import hashlib
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from test_task3_gender_narrow import _case, _gates
from test_task3_gender_stronger_dropout import _completed_darkening

import fashion.train.task3_gender_dropout_darkening as shared
import fashion.train.task3_gender_grayscale as entry
from fashion.train.augmentation import (
    GRAYSCALE_AUGMENTATION,
    GRAYSCALE_PROBABILITY,
    GRAYSCALE_SEED_XOR,
    apply_training_augmentation,
)
from fashion.train.config import Task3BaselineConfig
from fashion.train.task3_dataset_v2 import (
    check_task3_dataset_v2_setup,
    dataset_v2_spec,
    run_task3_dataset_v2_screen,
)
from fashion.train.task3_gender_grayscale import (
    NAME,
    PARENT_RUN_IDS,
    check_gender_grayscale_sources,
    evaluate_gender_grayscale_screen,
    grayscale_config,
    run_gender_grayscale_screen,
)


def test_frozen_recipe_changes_only_augmentation_and_preserves_old_digests():
    parent = dataset_v2_spec(shared.NAME, shared.PARENT_RUN_IDS)
    spec = dataset_v2_spec(NAME, PARENT_RUN_IDS)
    identity = {
        "name",
        "experiment_id",
        "hypothesis_id",
        "artifact_dir",
        "run_prefix",
        "changed_factor",
        "parent_artifact_dir",
        "parent_run_ids",
    }
    assert {k for k in asdict(parent) if asdict(parent)[k] != asdict(spec)[k]} - identity == {
        "training_augmentation"
    }
    assert spec.classifier_dropout == 0.30
    assert spec.training_augmentation == GRAYSCALE_AUGMENTATION
    assert spec.to_dict()["grayscale_probability"] == GRAYSCALE_PROBABILITY == 0.10
    assert spec.to_dict()["grayscale_rng"].endswith(hex(GRAYSCALE_SEED_XOR))
    assert spec.parent_folds == (0, 4)
    assert spec.parent_run_id_for_fold(4) == PARENT_RUN_IDS[1]
    assert not spec.saved_tensors_on_cpu
    config = grayscale_config(spec, fold=4, device_name="cuda")
    assert config == Task3BaselineConfig(target="gender")
    for old, expected in [
        (parent, "cfbed3f0fed4"),
        (dataset_v2_spec(shared.STRONGER_NAME, PARENT_RUN_IDS), "3064b151dbc0"),
    ]:
        payload = {"baseline_controls": config.to_dict(), "child_experiment": old.to_dict()}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        assert digest == expected
    for changes in [
        {"classifier_dropout": 0.45},
        {"training_augmentation": parent.training_augmentation},
    ]:
        with pytest.raises(ValueError, match="predeclared"):
            replace(spec, **changes)
    for child, fold, device in [
        (parent, 0, "cuda"),
        (spec, 1, "cuda"),
        (spec, 0, "cpu"),
        (dataset_v2_spec(NAME, shared.PARENT_RUN_IDS), 0, "cuda"),
    ]:
        with pytest.raises(ValueError):
            grayscale_config(child, fold=fold, device_name=device)
    for function in [check_task3_dataset_v2_setup, run_task3_dataset_v2_screen]:
        with pytest.raises(ValueError, match="gender_grayscale"):
            function(NAME, parent_run_ids=PARENT_RUN_IDS, output_root="unused")


def test_grayscale_preserves_translation_darkening_and_replays_exactly():
    image = Image.fromarray(np.random.default_rng(3).integers(0, 256, (80, 60, 3), dtype=np.uint8))
    saved_state = random.getstate()
    try:
        random.seed(23)
        dark = random.Random(77)
        parent = [
            apply_training_augmentation(
                image, "translation_2px_p05_mild_darkening_p025", darkening_rng=dark
            )
            for _ in range(200)
        ]
        translation_state, dark_state = random.getstate(), dark.getstate()
        random.seed(23)
        dark, gray = random.Random(77), random.Random(99)
        child = [
            apply_training_augmentation(
                image, GRAYSCALE_AUGMENTATION, darkening_rng=dark, grayscale_rng=gray
            )
            for _ in range(200)
        ]
        assert random.getstate() == translation_state and dark.getstate() == dark_state
        oracle = random.Random(99)
        changed = [oracle.random() < 0.10 for _ in range(200)]
        assert 0 < sum(changed) < len(changed)
        assert gray.getstate() == oracle.getstate()
        for before, after, convert in zip(parent, child, changed, strict=True):
            expected = before.convert("L").convert("RGB") if convert else before
            assert after.mode == "RGB" and after.size == image.size
            assert np.array_equal(np.asarray(after), np.asarray(expected))
        random.seed(23)
        dark, gray = random.Random(77), random.Random(99)
        for expected in child:
            actual = apply_training_augmentation(
                image, GRAYSCALE_AUGMENTATION, darkening_rng=dark, grayscale_rng=gray
            )
            assert np.array_equal(np.asarray(expected), np.asarray(actual))
    finally:
        random.setstate(saved_state)


@pytest.mark.parametrize(
    "draw,converted", [(0.0, True), (0.099999, True), (0.10, False), (0.99, False)]
)
def test_grayscale_probability_boundary_and_required_streams(draw, converted):
    class Stream:
        def random(self):
            return draw

    image = Image.new("RGB", (60, 80), (220, 80, 25))
    with pytest.raises(ValueError, match="own seeded"):
        apply_training_augmentation(image, GRAYSCALE_AUGMENTATION)
    with pytest.raises(ValueError, match="darkening"):
        apply_training_augmentation(image, GRAYSCALE_AUGMENTATION, grayscale_rng=Stream())
    state = random.getstate()
    dark = random.Random(123)
    expected = apply_training_augmentation(
        image, "translation_2px_p05_mild_darkening_p025", darkening_rng=dark
    )
    random.setstate(state)
    actual = apply_training_augmentation(
        image, GRAYSCALE_AUGMENTATION, darkening_rng=random.Random(123), grayscale_rng=Stream()
    )
    expected = expected.convert("L").convert("RGB") if converted else expected
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_prerequisites_use_030_darkening_parents_and_require_the_audit(tmp_path, monkeypatch):
    previous, sources, classes, evidence, paths = _completed_darkening(tmp_path)
    sources["Drop30Dark"] = previous
    spec = dataset_v2_spec(NAME, PARENT_RUN_IDS)

    def checked(**kwargs):
        assert kwargs["experiment_name"] == NAME
        assert kwargs["darkening_directory"] == paths["darkening_directory"]
        return sources, classes, spec, evidence

    monkeypatch.setattr(shared, "check_gender_dropout_darkening_sources", checked)
    monkeypatch.setattr(entry, "check_gender_dropout_darkening_sources", checked)
    assert check_gender_grayscale_sources(**paths)[2] == spec
    monkeypatch.setattr(shared, "require_narrow_prerequisites", lambda *a, **kw: evidence)
    with pytest.raises(ValueError, match="required before training"):
        shared.require_dropout_darkening_prerequisites(None, spec=spec, fold=0)
    audit = tmp_path / "source_audit.json"
    shared._save_source_audit(
        audit,
        shared._source_identity(sources, spec, evidence, paths, root=tmp_path),
        registry_path=paths["source_registry_path"],
    )
    # Audits save path strings; accept the same path representation on the replay.
    paths = {key: str(value.resolve()) for key, value in paths.items()}
    result = shared.require_dropout_darkening_prerequisites(audit, spec=spec, fold=4, root=tmp_path)
    assert result["parent_directory"].name == PARENT_RUN_IDS[1]


@pytest.mark.parametrize("fault", [None, "stronger_parent", "missing_directory"])
def test_source_checker_verifies_original_and_direct_darkening_parents(
    tmp_path, monkeypatch, fault
):
    previous, sources, classes, evidence, paths = _completed_darkening(tmp_path)
    dropout = sources.pop("Drop30")
    parent_spec = dataset_v2_spec("gender_dropout_030", [f"G2-{f}" for f in range(5)])
    for fold, run in dropout.items():
        run.update(
            fold=fold,
            config={
                **Task3BaselineConfig(target="gender").to_dict(),
                "child_experiment": parent_spec.to_dict(),
                "parent_run_id": sources["G2"][fold]["run_id"],
                "training_precision_settings": shared.TRAINING_PRECISION,
                "precision_evidence_sha256": evidence["artifact_sha256"],
            },
        )
    if fault == "stronger_parent":
        previous[4]["config"]["child_experiment"]["classifier_dropout"] = 0.45
    elif fault == "missing_directory":
        paths.pop("darkening_directory")
    monkeypatch.setattr(
        shared,
        "check_gender_narrow_sources",
        lambda **kw: (sources, classes, parent_spec, evidence),
    )
    monkeypatch.setattr(shared, "load_splits", lambda *a: None)
    runs = {run["run_id"]: run for group in (dropout, previous) for run in group.values()}
    inspected = []

    def inspect(path, **kwargs):
        inspected.append(path.name)
        return runs[path.name]

    monkeypatch.setattr(shared, "inspect_gender_run", inspect)
    if fault:
        with pytest.raises(ValueError, match="configuration|directory is required"):
            check_gender_grayscale_sources(**paths, root=tmp_path)
    else:
        checked, _, spec, _ = check_gender_grayscale_sources(**paths, root=tmp_path)
        assert spec.name == NAME and spec.parent_run_ids == PARENT_RUN_IDS
        assert checked["Drop30Dark"][4]["run_id"] == PARENT_RUN_IDS[1]
        assert inspected == [*shared.PARENT_RUN_IDS, *PARENT_RUN_IDS]


def test_actual_model_and_direct_trainer_keep_dropout_and_require_audit(tmp_path):
    torch = pytest.importorskip("torch")
    from fashion.train.task3_baseline import _build_task3_model, run_task3_baseline_fold

    spec = dataset_v2_spec(NAME, PARENT_RUN_IDS)
    model = _build_task3_model(Task3BaselineConfig(target="gender"), spec)
    assert model.classifier_dropout.p == 0.30
    assert sum(p.numel() for p in model.parameters()) == 390181
    model.eval()
    pooled = torch.ones(8, 256)
    assert torch.equal(model.classifier_dropout(pooled), pooled)
    with pytest.raises(ValueError, match="required before training"):
        run_task3_baseline_fold("gender", 0, output_root=tmp_path, child_spec=spec, root=tmp_path)


@pytest.mark.parametrize("mode", ["fresh", "reuse", "bad_reuse", "memory"])
def test_runner_trains_only_two_folds_with_correct_augmentation_and_parent(
    tmp_path, monkeypatch, mode
):
    previous, sources, classes, evidence, paths = _completed_darkening(tmp_path)
    sources["Drop30Dark"] = previous
    spec = dataset_v2_spec(NAME, PARENT_RUN_IDS)
    child = copy.deepcopy(previous)
    audit = tmp_path / spec.artifact_dir / "gender/source_audit.json"
    for fold, run in child.items():
        run["run_id"] = f"gray-{fold}"
        run["predictions"]["run_id"] = run["run_id"]
        run["robustness"]["run_id"] = run["run_id"]
        run["config"].update(
            child_experiment=spec.to_dict(), parent_run_id=spec.parent_run_id_for_fold(fold)
        )
    events = []
    monkeypatch.setattr(
        shared,
        "check_gender_dropout_darkening_sources",
        lambda **kw: (sources, classes, spec, evidence),
    )
    monkeypatch.setattr(shared, "require_narrow_prerequisites", lambda *a, **kw: evidence)
    monkeypatch.setattr(shared, "load_splits", lambda *a: None)

    def result(fold):
        child[fold]["config"]["refinement_prerequisite_sha256"] = shared.compute_sha256(audit)
        return {"run_dir": str(tmp_path / f"gray-{fold}")}

    def reuse(spec, fold, **kw):
        if mode not in {"reuse", "bad_reuse"}:
            return None
        value = result(fold)
        if mode == "bad_reuse":
            child[fold]["config"]["child_experiment"]["grayscale_probability"] = 0.25
        return value

    def fit(**kw):
        fold = kw["validation_fold"]
        events.append(("fit", fold))
        assert kw["parent_run_directory"] == previous[fold]["directory"]
        assert kw["registry_path"] == paths["source_registry_path"]
        assert kw["registry_mirrors"] == (tmp_path / "mirror.csv",)
        assert kw["child_spec"].classifier_dropout == 0.30
        assert kw["child_spec"].training_augmentation == GRAYSCALE_AUGMENTATION
        if mode == "memory":
            child[fold]["metrics"]["peak_memory_bytes"] = 3_000_000_000
        return result(fold)

    def evaluate(run, **kw):
        events.append(("evaluate", run["run_id"]))
        return run

    monkeypatch.setattr(shared, "_reusable_fold", reuse)
    monkeypatch.setattr(shared, "_train_fold", fit)
    monkeypatch.setattr(shared, "inspect_gender_run", lambda path, **kw: child[int(path.name[-1])])
    monkeypatch.setattr(shared, "evaluate_gender_ieee", evaluate)
    screen, compare = shared.evaluate_gender_narrow_screen, shared.compare_with_dropout
    monkeypatch.setattr(
        shared, "evaluate_gender_narrow_screen", lambda *a, **kw: screen(*a, **kw, repetitions=20)
    )
    monkeypatch.setattr(shared, "compare_with_dropout", lambda *a: compare(*a, repetitions=20))
    args = dict(
        **paths,
        registry_path=paths["source_registry_path"],
        output_root=tmp_path,
        root=tmp_path,
        registry_mirrors=(tmp_path / "mirror.csv",),
    )
    if mode == "bad_reuse":
        with pytest.raises(ValueError, match="configuration"):
            run_gender_grayscale_screen(**args)
        assert not any(kind == "fit" for kind, value in events)
        return
    report = run_gender_grayscale_screen(**args)
    assert [v for k, v in events if k == "fit"] == (
        [] if mode == "reuse" else [0] if mode == "memory" else [0, 4]
    )
    assert events[:6] == [
        ("evaluate", sources[name][f]["run_id"])
        for name in ("G2", "E6", "Drop30Dark")
        for f in (0, 4)
    ]
    if mode != "memory":
        assert report["rule_version"] == "gdrop030darkgray010_loss003_gap005_v1"
        assert (
            report["direct_parent_comparison"]["training_augmentation"]
            == "translation_2px_p05_mild_darkening_p025"
        )
        assert "Drop30Dark" in report["incremental_comparison"]["comparison"]
        assert json.loads((audit.parent / "screen_decision.json").read_text()) == report


def test_screen_retains_all_gates_including_grayscale_and_gap():
    child, sources, classes = _case()
    for run in child.values():
        run["metrics"]["parameter_count"] = 390181
    result = evaluate_gender_grayscale_screen(child, sources, classes, repetitions=20)
    assert len(result["checks"]) == 19 and result["status"] == "pass"
    for run in child.values():
        run["metrics"]["final_train_eval_macro_f1"] += 0.08
        run["metrics"]["final_train_validation_macro_f1_gap"] += 0.08
    assert (
        _gates(evaluate_gender_grayscale_screen(child, sources, classes, repetitions=20))[
            "mean_gap_reduction"
        ]
        == "fail"
    )


def test_notebook_is_one_unexecuted_frozen_trial():
    root = Path(__file__).resolve().parents[2]
    nb = nbformat.read(root / "notebooks/04ac_task3_gender_grayscale_screen.ipynb", as_version=4)
    nbformat.validate(nb)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert code.count("run_gender_grayscale_screen(") == 1
    assert 'spec.to_dict()["grayscale_probability"] == 0.10' in code
    assert "spec.classifier_dropout == 0.30" in code
    assert "train_test_split" not in code and "pretrained=True" not in code
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "04ac", "exec")
            assert cell.execution_count is None and not cell.outputs


def test_actual_dataset_uses_persistent_separate_rng_and_evaluation_stays_color(tmp_path):
    torch = pytest.importorskip("torch")
    from fashion.train.data import Task3ImageDataset

    image = Image.new("RGB", (60, 80), (230, 40, 70))
    image.save(tmp_path / "input.png")
    frame = pd.DataFrame(
        [dict(id=1, gender="Girls", path="input.png", cv_fold=0, product_family_group="one")]
    )
    kwargs = dict(
        target="gender", label_to_index={"Girls": 0}, mean=[0, 0, 0], std=[1, 1, 1], root=tmp_path
    )
    torch.manual_seed(2753)
    train = Task3ImageDataset(frame, augmentation=GRAYSCALE_AUGMENTATION, **kwargs)
    clean = Task3ImageDataset(frame, **kwargs)
    oracle = random.Random(torch.initial_seed() ^ GRAYSCALE_SEED_XOR)
    decisions = [oracle.random() < 0.10 for _ in range(60)]
    outputs = [train[0]["image"] for _ in decisions]
    assert train._grayscale_rng.getstate() == oracle.getstate()
    assert train._darkening_rng is not train._grayscale_rng
    for output, gray in zip(outputs, decisions, strict=True):
        assert torch.equal(output[0], output[1]) == gray
    assert clean._grayscale_rng is None and clean._darkening_rng is None
    expected = clean[0]["image"]
    assert not torch.equal(expected[0], expected[1])
    assert torch.equal(expected, clean[0]["image"])
    with pytest.raises(ValueError, match="cannot be combined"):
        Task3ImageDataset(
            frame, augmentation=GRAYSCALE_AUGMENTATION, corruption="grayscale", **kwargs
        )
