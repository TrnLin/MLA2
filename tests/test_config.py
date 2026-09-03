from pathlib import Path

from fashion import config


def test_task2_paths_stay_inside_expected_repository_areas() -> None:
    expected = {
        config.TASK2_CONFIG_DIR: config.ROOT / "configs/task2",
        config.TASK2_TMP_DIR: config.ROOT / "tmp/task2",
        config.TASK2_CHECKPOINT_DIR: config.ROOT / "tmp/task2/checkpoints",
        config.TASK2_RUN_DIR: config.ROOT / "tmp/task2/runs",
        config.TASK2_EVIDENCE_DIR: config.ROOT / "results/evidence/task2",
        config.TASK2_FIGURE_DIR: config.ROOT / "results/figures/task2",
        config.RUNS_CSV: config.ROOT / "results/runs.csv",
        config.TASK2_MODEL_PATH: config.ROOT / "models/task2_season.pt",
        config.TASK2_MODEL_MANIFEST_JSON: config.ROOT / "models/task2_season.manifest.json",
        config.TASK2_SELECTION_FREEZE_JSON: (
            config.ROOT / "results/evidence/task2/selection_freeze.json"
        ),
        config.TASK2_NOTEBOOK_HTML: config.ROOT / "results/notebooks/03_task2_season.html",
    }

    for actual, wanted in expected.items():
        assert isinstance(actual, Path)
        assert actual == wanted
        assert actual.is_relative_to(config.ROOT)


def test_importing_config_does_not_create_task2_artifact_directories(tmp_path: Path) -> None:
    # Path constants describe ownership only. Writers create their exact parent directories.
    nonexistent = tmp_path / "task2"
    assert not nonexistent.exists()
    _ = config.TASK2_TMP_DIR
    assert not nonexistent.exists()
