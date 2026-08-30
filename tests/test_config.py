from fashion.config import (
    EVIDENCE_DIR,
    FIGURE_DIR,
    RESULTS_DIR,
    RUNS_CSV,
    TASK1_EVIDENCE_DIR,
    TASK1_FIGURE_DIR,
    TASK1_RESULT_DIR,
)


def test_task1_result_paths_are_nested_under_shared_result_roots() -> None:
    assert RUNS_CSV == RESULTS_DIR / "runs.csv"
    assert TASK1_RESULT_DIR == RESULTS_DIR / "task1"
    assert TASK1_FIGURE_DIR == FIGURE_DIR / "task1"
    assert TASK1_EVIDENCE_DIR == EVIDENCE_DIR / "task1"
