from __future__ import annotations

import ast
import json
import re
from pathlib import Path

NOTEBOOK = Path("notebooks/task-4/10_task4_part3_final_evaluation.ipynb")
SCORED_ARTIFACT_NAMES = {
    "deployment_summary.csv",
    "holdout_bootstrap_intervals.csv",
    "holdout_coverage.csv",
    "holdout_error_examples.csv",
    "holdout_error_routes.csv",
    "holdout_metrics.json",
    "holdout_per_query_family.csv",
    "holdout_per_query_primary.csv",
    "holdout_robustness_metrics.csv",
    "holdout_scorecard.csv",
    "holdout_selective_retrieval.csv",
    "holdout_slice_metrics.csv",
    "holdout_source_robustness.csv",
    "ultimate_judgement.json",
}
SCORED_ARTIFACT_READ_CALLS = {"open", "read_bytes", "read_csv", "read_json", "read_text"}
SECTION_HEADING = re.compile(r"^## ([1-9]\d*)\. .+$")


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def _call_name(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def test_notebook_never_unlocks_labels() -> None:
    source = _source()
    for forbidden in (
        "load_splits_for_final_evaluation",
        "evaluation_unlocked",
        "train_test_split",
        "score_holdout(",
        "build_blind_holdout_evidence(",
        ".predict(",
        "data/raw/teacher",
        "styles.csv",
    ):
        assert forbidden not in source


def test_notebook_has_exactly_fifteen_sections_in_order() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    section_numbers = [
        int(match.group(1))
        for cell in payload["cells"]
        if cell["cell_type"] == "markdown"
        for line in "".join(cell["source"]).splitlines()
        if (match := SECTION_HEADING.fullmatch(line))
    ]
    assert section_numbers == list(range(1, 16))


def test_notebook_reads_the_verified_manifest_before_scored_artifacts() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    gateway_calls: list[tuple[int, int, int]] = []
    scored_reads: list[tuple[int, int, int, str]] = []
    for cell_index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node)
            position = (cell_index, node.lineno, node.col_offset)
            if call_name == "load_verified_holdout_evaluation":
                gateway_calls.append(position)
            literal_names = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }
            accessed = literal_names & SCORED_ARTIFACT_NAMES
            if call_name in SCORED_ARTIFACT_READ_CALLS and accessed:
                scored_reads.append((*position, min(accessed)))

    assert len(gateway_calls) == 1
    assert scored_reads, "no direct scored-artifact read found"
    assert gateway_calls[0] < min(read[:3] for read in scored_reads)


def test_r5_hog_interval_is_not_called_equivalence() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    all_text = json.dumps(payload, ensure_ascii=False).lower()
    required = (
        "the interval does not separate r5 and hog; "
        "small effects in either direction remain possible."
    )
    assert all_text.count(required) >= 2
    for forbidden in (
        "statistically tied",
        "hog tie",
        "hog is tied",
        "tied under this interval",
        "r5 and hog are equivalent",
        "r5 is equivalent to hog",
        "hog is equivalent to r5",
    ):
        assert forbidden not in all_text


def test_notebook_has_no_empty_cells() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    empty = [
        index
        for index, cell in enumerate(payload["cells"])
        if not "".join(cell["source"]).strip()
    ]
    assert empty == []


def test_every_code_cell_is_followed_by_an_interpretation() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = payload["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        assert isinstance(cell.get("execution_count"), int), f"code cell {index}"
        assert all(
            output.get("output_type") != "error" for output in cell.get("outputs", [])
        ), f"code cell {index}"
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                assert output.get("name") in {"stdout", "stderr"}, f"code cell {index}"
        following = [c for c in cells[index + 1 :] if "".join(c["source"]).strip()]
        assert following, f"code cell {index} is last"
        assert following[0]["cell_type"] == "markdown", f"code cell {index}"
        interpretation = "".join(following[0]["source"])
        assert interpretation.startswith("**Interpretation.**"), f"code cell {index}"
        for label in ("Result", "Meaning", "Decision", "Limitation", "Trace"):
            assert f"**{label}:**" in interpretation, f"code cell {index}: {label}"


def test_notebook_cites_every_scored_artifact() -> None:
    source = _source()
    for name in (
        "holdout_scorecard.csv",
        "holdout_bootstrap_intervals.csv",
        "holdout_selective_retrieval.csv",
        "holdout_slice_metrics.csv",
        "holdout_robustness_metrics.csv",
        "holdout_source_robustness.csv",
        "ultimate_judgement.json",
        "unlock_receipt.json",
        "prediction_receipt.json",
        "evaluation_manifest.json",
    ):
        assert name in source, name


def test_notebook_states_the_no_refit_limitation() -> None:
    source = _source()
    assert "0026" in source
    assert "6,556" in source or "6556" in source


def test_notebook_discloses_the_historical_blind_input_proof_limit() -> None:
    source = _source().lower()
    for required in (
        "bf64d2b",
        "used six non-image inputs before it hashed them",
        "cannot prove that no file changed during the run",
        "images and rankings were byte-bound",
        "the scores stand",
        "future runs snapshot",
        "decision 0028",
    ):
        assert source.count(required) >= 2, required
    assert "proves that no file changed during the run" not in source
