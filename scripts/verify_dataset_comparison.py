from __future__ import annotations

import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/figures/dataset-comparison/comparison-summary.json"
REPORT = ROOT / "docs/dataset-quality-comparison.md"
NOTEBOOK = ROOT / "notebooks/00_dataset_comparison.ipynb"
FIGURE_DIRECTORY = ROOT / "results/figures/dataset-comparison"


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    report = REPORT.read_text(encoding="utf-8")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    failures: list[str] = []

    pop = data["populations"]
    ids = data["id_reconciliation"]
    loss = data["teacher_train_image_loss"]
    artifacts = data["artifact_consistency"]
    images = data["images"]

    require(
        pop["original_metadata_rows"]
        == pop["teacher_train_metadata_rows"] + pop["teacher_test_rows"],
        "original row count does not equal teacher train + test",
        failures,
    )
    require(ids["original_equals_teacher_union"] is True, "ID union proof failed", failures)
    require(ids["teacher_train_test_overlap"] == 0, "teacher train/test IDs overlap", failures)
    require(
        sum(ids["shared_label_mismatch_counts"].values()) == 0,
        "shared teacher/original labels differ",
        failures,
    )
    require(
        loss["present_images"] + loss["missing_images"] == loss["metadata_rows"],
        "teacher-train image coverage does not reconcile",
        failures,
    )
    require(
        loss["recoverable_from_original"] + len(loss["unrecoverable_known_orphans"])
        == loss["missing_images"],
        "teacher missing-image recovery counts do not reconcile",
        failures,
    )
    require(not artifacts["csv_without_link"], "original CSV rows lack URL records", failures)
    require(not artifacts["link_without_csv"], "URL records lack original CSV rows", failures)
    require(
        not artifacts["json"]["csv_without_json"],
        "original CSV rows lack JSON records",
        failures,
    )
    require(
        not artifacts["json"]["json_without_csv"],
        "JSON records lack original CSV rows",
        failures,
    )
    require(
        not images["audits"]["teacher_train_present"]["decode_errors"],
        "teacher-train image decode failures found",
        failures,
    )
    require(
        not images["audits"]["teacher_test"]["decode_errors"],
        "teacher-test image decode failures found",
        failures,
    )
    require(
        images["same_id_comparison"]["sample_size"] >= 500,
        "same-ID comparison sample is smaller than planned",
        failures,
    )
    require(
        images["near_duplicate_sample"]["sample_size"] >= 2000,
        "near-duplicate sample is smaller than planned",
        failures,
    )

    required_figures = {
        "population-coverage.png",
        "label-shift.png",
        "article-type-long-tail.png",
        "label-review-candidates.png",
    }
    present_figures = {path.name for path in FIGURE_DIRECTORY.glob("*.png")}
    require(required_figures <= present_figures, "one or more report figures are missing", failures)

    quoted_numbers = [
        pop["original_metadata_rows"],
        pop["teacher_train_metadata_rows"],
        pop["teacher_test_rows"],
        loss["missing_images"],
        images["exact_duplicates"]["groups"],
    ]
    for value in quoted_numbers:
        require(f"{value:,}" in report, f"report does not quote measured value {value:,}", failures)

    require(len(notebook.cells) >= 12, "comparison notebook is incomplete", failures)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    require(code_cells, "comparison notebook has no code cells", failures)
    require(
        all(cell.execution_count is not None for cell in code_cells),
        "comparison notebook was not executed from top to bottom",
        failures,
    )
    require(
        not (ROOT / "data/processed/splits.csv").exists(),
        "audit unexpectedly created a dataset split",
        failures,
    )

    if failures:
        print("FAIL")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)

    print(
        "OK — IDs, labels, image coverage, figures, report claims, and "
        f"{len(code_cells)} executed notebook cells reconcile"
    )


if __name__ == "__main__":
    main()
