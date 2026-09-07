from __future__ import annotations

import csv
from pathlib import Path

from fashion.data.hashing import compute_sha256

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "results/evidence/task2/eda_figure_registry.csv"
EXPECTED_FIGURES = {
    "acquisition shortcut": (
        "results/figures/data_preparation/acquisition_shortcut_risk.png"
    ),
    "file-size shortcut": (
        "results/figures/data_preparation/season_file_size_shortcut.png"
    ),
    "ArticleType shortcut": (
        "results/figures/data_preparation/shortcut_risk_heatmaps.png"
    ),
    "transform risk": "results/figures/data_preparation/transform_risk.png",
}


def test_task2_eda_figure_registry_matches_current_shared_figures() -> None:
    with REGISTRY_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["evidence", "path", "sha256", "claim_boundary"]
    assert len(rows) == len(EXPECTED_FIGURES)
    assert len({row["evidence"] for row in rows}) == len(rows)
    assert {row["evidence"] for row in rows} == set(EXPECTED_FIGURES)

    for row in rows:
        assert row["path"] == EXPECTED_FIGURES[row["evidence"]]
        figure_path = (ROOT / row["path"]).resolve()
        assert figure_path.is_relative_to(ROOT.resolve())
        assert figure_path.is_file()
        assert row["sha256"] == compute_sha256(figure_path)
        assert row["claim_boundary"] == (
            "development-only description; not model accuracy or causality"
        )
