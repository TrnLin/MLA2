from __future__ import annotations

import csv
from pathlib import Path

from fashion.data.hashing import compute_sha256


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "results/evidence/task2/eda_figure_registry.csv"
EXPECTED_EVIDENCE = {
    "acquisition shortcut",
    "file-size shortcut",
    "ArticleType shortcut",
    "transform risk",
}


def test_task2_eda_figure_registry_matches_current_shared_figures() -> None:
    with REGISTRY_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["evidence", "path", "sha256", "claim_boundary"]
    assert {row["evidence"] for row in rows} == EXPECTED_EVIDENCE

    for row in rows:
        figure_path = (ROOT / row["path"]).resolve()
        assert figure_path.is_relative_to(ROOT.resolve())
        assert figure_path.is_file()
        assert row["sha256"] == compute_sha256(figure_path)
        assert row["claim_boundary"] == (
            "development-only description; not model accuracy or causality"
        )
