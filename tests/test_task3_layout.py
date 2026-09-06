"""The notebook move must preserve every experiment and historical result path."""

import ast
import csv
import hashlib
import json
from pathlib import Path

import nbformat

from fashion.config import ROOT
from fashion.task3_paths import resolve_task3_path


def test_historical_receipt_paths_keep_their_file_identity(tmp_path):
    target = tmp_path / "reports/task3/saved_run/model_manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"run_id": "original-run-id"}')
    old = "reports/task3_saved_run/model_manifest.json"
    assert resolve_task3_path(old, root=tmp_path) == target
    assert resolve_task3_path(tmp_path / old, root=tmp_path) == target
    assert resolve_task3_path(target, root=tmp_path).read_text() == target.read_text()
    assert resolve_task3_path("results/runs.csv", root=tmp_path) == tmp_path / "results/runs.csv"
    assert resolve_task3_path(Path("/another/checkout/file"), root=tmp_path) == Path(
        "/another/checkout/file"
    )


def test_all_companion_notebooks_have_one_retained_successor():
    folder = ROOT / "notebooks/task3_training"
    with (folder / "moves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    assert len({row["original_notebook"] for row in rows}) == 40
    assert {row["retained_notebook"] for row in rows} == {
        path.name for path in folder.glob("*.ipynb")
    }
    for row in rows:
        assert not (ROOT / "notebooks" / row["original_notebook"]).exists()
        notebook = nbformat.read(folder / row["retained_notebook"], as_version=4)
        nbformat.validate(notebook)
        calls = []
        for cell in notebook.cells:
            if cell.cell_type == "code":
                tree = ast.parse(cell.source)
                calls.extend(
                    ast.dump(node)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id.startswith("run_")
                    and node.func.id != "run_checked"
                )
        assert (
            hashlib.sha256(json.dumps(calls).encode()).hexdigest() == row["training_calls_sha256"]
        )
