"""Run the delivered Colab setup locally and stop before GPU training."""

from fashion.task3_paths import resolve_task3_path

import ast
import csv
import hashlib
import json
import os
import sys
import tempfile
import types
import zipfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
REPORT = ROOT / "reports/task3/usage_mixup_sam_20260906"
NOTEBOOK = ROOT / "notebooks/task3_training/usage_mixup_sam_screen.ipynb"
BUNDLE = REPORT / "usage_mixup_sam_training.zip"


class StopBeforeTraining(Exception):
    pass


class ContentPaths(ast.NodeTransformer):
    def __init__(self, content):
        self.content = str(content)

    def visit_Constant(self, node):
        # Validate the committed local code without depending on a published branch.
        if node.value == "https://github.com/TrnLin/MLA2.git":
            node.value = str(ROOT)
        if isinstance(node.value, str) and node.value.startswith("/content"):
            node.value = self.content + node.value[len("/content") :]
        return node


def main():
    import torch

    original_cwd = Path.cwd()
    notebook = json.loads(NOTEBOOK.read_text())
    checked = {}
    with tempfile.TemporaryDirectory(prefix="usage-sam-colab-check-") as temporary:
        content = Path(temporary) / "content"
        data = content / "drive/MyDrive/MLA2/data"
        data.mkdir(parents=True)
        (data / BUNDLE.name).symlink_to(BUNDLE)
        with (ROOT / "data/processed/splits.csv").open() as handle:
            teacher = next(r for r in csv.DictReader(handle) if r["partition"] == "development")
        with zipfile.ZipFile(data / "task3-data.zip", "w") as archive:
            archive.write(resolve_task3_path(teacher["path"], root=ROOT), teacher["path"])
        if "google" not in sys.modules:
            google = types.ModuleType("google")
            google.__path__ = []
            sys.modules["google"] = google
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=lambda *a, **k: None)
        sys.modules["google.colab"] = colab
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_name = lambda index: "CPU setup simulation"
        namespace = {}
        executed = []

        def intercept_training(**kwargs):
            from fashion.train import task3_usage_mixup_sam as screen

            assert kwargs["folds"] == (0, 4)
            assert kwargs["registry_mirrors"] == (
                namespace["REPO_DIR"] / "screen_registry/results/runs.csv",
            )
            references = screen.check_reference(
                directory=kwargs["baseline_directory"],
                registry_path=kwargs["baseline_registry_path"],
                splits=namespace["splits"],
            )
            assert set(references) == {0, 4}
            assert not kwargs["output_root"].exists()
            checked.update(
                folds=[0, 4],
                baseline_folds_verified=2,
                training_call_reached=True,
                production_training_started=False,
                separate_screen_registry=True,
            )
            raise StopBeforeTraining

        def execute(cell):
            tree = ContentPaths(content).visit(ast.parse("".join(cell["source"])))
            exec(compile(ast.fix_missing_locations(tree), cell["id"], "exec"), namespace)

        try:
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                if cell["id"] == "usage-sam-08":
                    namespace["run_usage_mixup_sam"] = intercept_training
                executed.append(cell["id"])
                try:
                    execute(cell)
                except StopBeforeTraining:
                    break
            assert checked["training_call_reached"]
            execute(notebook["cells"][2])
            assert (namespace["REPO_DIR"] / teacher["path"]).read_bytes() == (
                resolve_task3_path(teacher["path"], root=ROOT)
            ).read_bytes()
            checked.update(
                notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
                bundle_sha256=hashlib.sha256(BUNDLE.read_bytes()).hexdigest(),
                code_cells_executed=executed,
                setup_rerun_passed=True,
                teacher_archive_fixture_images=1,
                scope="Local setup simulation only; real training is not simulated as complete",
            )
        finally:
            os.chdir(original_cwd)
    (REPORT / "bootstrap_check_current.json").write_text(json.dumps(checked, indent=2) + "\n")
    print("Delivered Colab setup and both v2 references passed; stopped before training.")


if __name__ == "__main__":
    main()
