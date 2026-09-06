"""Execute the delivered Colab setup locally, stopping at the real GPU training boundary."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/task3_usage_expanded_v2_e8_20260906"
NOTEBOOK = ROOT / "notebooks/04al_task3_usage_expanded_v2_e8.ipynb"
BUNDLE = REPORT / "teacher_plus_rare_usage_v2_training.zip"


class StopBeforeTraining(Exception):
    pass


class LocalContentPaths(ast.NodeTransformer):
    def __init__(self, content):
        self.content = content

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value.startswith("/content"):
            node.value = str(self.content) + node.value[len("/content") :]
        return node


def main():
    import torch

    original_cwd = Path.cwd()
    notebook = json.loads(NOTEBOOK.read_text())
    with tempfile.TemporaryDirectory(prefix="usage-v2-colab-check-") as temporary:
        content = Path(temporary) / "content"
        data = content / "drive/MyDrive/MLA2/data"
        data.mkdir(parents=True)
        (data / BUNDLE.name).symlink_to(BUNDLE)
        with (ROOT / "data/processed/splits.csv").open() as handle:
            teacher = next(
                row for row in csv.DictReader(handle) if row["partition"] == "development"
            )
        # One real teacher image is enough to exercise archive extraction. The package
        # builder separately hashes and decodes all 33,459 development images.
        with zipfile.ZipFile(data / "task3-data.zip", "w") as archive:
            archive.write(ROOT / teacher["path"], teacher["path"])
        mounts = []
        if "google" not in sys.modules:
            google = types.ModuleType("google")
            google.__path__ = []
            sys.modules["google"] = google
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=lambda *a, **k: mounts.append((a, k)))
        sys.modules["google.colab"] = colab
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_name = lambda index: "CPU setup check; GPU call intercepted"
        namespace = {}
        executed = []
        checked = {}

        def intercept_training(**kwargs):
            from fashion.train import task3_usage_expanded_v2 as v2

            assert tuple(kwargs["folds"]) == (0, 1, 2, 3, 4)
            assert kwargs["registry_path"] == v2.usage_registry_path(kwargs["output_root"])
            assert kwargs["registry_mirrors"] == (namespace["REPO_DIR"] / "results/runs.csv",)
            assert kwargs["resume"] is True
            references = v2.check_references(
                root=kwargs["root"],
                e8_directory=kwargs["e8_directory"],
                source_registry_path=kwargs["source_registry_path"],
                previous_directory=kwargs["previous_directory"],
                previous_registry_path=kwargs["previous_registry_path"],
            )
            assert {name: len(rows) for name, rows in references.items()} == {
                "teacher_e8": 5,
                "previous_expansion": 5,
            }
            assert not kwargs["output_root"].exists()
            checked.update(
                folds=list(kwargs["folds"]),
                reference_folds_verified=10,
                separate_v2_registry=True,
                local_registry_mirror=True,
                training_call_reached=True,
                production_training_started=False,
            )
            raise StopBeforeTraining

        try:
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                if cell["id"] == "usage-v2-train":
                    namespace["run_expanded_usage_v2"] = intercept_training
                tree = LocalContentPaths(content).visit(ast.parse("".join(cell["source"])))
                executed.append(cell["id"])
                try:
                    exec(compile(ast.fix_missing_locations(tree), cell["id"], "exec"), namespace)
                except StopBeforeTraining:
                    break
            assert checked["training_call_reached"]
            # Re-running setup must reuse the exact checked extraction without overwriting it.
            setup = next(cell for cell in notebook["cells"] if cell["id"] == "usage-v2-unpack")
            tree = LocalContentPaths(content).visit(ast.parse("".join(setup["source"])))
            exec(compile(ast.fix_missing_locations(tree), setup["id"], "exec"), namespace)
            with io.BytesIO() as buffer:
                with zipfile.ZipFile(buffer, "w") as archive:
                    archive.writestr("../outside.txt", "blocked")
                buffer.seek(0)
                with zipfile.ZipFile(buffer) as archive:
                    try:
                        namespace["safe_members"](archive)
                    except ValueError:
                        checked["unsafe_archive_path_rejected"] = True
                    else:
                        raise AssertionError("Notebook accepted a path outside the extraction")
            assert (namespace["REPO_DIR"] / teacher["path"]).read_bytes() == (
                ROOT / teacher["path"]
            ).read_bytes()
            checked.update(
                notebook=str(NOTEBOOK.relative_to(ROOT)),
                notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
                bundle_sha256=hashlib.sha256(BUNDLE.read_bytes()).hexdigest(),
                code_cells_executed=executed,
                setup_rerun_passed=True,
                teacher_archive_fixture_images=1,
                mount_calls=len(mounts),
                dataset=namespace["contract"],
                scope="Local Colab setup simulation; GPU fitting is covered by separate CPU tests",
            )
        finally:
            os.chdir(original_cwd)
        (REPORT / "colab_bootstrap_check.json").write_text(json.dumps(checked, indent=2) + "\n")
        print("Colab setup, both references, training arguments and repeat setup passed.")


if __name__ == "__main__":
    main()
