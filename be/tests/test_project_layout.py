"""The workspace stays portable across working folders and historical Git layouts."""
import os
import subprocess
import sys
from pathlib import Path

from fashion.config import ROOT


def test_imports_and_model_paths_work_from_every_workspace_folder():
    workspace = ROOT.parent
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(
        str(workspace / path) for path in ('core/src', 'be/src'))}
    for cwd in (workspace, ROOT, workspace / 'be', workspace / 'fe', ROOT / 'notebooks'):
        result = subprocess.run(
            [sys.executable, '-c',
             'from fashion.config import ROOT; from fashion_api.api import app; '
             'assert (ROOT / "data/processed/splits.csv").is_file(); print(ROOT)'],
            cwd=cwd, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()) == ROOT
