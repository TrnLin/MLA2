from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import fashion.train.registry as registry_module


def test_shared_registry_imports_when_fcntl_is_unavailable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['fcntl'] = None; "
                "import fashion.train.registry"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_union_registry_write_tolerates_unavailable_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "runs.csv"
    real_open = os.open

    def deny_directory_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        if Path(path) == tmp_path:
            raise PermissionError("directory descriptors are unavailable")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(registry_module.os, "open", deny_directory_open)

    registry_module._write_union_rows_with_csv(registry_path, [])

    assert registry_path.is_file()
