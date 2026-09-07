from __future__ import annotations

import subprocess
import sys


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
