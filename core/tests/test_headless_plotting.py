from __future__ import annotations

import matplotlib


def test_pytest_uses_the_noninteractive_matplotlib_backend() -> None:
    assert matplotlib.get_backend().lower() == "agg"
