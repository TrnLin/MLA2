from __future__ import annotations

import ast

from fashion.config import ROOT

LAUNCHER = ROOT / "scripts/build_task2_slice_evidence.py"


def test_slice_evidence_launcher_is_analysis_only_and_windows_safe() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "__main__"
            for comparator in node.test.comparators
        )
    ]
    imports = {
        alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names
    }

    assert len(guards) == 1
    assert "build_shortcut_error_slice_evidence" in imports
    assert "run_or_load" not in source
    assert 'mode="run"' not in source
