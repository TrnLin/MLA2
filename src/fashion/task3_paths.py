"""Read historical Task 3 report paths without rewriting saved receipts."""

from pathlib import Path

from fashion.config import ROOT


def resolve_task3_path(path: str | Path, *, root: Path = ROOT) -> Path:
    """Map the former report folders and preserve all other path semantics."""
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            return path
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "reports" and parts[1].startswith("task3_"):
        path = Path("reports", "task3", parts[1].removeprefix("task3_"), *parts[2:])
    return root / path
