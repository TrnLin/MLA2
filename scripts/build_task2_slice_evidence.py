"""Build Task 2 shortcut-slice and error evidence from frozen OOF files only."""

from __future__ import annotations

import json
from typing import Sequence

from fashion.task2.slice_evidence import build_shortcut_error_slice_evidence


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the closed G5 boundary and write the deterministic G6 evidence."""
    if argv:
        raise ValueError("slice evidence uses only the frozen G6 declaration")
    manifest = build_shortcut_error_slice_evidence()
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "manifest_sha256": manifest["manifest_sha256"],
                "candidate_selection_affected": manifest["candidate_selection_affected"],
                "ultimate_winner_frozen": manifest["ultimate_winner_frozen"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
