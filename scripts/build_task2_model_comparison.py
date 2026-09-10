"""Build the complete Task 2 development OOF comparison evidence."""

from __future__ import annotations

import json

from fashion.task2.model_comparison import build_development_model_comparison_evidence


def main() -> int:
    """Verify all experiment manifests and write the comparison artifacts."""
    manifest = build_development_model_comparison_evidence()
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "manifest_sha256": manifest["manifest_sha256"],
                "model_count": manifest["model_count"],
                "source_partition": manifest["source_partition"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
