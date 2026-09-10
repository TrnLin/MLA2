"""Verify frozen E1 artifacts and separately check adapted inference scripts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = Path("reports/task3/usage_final_e1_20260907")


def references(value, key=None):
    """Yield every hash-bearing reference, including paths used as JSON keys."""
    if isinstance(value, dict):
        if "sha256" in value:
            yield value.get("path", key), value["sha256"]
        for child_key, child in value.items():
            yield from references(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def resolve_frozen(path, mapping, root=ROOT):
    """Resolve evaluated sources to archives and other artifacts to moved paths."""
    if path in mapping["evaluated_sources"]:
        return root / mapping["evaluated_sources"][path]["archive_path"]
    return root / path.replace("reports/task3_", "reports/task3/", 1)


def check_hash(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}; expected {expected}, got {actual}")


def verify(root=ROOT):
    manifest = json.loads((root / PACK / "model_manifest.json").read_text())
    mapping = json.loads((root / PACK / "source-archive-map.json").read_text())
    check_hash(root / PACK / "model_manifest.json", mapping["frozen_manifest_sha256"])
    frozen = list(references(manifest))
    if len(frozen) != 38:
        raise ValueError(f"Expected 38 frozen references, found {len(frozen)}")
    for original, entry in mapping["evaluated_sources"].items():
        if manifest["source_contracts"][original]["sha256"] != entry["evaluated_sha256"]:
            raise ValueError(f"Archive mapping disagrees with frozen manifest: {original}")
    for path, expected in frozen:
        check_hash(resolve_frozen(path, mapping, root), expected)
    for entry in mapping["evaluated_sources"].values():
        check_hash(root / entry["current_path"], entry["current_sha256"])
    return {
        "frozen_references_verified": len(frozen),
        "adapted_scripts_verified_separately": len(mapping["evaluated_sources"]),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
