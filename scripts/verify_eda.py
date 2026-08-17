"""Verify an EDA evidence bundle without creating or changing a split."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion.config import EDA_OUTPUT_DIR, PROCESSED_DATA_DIR
from fashion.eda import EDA_REVIEW_GRIDS, GENERATED_EDA_OUTPUTS


REQUIRED_OUTPUTS = GENERATED_EDA_OUTPUTS


def verify_eda(output_dir: Path = EDA_OUTPUT_DIR, *, default_real_output: bool = False) -> list[str]:
    """Return reconciliation failures; real-dataset totals apply only to default outputs."""
    output_dir = Path(output_dir)
    failures: list[str] = []
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return [f"missing summary: {summary_path}"]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"invalid summary: {error}"]
    names = {path.name for path in output_dir.iterdir() if path.is_file()}
    missing = REQUIRED_OUTPUTS.difference(names)
    if missing:
        failures.append(f"missing required outputs: {sorted(missing)}")
    empty = sorted(
        name for name in REQUIRED_OUTPUTS.intersection(names)
        if (output_dir / name).stat().st_size == 0
    )
    if empty:
        failures.append(f"empty required outputs: {empty}")
    manifest = summary.get("output_manifest", {})
    if set(manifest.get("files", [])) != REQUIRED_OUTPUTS:
        failures.append("summary manifest does not match the generated-output contract")
    if set(manifest.get("review_grids", [])) != set(EDA_REVIEW_GRIDS):
        failures.append("summary manifest does not list all six review grids")
    if summary.get("test_quarantine", {}).get("overlap_count") != 0:
        failures.append("test IDs overlap the selected EDA population")
    split = summary.get("split_provenance", {})
    before = split.get("before", {})
    after = split.get("after", {})
    if split.get("unchanged") is not True:
        failures.append("EDA run reported split provenance changed")
    if before != after:
        failures.append("split before/after provenance does not match")
    split_path = PROCESSED_DATA_DIR / "splits.csv"
    if after.get("status") == "absent" and split_path.exists():
        failures.append("split was absent before EDA but now exists")
    if after.get("status") == "present":
        current = sha256(split_path.read_bytes()).hexdigest() if split_path.is_file() else None
        if current != after.get("sha256"):
            failures.append("split content changed after EDA")
    if default_real_output:
        population = summary.get("population", {})
        if population.get("source_train_ids") != 38_617:
            failures.append("default EDA source total is not 38,617")
        if population.get("usable_products") != 38_612:
            failures.append("default EDA usable total is not 38,612")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EDA_OUTPUT_DIR)
    args = parser.parse_args()
    failures = verify_eda(
        args.output_dir,
        default_real_output=args.output_dir.resolve() == EDA_OUTPUT_DIR.resolve(),
    )
    if failures:
        print("FAIL")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print("OK — EDA outputs, test quarantine, and split provenance reconcile")


if __name__ == "__main__":
    main()
