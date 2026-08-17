"""Generate the reproducible EDA evidence bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fashion.eda import run_eda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="recompute image measurements even when cache provenance matches",
    )
    args = parser.parse_args()
    result = run_eda(refresh=args.refresh)
    print(f"EDA outputs written to {result.output_dir}")
    print(f"Image cache status: {result.cache_status}")


if __name__ == "__main__":
    main()
