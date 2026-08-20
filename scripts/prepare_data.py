"""Rebuild processed manifests, the shared split, and train-only statistics."""

from __future__ import annotations

import argparse

from fashion.data.pipeline import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None, help="image worker threads")
    arguments = parser.parse_args()
    prepare_data(workers=arguments.workers)
    print("Data preparation complete: data/processed/splits.csv is the only shared split.")


if __name__ == "__main__":
    main()
