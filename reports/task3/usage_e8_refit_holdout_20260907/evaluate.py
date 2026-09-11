"""Recompute the selected Usage E8 refit's metrics from saved predictions."""

import json

from fashion.config import ROOT
from fashion.task3_refit_evaluation import load_selected_evaluation


def main():
    selected = load_selected_evaluation(ROOT)["Usage E8"]
    print(json.dumps(selected["metrics"], indent=2))


if __name__ == "__main__":
    main()
