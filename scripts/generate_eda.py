"""Regenerate protected EDA evidence and report figures."""

from fashion.eda.generate import generate_eda

if __name__ == "__main__":
    summary = generate_eda()
    print(
        f"EDA complete: {summary['scope']['modelling_rows']:,} training rows; "
        "holdout, quarantine, and prediction outcomes stayed closed."
    )
