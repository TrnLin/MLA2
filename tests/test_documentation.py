from fashion.config import ROOT


def test_assignment_breakdown_uses_canonical_data_contract():
    document = (ROOT / "docs/assignment-breakdown.html").read_text(encoding="utf-8")

    for stale_claim in (
        "01_preprocessing.ipynb",
        "clean.csv",
        "train | val | test",
        "pixels, every image",
        "Every image is 4,800 pixels",
    ):
        assert stale_claim not in document

    for current_claim in (
        "scripts/prepare_data.py",
        "data/processed/splits.csv",
        "train | val | holdout | quarantine",
        "38,595",
        "17 at other resolutions",
    ):
        assert current_claim in document
