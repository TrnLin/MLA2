# Data-preparation figures

The official `notebooks/01_data_preparation.ipynb` generates all report-cited preparation and exploratory
figures here during **Run All**.

The plots use training data for modelling evidence. Validation appears only in
normalized development diagnostics and as Task 4 queries against a train-only
gallery. Holdout and quarantine outcomes stay closed. Official prediction images
appear only in label-free duplicate triage. Every cross-role automatic match is quarantined without
reading a human decision. Remaining human questions are non-blocking and listed only in
`docs/reviews/open_decisions.md`.
