# Results

Store experiment records and report evidence here.

Generated training runs belong in `runs.csv`. Report-ready plots belong in
`figures/`; compact machine-readable evidence belongs in `evidence/`.

The accepted Task 3 models are packaged with real weights and loading instructions:

- [Gender model](task3/gender_model/README.md): SAM25 full-development refit.
- [Usage model](task3/usage_model/README.md): E8 full-development refit.

Each folder contains `final_epoch.pt`, `config.json`, `normalization.json`,
`class_names.json`, and a `package_manifest.json` with file hashes and source paths.
The loading examples use this repository's `src/fashion` code.
