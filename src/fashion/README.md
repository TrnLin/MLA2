# Fashion package

Reusable project code lives here. Notebooks import it instead of copying shared logic.

- `data/` owns teacher audits, the sole split, fold loaders, image loading, and the
  protected final-evaluation boundary.
- `eda/` owns reusable calculations, diagnostics, provenance, and plotting helpers.
- `train/` owns the run registry. Every real training run must be recorded there.
- `retrieval/` owns reusable visual-search logic once Task 4 choices are made.

Dataset callers pass transforms explicitly. Shared code does not choose one for them.
