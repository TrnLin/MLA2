# Fashion package

Reusable project code lives here. Notebooks import it instead of copying shared logic.

- `config.py` owns shared paths, target names, split sizes, and fixed seeds.
- `data/` owns teacher audits, metadata cleanup, the sole split, fold loaders, image
  loading, data evidence, and the protected final-evaluation boundary.

Update this if you are adding more scripts.