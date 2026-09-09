# Fashion Intelligence

The project has three parts:

| Folder | Contents |
| --- | --- |
| [core](core/README.md) | Models, data, training code, notebooks, saved results, reports and assignment docs |
| [be](be/README.md) | Python API, upload storage and API tests |
| [fe](fe/README.md) | React website and sample images |

## Run the demo

Use Python 3.12–3.14 and Node 22.18 or newer. The shared Python environment is
`.venv/` at this repository root. For a fresh setup, create the environment and
install both local packages:

```sh
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e "./core[dev]" -e ./be
```

Start the API from this folder:

```sh
./.venv/bin/python -m uvicorn fashion_api.api:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```sh
cd fe
npm install
npm run dev
```

Open http://127.0.0.1:5173/. Keep both terminals running. The API needs the local
model weights and product photos listed in [be/README.md](be/README.md).

## Model work

Run model commands and notebooks from `core/`. Saved paths such as
`data/processed/splits.csv` are relative to `core/`. The split, model weights and
saved evidence keep their original bytes. See [core/README.md](core/README.md)
for the notebook order and model setup.

Older training notebooks check for `core/.venv`. On macOS/Linux, create the link
once if it is missing: `ln -s ../.venv core/.venv`. This shares the root environment.

```sh
cd core
../.venv/bin/python -m jupyter lab
```

For a wheel install, set `FASHION_PROJECT_ROOT` to the absolute `core/` directory.
API uploads stay in `be/tmp/demo-api/uploads/`; verified model snapshots stay in
`core/tmp/demo-api/snapshots/`.

## Checks

From the repository root:

```sh
./.venv/bin/python -m pytest be/tests -q
./.venv/bin/python -m pytest core/tests -q
npm --prefix fe test
npm --prefix fe run build
```
