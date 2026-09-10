# Fashion Intelligence

The project has three parts:

| Folder | Contents |
| --- | --- |
| [core](core/README.md) | Models, data, training code, notebooks, saved results, reports and assignment docs |
| [be](be/README.md) | Python API, upload storage and API tests |
| [fe](fe/README.md) | React website and sample images |

## Where to put the dataset

All dataset paths below start at this repository root. Extract the image files
before starting the app; a ZIP file alone is not enough.

**For the demo**, put the original teacher product photos here:

```text
core/
└── data/
    ├── train/
    │   └── images_train/
    │       ├── <id>.jpg
    │       └── ...
    └── processed/
        └── splits.csv       # Use the existing shared split.
```

The API reads `core/data/train/images_train/<id>.jpg` to show Task 4 matches.
Keep the original file names and image bytes. Avoid an extra nested
`images_train/` folder when extracting. Model weights go in the separate paths
listed in [be/README.md](be/README.md).

**For data preparation and model work**, the source dataset belongs here:

```text
core/data/raw/teacher/
├── train/
│   ├── styles_train.csv
│   └── images_train/
│       └── <id>.jpg
└── test/
    ├── styles_prediction.csv
    └── images_test/
        └── <id>.jpg
```

If the photos are on another drive, these folders can be symbolic links
(shortcuts to the real folders), so you do not need a second copy. Keep that
drive connected while using the app or notebooks. Raw datasets stay local and
are ignored by Git.

The optional Task 4 high-resolution dataset belongs in
`core/data/raw/external/fashion_product_images_v1/`, with `images.csv` and an
`images/` folder inside. It is only needed for the experiments described in
[core/README.md](core/README.md#task-4-external-high-resolution-images).

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
