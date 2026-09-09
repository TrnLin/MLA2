# Backend API

This folder receives images from `fe/` and calls the model adapters in
`core/src/fashion/demo.py`. Each model uses its own saved image settings.

From the repository root:

```sh
./.venv/bin/python -m pip install -e "./core[dev]" -e ./be
./.venv/bin/python -m uvicorn fashion_api.api:app --host 127.0.0.1 --port 8000
```

Check http://127.0.0.1:8000/api/health. All five models should show `ready`.

## Local model files

All of these paths are inside `core/`:

- `models/task1_article_type.pt` and its manifest/support files.
- `models/task2_season.pt`, its manifest/support files, and
  `results/evidence/task2/final_handoff/registry_snapshot.csv`.
- `model-weight/task3/gender_model/` and `model-weight/task3/usage_model/`.
- `model-weight/task4_r5/` and `models/task4_teacher_gallery/`.
- `data/train/images_train/<id>.jpg` and `data/processed/splits.csv`.

Uploads are retained in `be/tmp/demo-api/uploads/`. Set `FASHION_UPLOAD_DIR` to
an absolute path to use another folder. Verified model snapshots are retained
in `core/tmp/demo-api/snapshots/`. Clearing the page does not delete these files.

## Code and checks

- `src/fashion_api/api.py`: HTTP routes and result caching.
- `src/fashion_api/storage.py`: image validation and original upload bytes.
- `src/fashion_api/config.py`: upload location.
- `tests/`: API checks and real model integration checks.

From the repository root:

```sh
./.venv/bin/python -m pytest be/tests -q
```

## Task 4 crop request

`GET /api/images/{image_id}/similar?limit=5` searches the whole image. Add all four
fields to search a 3:4 crop in EXIF-oriented original pixels:

```text
crop_left=190&crop_top=130&crop_right=730&crop_bottom=850
```

The API rejects partial, non-integer, out-of-bounds or non-3:4 crops with 422.
The response includes `crop` (coordinates or null). Crop coordinates are part of
the result cache key. The original image bytes/hash stay intact, including the
checks that exclude protected images and duplicate/family gallery matches.
R5 preprocessing and model weights stay unchanged.
