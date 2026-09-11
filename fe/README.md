# Thread — Fashion intelligence

An animated landing page and a React/TypeScript demo connected to five frozen local models.

## Pages

- `/`: Thread landing page, with GSAP scroll animation and Lenis smooth scrolling.
- `/demo`: the existing image upload, classification, and adjustable Task 4 search.

The landing page works without the API. Its motion switch and the system's
reduced-motion preference can turn animation off. Animation code loads only on
the landing page. Photos are served locally; their sources are recorded in
`public/landing/sources.json` and `public/samples/sources.json`.

The landing page and demo share the Style picker and its saved choice: Outfit,
Liquid Glass, Brutalist, Nocturne, Field Notes, Minimal, and Original Thread.
The landing treatments live in `src/landing-themes.css`.

When hosting the frontend, route both `/` and `/demo` to `index.html` so direct
links and refreshes work. Vite already does this during development and preview.

## Run locally

From the repository root, start the backend:

```sh
./.venv/bin/python -m pip install -e "./core[dev]" -e ./be
./.venv/bin/python -m uvicorn fashion_api.api:app --host 127.0.0.1 --port 8000
```

In another terminal, from the repository root:

```sh
cd fe
npm install
npm run dev
```

Open http://127.0.0.1:5173/. Keep both terminals running. Vite proxies `/api` to
port 8000 in development and preview mode. This setup stays on this computer.
Models load once per backend process on CPU with two PyTorch threads. Do not
use multiple workers unless memory and concurrency have been reviewed.

## Required local files

These paths are relative to `core/`:

- Task 1: `models/task1_article_type.pt` and its tracked manifest/support files.
- Task 2: `models/task2_season.pt`, its tracked manifest/support files, and the
  immutable `results/evidence/task2/final_handoff/registry_snapshot.csv`.
- Task 3: `model-weight/task3/gender_model/` and `model-weight/task3/usage_model/`.
- Task 4: `model-weight/task4_r5/` and `models/task4_teacher_gallery/`.
- Product photos: `data/train/images_train/<id>.jpg`.
- Saved split: `data/processed/splits.csv`.

The demo uses the fixed 26,217-product teacher gallery from development folds
0, 2, 3 and 4. It does not add holdout or test images. Gallery photos are served
by validated product ID; their bytes must match the saved hash. Existing photo
paths in saved manifests are preserved. The app resolves photos in the local
folder above without copying the dataset.

## Image flow

Original image → local backend → each model's own preprocessing → model output
→ JSON response → page. The frontend never resizes the bytes sent for inference.
Tasks 1–3 use the original image. Task 4 waits for a crop or whole-image choice.

| Predictor | Width × height | Saved settings |
| --- | --- | --- |
| Article type | 60 × 80 | Task 1 bundle |
| Season | 60 × 80 | Task 2 verified bundle and calibration |
| Gender | 60 × 80 | Gender package normalization |
| Usage | 60 × 80 | Usage package normalization |
| Visual search | 240 × 320 | R5 teacher normalization |

Each branch uses RGB, EXIF orientation, aspect-preserving LANCZOS resizing,
letterbox padding, and its own saved normalization. Padding becomes neutral
zero after normalization. No random training augmentation runs. Advanced mode's
small original-image preview is a display preview, not the model tensor.

## Try the flow

Choose a sample, upload JPG/PNG/WebP, or drag a photo into the image frame.
Uploads must be at most 10 MiB and decode to at most 25 million pixels. The server
checks bytes, type and decoding independently of the browser.

Four model labels appear, followed automatically by 5 or 10 similar products.
Standard/Advanced, colours and layouts preserve the current image and cached
results. Advanced shows actual prediction scores, exact model input sizes,
measured server times, and saved evaluation summaries separately.

Article type shows its top five scores; Season shows all four; Gender and Usage
show all their classes. Scores are not accuracy. Cosine similarity is `1 - distance`,
not a probability. Known catalogue self-matches and duplicate/family matches are
excluded before ranking. Frozen evaluation scores can use a different gallery:
Task 4's saved holdout score uses 32,773 development products, disclosed in the UI.

Errors show a retry button or identify the failed predictor. The page never
falls back to fixture labels or a sample gallery. The sample photos are bundled
DummyJSON catalogue assets, recorded in `public/samples/sources.json`; they are
query examples, not the assignment gallery or a source of ground-truth labels.

## Task 4 crop

Upload or select an image, then use **Frame your item** in the Task 4 panel.
Keep the whole item with about 15% space around it. Drag the 3:4 box, adjust
**Crop size**, or open **Fine tune** for exact left/top/width values. Arrow keys
move the focused box; Shift makes larger moves.

Choose **Search this crop**, or **Use whole image**. **Adjust crop** lets you try
again. Changing the image resets the choice. This is a manual crop; the initial
centered box does not detect an item or a person.

Only coordinates are sent for the crop. The server applies them to the oriented
original image, then uses the saved R5 RGB/LANCZOS/normalization settings at
240 × 320. Other predictions use the original image. Cropping can help a query,
but it does not guarantee that the matches are the right product type.

## Local uploads and caching

Uploads are retained on this computer in `be/tmp/demo-api/uploads/<sha256>`.
Retained verified search snapshots live under `core/tmp/demo-api/snapshots/`.
The server never automatically deletes these files. Clear removes the current
browser selection; it does not erase the server copy.

Analysis cache keys use the original image's SHA-256 and backend model version.
Search keys also include gallery/model version, requested count and crop coordinates. Requests use
AbortSignal; old image/decode results cannot replace a later selection. Browser
results stay fresh for five minutes. Server result caches are bounded in memory.
Repeated cached timings refer to the original measured run and are marked as such.
No query triggers training or a new evaluation run.

## Checks and main files

```sh
# From fe/
npm test
npm run typecheck
npm run build

# From the repository root (install the dev extra for pytest)
./.venv/bin/python -m pytest be/tests -q
```

Client tests require a Node version with TypeScript stripping (Node 22.18+).
Builds preserve existing output files (`emptyOutDir: false`). API tests use
retained local fixtures; real-model tests require the packages above.

- `src/App.tsx`: upload, classification cards, crop/search choice and dialogs.
- `src/api/client.ts`: typed local API calls and image identity.
- `src/hooks/useAnalysis.ts`: image preparation state and model requests.
- `src/hooks/useSimilarItems.ts`: versioned retrieval query, including crop coordinates.
- `src/components/CropEditor.tsx` and `src/crop.ts`: adjustable 3:4 crop and pixel geometry.
- `src/components/ModelDetails.tsx`: model settings, timings and saved evaluation.
- `../be/src/fashion_api/`: backend validation, model adapters and API routes.
- `src/api/mockSearch.ts`: retained old fixture; no longer imported by the app.

## Visual styles

Use **Style** in the top bar to change the look without resetting the image,
results, or Standard/Advanced mode. The choice is saved in local storage.
The URL also records the selected style so a look can be shared directly.

| Style | Look | URL value |
| --- | --- | --- |
| Outfit | Warm beige, bold red type, graphic product surfaces | `?style=outfit` |
| Liquid Glass | Frosted surfaces, rounded floating controls, cool light | `?style=glass` |
| Brutalist | Heavy black lines, hard shadows, electric yellow | `?style=brutalist` |
| Nocturne | Dark gallery, lit product backgrounds, champagne details | `?style=nocturne` |
| Field Notes | Sage, paper texture, notebook tabs, stitched edges | `?style=field` |
| Minimal | Neutral white and grey surfaces, charcoal type, quiet borders | `?style=minimal` |
| Original | The original ivory and terracotta design | `?style=original` |

Outfit is inspired by the colours and graphic tone of
[OUTFIT by ++hellohello](https://outfit.hellohello.is/). Liquid Glass is a CSS
interpretation inspired by [Apple's design guidance](https://developer.apple.com/documentation/technologyoverviews/liquid-glass),
not the native Apple material. No assets or proprietary fonts were copied from
those sites. Nocturne and Field Notes are original directions for this demo.

`src/styles.css` keeps the shared layout in the `foundation` CSS layer.
`src/themes.css` supplies the six visual skins and the compact picker styling.
`src/components/StylePicker.tsx` validates and remembers the selection. Themes
use existing local fonts and CSS textures. The glass style supports reduced
transparency; all styles inherit reduced-motion behaviour.

## Layout experiments

Use **Layout** beside Standard/Advanced to compare **Current** with **Studio**.
Studio is the default for a first visit. The choice is saved separately from the
visual style; both layouts work with all seven styles. Switching keeps the current
image, results, and query cache. Direct links accept `?layout=current` or
`?layout=studio`, and can combine both choices: `?layout=studio&style=glass`.

Studio uses a narrow image rail and a wider results area. Advanced information
moves to a scrollable inspector at widths of 1280px and above. Below that it
stays below the results. On screens up to 800px, the image becomes a compact
horizontal summary above the demo. Current keeps the earlier two-column layout.
Both layouts fold image choices after a sample or upload is selected; **Change
image** opens them again.

`src/layouts.css` holds these layout rules, after the theme styles.
`src/components/LayoutPicker.tsx` manages the layout preference without remounting
the demo. Predictions and matches come from the local model server.

## Shared Select component

Style, Layout, and result count use the local `src/components/ui/select.tsx`
component, manually adapted from [shadcn/ui's Radix Select](https://ui.shadcn.com/docs/components/radix/select).
It uses `@radix-ui/react-select` for keyboard navigation, focus handling, selected
items, and portal positioning. Its styling is in `select.css`, using the existing
seven theme palettes instead of Tailwind utilities. The upstream MIT notice is
included beside the component in `LICENSE.md`.

Menus support arrow keys, typing to find an option, Enter to select, and Escape
to close. Opening a menu does not clear the current demo. The style menu also
shows a short description of each look.

The inline gallery is styled in `src/similar-banner.css`. Item details and Task 4
are visible together in both layouts. Task 4 starts after a crop or whole-image choice. The banner keeps the 5/10 result picker, product previews, loading/error
states, and Advanced similarity scores and search settings.

## Light and dark appearance

Use the sun/moon button beside Style. All seven styles have light and dark
palettes; Minimal stays neutral in both. On a first visit the app follows the
device colour setting and tracks changes. Choosing a mode saves an explicit
override in `thread-color-mode`. Style and layout switches preserve that choice,
the uploaded image, and cached results. URLs accept `?appearance=light` or
`?appearance=dark`, together with the style and layout parameters.

`src/components/ColorModeToggle.tsx` owns the preference and browser theme colour.
`src/color-modes.css` supplies the palettes and surface overrides. Product photos
keep their original colours.

Each style has its own dark palette and surface tokens. The current
`src/color-modes.css` is the source of truth for those colours.
