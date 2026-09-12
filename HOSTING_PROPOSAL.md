# Fashion demo hosting proposal

Historical options review, written before deployment. Oracle trial hosting was
selected and deployed; see [current setup](deploy/oracle/README.md) and
[deployment evidence](deploy/oracle/STATUS.md). The options and unfinished work
below describe the earlier proposal, not the live server.

Reviewed 12 September 2026. Use `feat/demo-model-api` only. Fetched and checked commit `9be8dda369e97cb197565b9f64856790aa96780b` from [PR 20](https://github.com/TrnLin/MLA2/pull/20). No merge into `main` is needed.

## Pick: Hugging Face Docker Space

For this class demo, use one Docker Space for the React website and FastAPI backend. It gives one HTTPS address and enough memory without managing a Linux server.

- **Expected cost: $9/month** for a personal PRO account, plus $0/hour for CPU Basic. A new Docker Space now needs a paid account. If you already have PRO, there is no extra CPU Basic charge. [Creation rules](https://huggingface.co/docs/hub/en/spaces-overview), [PRO price](https://huggingface.co/pricing).
- CPU Basic supplies **2 vCPU, 16 GB RAM, 50 GB temporary disk**. It sleeps after 48 hours without use; a visitor wakes it. Open the demo before presenting. Local startup time below excludes platform wake-up, build and download time. [Hardware and sleep rules](https://huggingface.co/docs/hub/en/spaces-gpus).
- Use the standard Space address. No separate frontend host, database or GPU is needed. Set Docker SDK and `app_port: 7860`; run as UID 1000 with writable runtime folders. [Docker setup](https://huggingface.co/docs/hub/en/spaces-sdks-docker).
- Store the fixed model/photo bundle in a private artifact repository and fetch a pinned revision with a read-only token saved as a Space secret. Keep code updates separate from that bundle. Temporary disk can disappear, so rebuilds must restore the bundle. Do not rely on retained visitor uploads.
- Prepare the Space source from an explicit checkout of `feat/demo-model-api`; record its commit. Any later sync must name that branch, never default to `main`. Do not upload the whole project history or training data.

## Other option: DigitalOcean server

Choose this if the site must stay awake and you want full control. A Basic Regular Droplet with **2 vCPU, 4 GiB RAM, 80 GiB SSD and 4,000 GiB monthly transfer costs $24/month**. This excludes tax, optional backups, a domain and excess transfer. [Official price](https://www.digitalocean.com/pricing/droplets).

Run the same app with one Uvicorn worker. Put Caddy or Nginx in front for HTTPS, run the app under a service manager, and keep artifacts on disk. Upload the fixed bundle once by SSH/SFTP. Deploy code from `feat/demo-model-api` separately. This avoids Space sleep, but you must manage updates, HTTPS and disk cleanup yourself.

## What was measured

The latest branch code was tested with the original checkout's local weights and gallery photos. The original checkout was read only. A separate scratch tree held writable snapshots, lock files and test uploads; gallery images were linked, not copied.

- All **five models loaded without errors**. All **33 backend API, crop and real-model tests passed**. One existing FastAPI test-client deprecation warning remains.
- Startup including imports: **3.61 seconds**. Loaded process RAM: **636 MiB**. Peak process RAM: **674 MiB**.
- Three uncached runs of one small catalogue image: four labels together **27–42 ms**; search **73–100 ms**. This was an Intel Core Ultra 5 225H laptop, CPU-only Torch 2.13.0, Python 3.14.7 and two Torch threads. These are not cloud latency or load-test results.
- Fixed search gallery: **26,217 photos, 327,181,830 bytes (312 MiB)**. Every file exists and every photo hash matches the saved gallery.
- Five weight files: **67,127,605 bytes (64 MiB)**. Search features alone: **13,423,232 bytes (12.8 MiB)**.
- Observed startup artifact reads total about **99 MiB**, excluding source code. With gallery photos, the fixed data is about **411 MiB uncompressed**, before a few small package README files. Frontend public assets occupy about **3.5 MiB on disk**.
- Each startup currently retains another **76,731,712 bytes (73.2 MiB)** of search snapshots. Uploaded images can each use 10 MiB and are never deleted automatically.

**Sizing estimate:** start with 2 vCPU and 4 GiB RAM for one worker. Budget 5–10 GiB for a CPU-only environment/container, files and working space. This leaves room beyond the small-image test for large uploads and temporary copies. The code allows 25-million-pixel images. No maximum-size or concurrent-user memory test was run. Requests share a lock and wait their turn; adding workers duplicates models, caches and snapshots.

## Files to upload

Keep the current paths under the chosen `FASHION_PROJECT_ROOT`. The variable points to the **core folder**, not the repository root.

From the original checkout at `/home/dinhquan/personal/academic/RMIT/Machine-Learning/MLA2-eda`:

- `core/models/task1_article_type.pt`
- `core/models/task2_season.pt`
- `core/model-weight/task3/gender_model/`
- `core/model-weight/task3/usage_model/`
- `core/model-weight/task4_r5/`
- Only photos whose IDs occur in the branch's `core/models/task4_teacher_gallery/metadata.csv`, placed at `core/data/train/images_train/<id>.jpg`.

The matching weights also exist in `MLA2 weight/`, but that folder needs path mapping. The checked `core/` paths already work. Gender must remain the selected 30-epoch MixUp package: `t3_gender_name_truth_mixup_alpha020_refit_20260911T041436Z_3af0b94e`.

From the branch, include `core/src/`, both package definitions and the model manifests, canonical `data/processed/splits.csv`, `label_maps.json`, and the complete `models/task4_teacher_gallery/` directory. Also retain the small config/report/evidence files listed in `HOSTING_RUNTIME_FILES.txt`; loaders verify these before using the weights. Removing them to save space can break inference. Include the built `fe/dist/` files.

Do not regenerate splits or the gallery. No external Task 3 training images, holdout/test image collection, training checkpoints, notebook outputs or entire data ZIP is needed. The saved split rows and saved evaluation evidence still belong in the bundle.

## Changes needed before hosting

1. Add a Docker build with a Node build stage and Python CPU runtime. Use a narrow `.dockerignore`/file list. Existing dependencies match the working local environment, but a clean cloud installation and frontend build still need testing. `core` currently pulls in Jupyter and other training packages; moving those into an optional training extra would reduce the image, but is not required for the first deployment.
2. Serve `fe/dist/` through FastAPI after the API routes, with `/` and `/demo` returning `index.html`. Keep unknown `/api/*` requests as API errors. Or use a reverse proxy with the same routing. Vite's proxy only exists in dev/preview; it is not inside the built website. The client already uses same-origin `/api` URLs, so this needs no API URL variable or CORS change.
3. Add startup bundle download/hash verification for Spaces, or mount the local bundle on the server. Preserve the source files checked by Task 2 provenance. Keep the snapshot folder writable; registry readers also create `.lock` files beside their CSVs.
4. Add upload expiry and a disk cap. Reuse one verified snapshot per artifact version, or remove old snapshots only after their owning process exits. Add an upload request limit/queue for a public demo. Keep a single worker.
5. Make a readiness check return HTTP 503 if any model is missing. Today `/api/health` can return HTTP 200 with `status: degraded`; a host that checks only the status code could call a broken demo healthy.

## Build and start recipe

These are proposed deployment commands, not commands already run on a host. First add the static-file routing above. On either option, prepare the code explicitly:

```sh
git clone --single-branch --branch feat/demo-model-api https://github.com/TrnLin/MLA2.git /app
cd /app
git rev-parse HEAD
python3.14 -m venv .venv
./.venv/bin/python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
./.venv/bin/python -m pip install -e ./core -e ./be
cd /app/fe
npm ci
npm run build
```

Use Node 22.18 or later in the build stage. Python requirements allow 3.12–3.14; 3.14 matches the local test. Keep the selected branch commit fixed for each release.

After placing the fixed bundle at the paths above:

```sh
cd /app
export FASHION_PROJECT_ROOT=/app/core
export FASHION_UPLOAD_DIR=/app/runtime/uploads
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
./.venv/bin/python -m uvicorn fashion_api.api:app --host 0.0.0.0 --port 7860 --workers 1
```

Create `/app/runtime/uploads` and make it writable by the app user. `/app/core/tmp/demo-api/snapshots` and the registry CSV parent folders must also allow writes. On DigitalOcean with a same-machine reverse proxy, bind Uvicorn to `127.0.0.1` instead.

`FASHION_PROJECT_ROOT` and `FASHION_UPLOAD_DIR` already exist. `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are library settings; the code also sets Torch threads to two. For the proposed private artifact downloader, add `HF_TOKEN` as a secret and new settings for artifact repository ID and immutable revision. Those downloader settings do not exist yet. Do not put a token into frontend variables or the Docker image.

Before sharing the address, check all five model statuses, an upload, a crop search, gallery images, and direct `/demo` refresh. Also test a restart, expired-upload handling and the built page in a browser. Frontend build/visual checks and cloud load tests were not run in this review. Nothing was deployed, purchased, committed, pushed or merged.
