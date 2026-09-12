# Oracle deployment

Source branch: `feat/demo-model-api`. No merge into `main` is needed.

The deployment uses trial credits: `VM.Standard.E5.Flex`, 1 OCPU (2 threads),
4 GB RAM, Ubuntu 24.04 x86, and a 45 GB filesystem. A1 and E4 were unavailable.
The trial expires 11 October 2026. No paid account upgrade was authorized.

## Release files

From this branch checkout, fetch the branch and build two archives:

```sh
git fetch origin feat/demo-model-api
./.venv/bin/python deploy/oracle/prepare.py \
  --artifact-root /absolute/path/to/original/core \
  --output /absolute/path/to/new-release
```

The script requires HEAD to match the fetched branch tip. It includes current
local deployment changes and records each code file's hash in `release.json`.
It reads the original weights/photos without changing them, checks every gallery
photo hash, and excludes all other training images. `code.tar.gz` contains the
website, API, inference source and small verified support files.
`artifacts.tar.gz` contains only model packages and the fixed gallery photos.
Do not put either archive or an SSH private key in Git.

Upload over SSH. Verify `SHA256SUMS` before extracting into new, separate release
and artifact directories. Keep the existing release until the new one passes.
Do not extract over user files or a running release. Future code-only releases
reuse the same fixed artifact directory.
Pass `--code-only` to prepare later code releases without rebuilding the artifact archive.

Current server: `ubuntu@140.245.124.14`, SSH key
`/home/dinhquan/.ssh/fashion-demo-oracle-20260912`.
Live code: `/home/ubuntu/fashion-demo/release-web`.
Fixed artifacts: `/home/ubuntu/fashion-demo/artifacts`.
Site: https://fashion-demo.140.245.124.14.sslip.io
The public IP is ephemeral; a replacement requires updating the hostname and Caddyfile.

## Start on the server

Install `docker.io`, `docker-compose-v2` and `caddy` from Ubuntu's package repository.
Build on the x86 server.
From the extracted code directory:

```sh
export FASHION_ARTIFACT_DIR=/home/ubuntu/fashion-demo/artifacts
sudo --preserve-env=FASHION_ARTIFACT_DIR docker compose -f deploy/oracle/compose.yaml up -d --build
curl --fail http://127.0.0.1:8000/readyz
```

The artifact directory must contain `core/models`, `core/model-weight` and
`core/data/train/images_train`. All artifact files must be readable by UID 10001.
The app listens only on the server's loopback address. Add an HTTPS reverse proxy
before sharing it: the browser image hashing uses `crypto.subtle`, which needs
HTTPS outside localhost. Route the whole hostname to `127.0.0.1:8000`.

Allow public TCP ports 80 and 443 for the HTTPS proxy. Limit SSH port 22 to the
administrator's current public IP. Keep port 8000 private. Add a public IPv4
address if the creation wizard leaves it disabled. Keep encryption enabled.

## Checks and limits

`/readyz` returns 503 until every model is ready. Test `/`, direct `/demo`, an
upload, a crop search and a gallery photo over HTTPS. Restart once and repeat.

One process uses two CPU threads. The container has a 3 GiB RAM limit and a
bounded request count. Uploads and snapshots each have a 256 MiB temporary disk
limit and disappear on container restart. Old image IDs then need a new upload.
If the upload disk fills, restart the app container to clear it; automatic age
based upload expiry is not implemented. Log rotation limits local log growth.

The live deployment is x86; ARM compatibility was not tested.
Restart with `sudo docker restart oracle-app-1`. Logs:
`sudo docker logs --tail 100 oracle-app-1` and `sudo journalctl -u caddy -n 50`.
