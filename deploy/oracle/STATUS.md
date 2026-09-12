# Oracle deployment — 12 September 2026

Live website: https://fashion-demo.140.245.124.14.sslip.io

Instance `fashion-demo`: `VM.Standard.E5.Flex`, 1 OCPU (2 threads), 4 GB RAM,
Ubuntu 24.04 x86, Singapore AD-1. Public IP `140.245.124.14`; private IP `10.0.0.157`.
Instance OCID:
`ocid1.instance.oc1.ap-singapore-1.anzwsljrgfvbyxqc5uif3rby5nlzwx4yqjki52zbfprm6vtx6e4oljbjzorq`.

The user approved trial credits, SSH key creation, public IP, public TCP 80/443,
and SSH restricted to `14.169.90.157/32`. These are applied. Port 8000 binds only
to loopback. Caddy serves a valid public HTTPS certificate and redirects HTTP.
The account remains Free Trial; no paid upgrade was made. Console shows a
S$400 allowance ending 11 October 2026, with usage not yet reported.
No spending beyond trial credits is authorized. A1 and E4 lacked capacity;
only this E5 instance was created successfully.

Code is from `feat/demo-model-api`, base commit
`9be8dda369e97cb197565b9f64856790aa96780b`, plus the hosting changes in this PR.
Deployment uses uploaded files, without automatic GitHub updates.
The original checkout was read only; no merge into main was made.

Deployed code archive SHA256:
`fbc62fae448c7b6ba3160ba51fc823a349e0366c41e14a573367500f6c726565`.
Fixed artifact archive SHA256:
`1b0ddb2a6cac47671e95c8ad61143bcbf52a0c13ca86eaa625c72c43c1477eaa`.
Both hashes were checked on the server before extraction.
Code: `/home/ubuntu/fashion-demo/release-web`.
Artifacts: `/home/ubuntu/fashion-demo/artifacts` (all 26,217 gallery hashes checked).
Container: `oracle-app-1`; Docker Compose and Caddy start automatically.
SSH key stays local: `/home/dinhquan/.ssh/fashion-demo-oracle-20260912`.

Validation: 34 backend tests passed, Ruff and whitespace checks passed, production
frontend and full x86 container built on Oracle. HTTPS smoke checks passed:
all five models ready, `/` and direct `/demo`, upload, four predictions, full
image search, crop search and gallery image delivery. Browser upload and crop
search worked; landing and search screenshots were inspected.
The container restart passed the same complete HTTPS smoke checks, and browser
reload showed live predictions. Docker reports the restarted container healthy.
Smoke evidence: `/home/dinhquan/.codex/fashion-oracle-release-web-20260912/`.

The container has a 3 GiB memory limit. Uploads and snapshots each use a 256 MiB
temporary filesystem, cleared on restart. Photos are sent to the server; frontend
wording was corrected to say so. No automatic upload expiry is implemented.
The free sslip.io hostname depends on the assigned ephemeral public IP.

Gallery fix: the initial request cap of 8 rejected parallel thumbnail requests
with HTTP 503. The Compose runtime override and Dockerfile now allow 128 concurrent
connections/tasks. This keeps the same single model worker and 3 GiB memory cap.
The deployed Compose/Dockerfile were updated after the archive listed above.
Validation: all 40 simultaneous HTTPS requests for the affected 10 photos passed
with HTTP 200 and valid JPEG bytes. Runtime command was verified at 128, with no
concurrency warnings during the test.
