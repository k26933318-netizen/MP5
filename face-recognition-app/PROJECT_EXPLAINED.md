# FaceVerify: Project Explained

This guide describes the code and files currently in this repository. It inventories the local `.env` settings, but withholds secret values. It inspected the SQLite database read-only for its schema, selected non-secret metadata, and aggregate row counts; it does not reproduce session IDs, subject IDs, tokens, IP bindings, audit details, or embedding vectors.

## 1. Project overview

### Five-line explanation

1. FaceVerify is a Python service and browser demo for checking a person's face.
2. A person can enroll several camera frames under an opaque ID, with consent.
3. Later, the service can compare a live face to that ID (1:1 verification) or search all active profiles (1:N identification).
4. It checks image quality and liveness, then compares numerical face embeddings rather than keeping the submitted photographs.
5. A consuming application decides what to do with the result; FaceVerify does not make election, eligibility, or other business decisions.

### Analogy first

Think of the service as a secure reception desk. Enrollment is like making a reference card for an ID; verification is checking a visitor against the card they claim; identification is searching the filing cabinet when they do not give an ID. The camera frames are temporary evidence, while the stored reference is a compact mathematical fingerprint. A second guard (liveness checks) asks the visitor to blink or turn their head to make a static photograph harder to use.

### Technical detail

The application accepts JPEG frames from a browser, validates and decodes them with OpenCV, detects faces with InsightFace, and creates normalized 512-dimensional embeddings. It applies quality gates, optional passive MiniFASNet anti-spoof scoring, and a randomized blink/head-turn challenge. SQLite stores subject metadata, embeddings, sessions, challenges, one-use token state, and audit events. FAISS performs exact cosine-style similarity search using an in-memory inner-product index.

The main supported app path is the browser demo mounted with the API in one Waitress process:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Set secrets and the two local MiniFASNet model paths in .env.
python -m demo.demo_backend
```

Open `http://127.0.0.1:5050`. For the API-only development server, use `python app.py`; for the API-only Waitress server/container, use `python serve.py`. The README recommends CPython 3.11 or 3.12 on Windows.

### Stack

| Area | Technology | Job |
|---|---|---|
| Languages | Python, JavaScript, HTML, CSS, YAML, PowerShell | API/business logic and tools; camera/UI; API contract; Windows launcher |
| HTTP application | Flask, Waitress | JSON APIs and pages; production-style single-process WSGI server |
| Face analysis | InsightFace, ONNX Runtime | Detection, landmarks, recognition embeddings, local MiniFASNet inference |
| Image/science | OpenCV, NumPy | Decode/resize/crop images, image quality and geometry, vector calculations |
| Similarity search | FAISS CPU | Exact inner-product search over normalized vectors |
| Persistent storage | SQLite | Subject and workflow records; WAL journaling |
| Biometric encryption | Fernet from `cryptography` | Optional encryption of embedding blobs |
| Tokens | PyJWT | Signed HS256 verification tokens with database-backed one-use state |
| Tests | pytest | Unit and Flask-client tests |
| External service | None configured | The demo calls its own local FaceVerify API; no hosted recognition/cloud API is used |

### Architecture diagram

```text
Person
  |
  | camera permission + live frames (base64 JPEG)
  v
Browser UI (demo HTML / JavaScript / FaceCapture)
  |                                  |
  | session/result proxy             | X-Session-Id: challenge, preview, submit
  v                                  v
demo_backend.py (Flask pages +       Face API (Flask, /service/api/v1)
server-side ServiceClient)             |
  | X-API-Key                         +--> FaceService
  +---------------------------------->|      +--> FaceEngine / InsightFace
                                       |      +--> quality + liveness checks
                                       |      +--> MiniFASNet ONNX models
                                       +--> FaceStore (SQLite)
                                              +--> embeddings -> in-memory FAISS
                                              +--> sessions/challenges/tokens/audit
  |
  +--> consuming backend makes its own business decision
```

The camera sends images to the service process, not to a third-party API. The server-side browser gateway keeps the service API key out of JavaScript; the browser uses only a session ID for camera endpoints.

## 2. File-by-file explanation

### Analogy first

The repository is like a small clinic: `demo/` is the reception area, `face_verification/` is the examination room and records desk, `client/` is the camera operator, `models/` contains the locally used examination tools, and `tools/` contains calibration and migration utilities. `legacy/` is an older clinic layout retained separately; it is not the primary workflow described by the README.

### Technical detail

The following lists project-owned files, excluding `.venv`, Python caches, and test caches. “Talks to” means a direct code/data dependency, not every indirect Python import.

| File or folder | Purpose / entry-point status | Talks to |
|---|---|---|
| `README.md` | Main usage, architecture, model licensing, API overview, calibration, and deployment notes | `requirements.txt`, `demo.demo_backend`, `openapi.yaml`, `tools/` |
| `app.py` | **Entry point:** Flask development server for the API only | `face_verification.api`, `face_verification.config` |
| `serve.py` | **Entry point:** Waitress API-only server; also the Docker command | `face_verification.api`, `face_verification.config` |
| `run-demo.ps1` | Windows convenience launcher for the demo | `demo.demo_backend` |
| `sitecustomize.py` | Python startup hook; reads repo `.env` and limits BLAS thread counts | `.env`, Python process environment |
| `.env` | Local runtime configuration; see the redacted settings inventory in section 3 | `face_verification.config`, `demo.demo_backend` |
| `.env.example` | Template for runtime settings; contains placeholders, not production secrets | `face_verification.config`, `demo.demo_backend` |
| `.gitignore` | Excludes local database, models/data directories, `.env`, and Python artifacts from Git | Git |
| `.dockerignore` | Excludes `.env`, local face data/database, models, cache, and virtual environments from Docker context | Docker build |
| `requirements.txt` | Pinned runtime Python package list | Python package installer |
| `requirements-dev.txt` | Development dependency include: runtime requirements plus pytest | `requirements.txt`, pytest |
| `Dockerfile` | Builds an API-only Python container; starts `serve.py`, listens on port 5000, uses a non-root user | `requirements.txt`, `serve.py`, Docker runtime mounts |
| `openapi.yaml` | Human/machine-readable contract for the current `/api/v1` service endpoints | `face_verification/api.py` |
| `client/face-capture.js` | Browser camera controller: preview loop, challenge capture, frame submission, session header | Browser camera APIs; `/api/v1/session/*` |
| `client/python_client.py` | Small authenticated Python HTTP client for session, subject, and token operations | `urllib` standard library; `/api/v1/*` |
| `demo/demo_backend.py` | **Main demo entry point:** serves UI pages, proxy endpoints, and mounts the face API under `/service` | `demo/*.html`, `demo/static/*`, `client/python_client.py`, `face_verification.api` |
| `demo/home.html` | Demo landing page | `demo_backend.py`, `demo/static/app.css` |
| `demo/enroll.html` | Enrollment form, subject ID, consent checkbox, camera UI | `demo/static/app.js`, `demo/static/app.css` |
| `demo/verify.html` | Browser UI for an ID-free identify flow (despite the human-facing “verify” wording) | `demo/static/app.js`, `demo/static/app.css` |
| `demo/static/app.js` | UI controller; creates session, starts camera, displays decisions, proxies 1:1 token confirmation | `client/face-capture.js`, `/service-api/*` |
| `demo/static/app.css` | Styling for the current demo UI | `demo/*.html` |
| `face_verification/__init__.py` | Package marker | Python package loader |
| `face_verification/api.py` | API factory, authentication/session guards, rate limiter, route handlers, response headers/errors | `service.py`, `store.py`, `engine.py`, `tokens.py`, `liveness.py`, `audit.py` |
| `face_verification/audit.py` | Filters sensitive detail keys and forwards audit events to storage | `store.py` |
| `face_verification/config.py` | Typed settings; parses `.env`/environment and selects defaults | `.env`, all service modules |
| `face_verification/engine.py` | Lazy InsightFace wrapper, image validation/decoding, detector readiness checks | InsightFace, OpenCV, NumPy, `config.py` |
| `face_verification/quality.py` | Face count/size/confidence, blur, lighting, pose, and optional eye-open gates | OpenCV, NumPy, `liveness.py`, `config.py` |
| `face_verification/liveness.py` | Local ONNX anti-spoof model pair, blink/turn challenge, challenge persistence, IP binding | ONNX Runtime, OpenCV, NumPy, `store.py`, local `.onnx` files |
| `face_verification/service.py` | Main enrollment, 1:1 verification, 1:N identification, quality/liveness/matching decisions | `engine.py`, `quality.py`, `liveness.py`, `store.py`, `audit.py` |
| `face_verification/store.py` | SQLite schema and transactions, optional embedding encryption, in-memory FAISS search | SQLite, NumPy, FAISS, Fernet, `config.py` |
| `face_verification/tokens.py` | Signed verification JWT issue/introspection and one-use consumption | PyJWT, `store.py`, `config.py` |
| `models/minifasnet_a.onnx`, `models/minifasnet_b.onnx` | Runtime passive anti-spoof models; loaded locally, not sent to a hosted service | `face_verification/liveness.py`, ONNX Runtime |
| `models/minifasnet_a_source.pth`, `models/minifasnet_b_source.pth` | Source PyTorch weights retained with the model files; the runtime service uses ONNX | `tools/export_minifasnet_onnx.py` |
| `models/.gitkeep` | Keeps the models directory present when empty | Git |
| `face_data/face.db` | Current service's SQLite database, present in the workspace | `FaceStore` |
| `face_data/face.db-wal`, `face_data/face.db-shm` | SQLite write-ahead log and shared-memory sidecar files present in this workspace; SQLite manages them | SQLite WAL connection in `FaceStore` |
| `database/faces.index` | Persisted FAISS index belonging to the legacy app, not the current SQLite-backed app | `legacy/vector_store.py`, `tools/import_legacy.py` |
| `database/names.json` | Legacy index-position-to-name mapping; contains identifying labels and is not used by the current service | `legacy/vector_store.py`, `tools/import_legacy.py` |
| `database/faiss_index/.gitkeep` | Empty placeholder directory; no current code reference was found | Git |
| `legacy/app.py` | **Legacy entry point:** older, unauthenticated name-based Flask app with direct register/verify routes | `legacy/face_utils.py`, `legacy/vector_store.py`, `legacy/templates/*` |
| `legacy/face_utils.py` | Legacy InsightFace wrapper, image quality checks, basic two-frame movement check | InsightFace, OpenCV, NumPy |
| `legacy/vector_store.py` | Legacy persistent FAISS index plus plaintext names JSON | FAISS, NumPy, `database/faces.index`, `database/names.json` |
| `legacy/__init__.py` | Legacy package marker | Python package loader |
| `legacy/templates/index.html`, `register.html`, `verify.html` | Old UI pages; distinct from the current `demo/` UI | `legacy/app.py`, `legacy/static/*` |
| `legacy/static/js/camera.js` | Old browser camera loop; posts images to legacy `/detect_face`, `/register`, and `/verify` | Legacy routes |
| `legacy/static/styles.css` | Styling for old UI | `legacy/templates/*` |
| `tools/calibrate_from_db.py` | Computes genuine/impostor similarity distributions and writes a suggested `threshold.json` | `FaceStore`, `face_data/face.db`, NumPy |
| `tools/tune_threshold.py` | Evaluates face-match rates from an on-disk labeled `dataset/<person>/*.jpg` | `FaceEngine`, OpenCV, NumPy |
| `tools/antispoof_eval.py` | Evaluates anti-spoof thresholds on `attacks/<type>/*.jpg`; writes JSON and PNG report | `FaceEngine`, `AntiSpoofEnsemble`, OpenCV, NumPy |
| `tools/export_minifasnet_onnx.py` | Developer utility to export local upstream MiniFASNet PyTorch weights to ONNX | Local upstream clone, PyTorch, ONNX |
| `tools/import_legacy.py` | Explicit, test-data-only import of legacy vectors into a new SQLite store | Legacy `faces.index`/`names.json`, FAISS, `FaceStore` |
| `tests/test_api.py` | API endpoint/auth/session behavior tests | Flask test client, API/service doubles |
| `tests/test_calibration.py` | Calibration math tests | `tools/calibrate_from_db.py` |
| `tests/test_engine.py` | Image decoding and engine behavior tests | `FaceEngine` |
| `tests/test_liveness.py` | Anti-spoof/challenge/movement tests | `liveness.py` |
| `tests/test_quality.py` | Frame quality gate tests | `quality.py` |
| `tests/test_service.py` | Enrollment, verification, identify business-flow tests | `FaceService` |
| `tests/test_store.py` | SQLite schema and persistence tests | `FaceStore` |
| `tests/test_vector_store.py` | Legacy vector index tests | `legacy/vector_store.py` |
| `tests/test_web_app.py` | Demo page/gateway tests | `demo_backend.py` |

## 3. What the project does with files and data

### Analogy first

The SQLite database is the records cabinet. A face image is like a visitor being viewed briefly at the desk: it is decoded and analyzed in memory, but is not filed away. The embedding is the compact reference card. The legacy app has a separate old cabinet (`database/faces.index` plus `names.json`) that the new service does not automatically read.

### Technical detail: runtime and utility file I/O

| Path / data | Function or component | Read/write | When and format |
|---|---|---|---|
| `.env` | `config._load_local_dotenv()`; also the demo launcher module | Read | Startup; simple `KEY=value` lines fill environment values only when not already set |
| Environment variables | `config.load_settings()` | Read | Startup/configuration; strings converted to typed settings |
| `models/minifasnet_a.onnx`, `models/minifasnet_b.onnx` | `AntiSpoofEnsemble.load()` | Read | Loaded by ONNX Runtime locally when required; 80x80 MiniFASNet models |
| InsightFace model cache | InsightFace internals via `FaceEngine.analysis` | May read/download | First inference if model cache is absent; model package can be fetched by InsightFace. This is the only external model-download behavior documented by this project; no per-frame hosted API call |
| `face_data/face.db` | `FaceStore.__init__`, `_initialize_schema`, `FaceStore` CRUD methods | Read/write | Current app startup and requests; SQLite relational data, with embeddings as float32 BLOBs and JSON text for flexible metadata/results |
| `face_data/face.db-wal`, `face_data/face.db-shm` | SQLite WAL mode | Read/write/create/delete by SQLite | While the database is open or has WAL changes; implementation/runtime sidecars, not app-level documents |
| Camera frame payloads | `/session/preview`, `/session/submit`, `decode_image`, `FaceService` | Read/temporary memory only | Browser sends JPEG data URL/base64 as request JSON; images are decoded to OpenCV BGR arrays, not inserted into the database |
| Embedding blobs | `FaceStore._encode_vector`, `_decode_vector` | Write/read | Enrollment, verification, identify index rebuild; little-endian float32 vectors, Fernet-encrypted only if `FACE_DB_KEY` is configured |
| `face_data/threshold.json` | `FaceService._load_verify_threshold()` | Read | Service construction if no threshold environment setting overrides it; JSON containing `verify_threshold` |
| `face_data/threshold.json` (or `--output`) | `tools/calibrate_from_db.py` | Write/create | When calibration finds a threshold meeting the target false-accept rate; JSON summary plus selected threshold |
| `database/faces.index` | `legacy.VectorStore.load/save()`; import utility | Read/write | Legacy registration saves a FAISS inner-product index; import utility reads it |
| `database/names.json` | `legacy.VectorStore.load/save()`; import utility | Read/write | Legacy person-name mapping serialized as JSON; this is separate from current opaque subject IDs |
| Temporary files in `database/` | Legacy `VectorStore.save()` | Create, replace, delete temp files | Atomic-save preparation before `os.replace()` installs the index and mapping |
| `dataset/<person>/*.jpg` | `tools/tune_threshold.py` | Read | Optional offline threshold evaluation; classifies folder names as people, extracts embeddings, outputs score/rate statistics to stdout |
| `attacks/<type>/*.jpg` | `tools/antispoof_eval.py` | Read | Optional offline anti-spoof evaluation; type subfolders include `real`, `print`, `screen`, `video`, `mask` |
| `database/antispoof_threshold.json` (default or `--output-json`) | `tools/antispoof_eval.py` | Write/create | Evaluation rates and suggested anti-spoof threshold as JSON |
| `database/antispoof_roc.png` (default or `--output-png`) | `tools/antispoof_eval.py` | Write/create | Locally drawn threshold evaluation chart |
| Local `.pth` weights and upstream source checkout | `tools/export_minifasnet_onnx.py` | Read | Explicit developer command only; loads PyTorch checkpoint from a local path and writes `.onnx` |
| `models/*.onnx` output | `tools/export_minifasnet_onnx.py` | Write/create | Developer export result; not auto-downloaded by this exporter |
| Logs | Python logging / Flask / Waitress | Write to configured process logging stream (normally console) | Per-request status/timing and selected warnings/errors; no application-managed log file was found |

### Actual local `.env` settings (secret values redacted)

The names and non-secret values below were read from this workspace's `.env`. API-key, JWT-secret, and database-key values are intentionally not printed or copied into this guide.

| Variable | Current local value | Purpose |
|---|---|---|
| `FACE_PROFILE` | `default` | Selects normal or max-accuracy tuning profile |
| `FACE_MODEL` | `buffalo_l` | InsightFace recognition model name |
| `FACE_DET_SIZE` | `640` | Face detector input size |
| `FACE_DATA_DIR` | `face_data` | Directory for SQLite database and calibration output |
| `FACE_DB_KEY` | Empty (not configured) | Fernet key for optional embedding encryption |
| `ALLOW_MODEL_MISMATCH` | `false` | Whether to allow a stored embedding model mismatch |
| `FACE_SERVICE_API_KEY` | Non-empty; value withheld (may be a placeholder) | Authenticates trusted backend requests |
| `FACE_TOKEN_SECRET` | Non-empty; value withheld (may be a placeholder) | Signs verification JWTs and derives the keyed client-IP binding |
| `AUTH_DISABLED_FOR_DEV` | `false` | Prevents bypassing API-key authentication |
| `ANTISPOOF_MODEL_A_PATH` | `models/minifasnet_a.onnx` | Local first MiniFASNet model |
| `ANTISPOOF_MODEL_B_PATH` | `models/minifasnet_b.onnx` | Local second MiniFASNet model |
| `REQUIRE_ANTISPOOF` | `true` | Requires passive anti-spoof evaluation |
| `REQUIRE_CHALLENGE` | `true` | Requires a randomized live-action challenge |
| `KIOSK_MODE` | `false` | Kiosk IP allowlist mode is disabled |
| `KIOSK_ALLOWED_IPS` | Empty | Comma-separated kiosk allowlist |
| `FACE_CORS_ORIGINS` | `http://127.0.0.1:5050` | Allowed browser origin for session API responses |
| `HTTPS_TERMINATED` | `false` | Declares HTTPS termination by a trusted proxy |
| `HOST` | `127.0.0.1` | Bind host for the API-only entry points |
| `PORT` | `5000` | API-only listener port |
| `WAITRESS_THREADS` | `4` | Waitress worker-thread count |

This is a local configuration snapshot, not a recommended production template. The API-key and JWT-secret settings are non-empty, but whether they are real high-entropy secrets or example placeholders, and their ownership/rotation history, are **unsure** because their contents were deliberately withheld. `FACE_DB_KEY` is empty, which matches the database encryption marker described below: any embedding BLOBs written here would not be Fernet-encrypted. The current demo's listener is independently configured by `DEMO_HOST`/`DEMO_PORT` in `demo/demo_backend.py`; the `.env` does not set those, so the demo defaults to `127.0.0.1:5050`.

### Current face database snapshot

The database was opened read-only. This is an aggregate snapshot of the local `face_data/face.db`, not a dump of its rows. Zero subject and embedding rows means this particular database currently contains no enrolled face profiles or embedding vectors.

| Table | Rows at inspection | What the rows represent |
|---|---:|---|
| `meta` | 3 | Schema version, recognition model name, embedding encryption marker |
| `subjects` | 0 | Enrolled opaque subject IDs and status |
| `embeddings` | 0 | Face feature vectors; no vectors are currently stored |
| `sessions` | 16 | Short-lived workflow session metadata and results |
| `challenges` | 0 | Randomized, single-use liveness challenges |
| `tokens` | 0 | Issued verification token state |
| `audit_log` | 3 | Append-only event metadata |

Aggregate session purposes: `enroll` 15, `verify` 1. The database schema uses `verify` as the stored purpose for ID-free `identify` sessions too; the identify intent is encoded internally in the session subject sentinel. This count therefore describes the stored schema value, not necessarily the user-facing purpose. No individual session result, identifier, timestamp, audit detail, or challenge/token value is reproduced.

The metadata values relevant to this explanation are `schema_version=1`, `model_name=buffalo_l`, and `embeddings_encrypted=disabled`. The database has no enrolled identities right now, but it does contain historical session and audit records. Those records were not individually inspected or included.

### Database tables and columns

`FaceStore._initialize_schema()` creates tables if missing. `PRAGMA journal_mode=WAL` enables SQLite write-ahead logging; `_write_transaction()` serializes writes with a lock, starts `BEGIN IMMEDIATE`, retries a locked database, then commits or rolls back.

| Table | Columns / meaning | Main operations |
|---|---|---|
| `meta` | `key` primary key, `value`; includes `schema_version`, `model_name`, `embeddings_encrypted` | `_check_model_metadata()` writes/checks schema and model; `_configure_encryption()` records encryption state |
| `subjects` | `subject_id` primary key, `status` (`active`/`locked`/`disabled`), `created_at`, `consent_at`, `failed_attempts`, `locked_at`, `extra` JSON | `add_subject_with_embeddings`, `get_subject`, `list_subjects`, `set_status`, `increment_failed`, `reset_failed`, `delete_subject` |
| `embeddings` | `id` primary key, `subject_id` foreign key, `vector` BLOB, `quality`, `created_at` | `add_subject_with_embeddings`, `get_embeddings`, `delete_subject`; loaded/decrypted to rebuild FAISS |
| `sessions` | `session_id` primary key, `subject_id`, `purpose`, `created_at`, `expires_at`, `client_ip`, `allow_replace`, `metadata` JSON, `submit_count`, `terminal`, `result` JSON | `create_session`, `get_session`, `update_session`, `increment_session_submits`; API reads terminal result |
| `challenges` | `challenge_id` primary key, `session_id` foreign key, keyed `client_ip`, `actions` JSON, `issued_at`, `expires_at`, `consumed_at` | `create_challenge`, `get_challenge`, `consume_challenge`; single use and expiry/IP/session bound |
| `tokens` | `jti` primary key, `subject_id`, `expires_at`, `consumed_at` | `add_token`, `token_exists`, `consume_token`; backs one-use token state |
| `audit_log` | `id` primary key, `event`, optional `subject_id`, `created_at`, `data` JSON | `audit`, `list_audit`, `_insert_audit`; SQLite triggers reject update/delete |

The database does not store captured face images. The code stores numerical embeddings; if a valid `FACE_DB_KEY` is supplied, it encrypts those BLOBs with Fernet. Without that setting, it marks embedding storage unencrypted and writes raw float32 bytes. In this workspace, the local `.env` has `FACE_DB_KEY` empty and SQLite reports `embeddings_encrypted=disabled`; therefore, any embeddings written to this database would be unencrypted at the SQLite BLOB level. At inspection there were no embeddings (`embeddings` row count 0). Protect the database and its WAL sidecars as sensitive data regardless.

## 4. Internal APIs

### Analogy first

The current API has two kinds of keys at the front desk: a server API key for trusted back-end operations, and a short-lived session ticket for the browser camera. This prevents the service's long-lived server key from being embedded in public JavaScript.

### Current FaceVerify API (`face_verification/api.py`)

All paths below are prefixed by `/api/v1`. The line references point to route declarations; request/response details are summarized from the handlers and `openapi.yaml`.

| Method | URL | File / line | Input | Output and responsibility |
|---|---|---|---|---|
| POST | `/sessions` | `face_verification/api.py:116` | API key; JSON `purpose`, optional `subject_id`, `ttl_sec`, `allow_replace`, metadata | `201` session ID, expiry, and challenge requirement; creates enrollment, claimed verification, or identify session |
| GET | `/sessions/{session_id}/result` | `face_verification/api.py:171` | API key; path session ID | Detailed terminal result (scores and optional token); `404` if absent/not terminal |
| GET | `/session/challenge` | `face_verification/api.py:179` | `X-Session-Id` | Random two-action challenge ID, actions, expiry, frames per action |
| POST | `/session/preview` | `face_verification/api.py:192` | Session header; JSON base64 `frame` | Face-ready boolean and positioning guidance; no match result |
| POST | `/session/submit` | `face_verification/api.py:214` | Session header; JSON base64 `frames`, optional `consent` and `challenge_id` | Browser-safe decision/message/retry flag; routes to enroll, verify, or identify |
| GET | `/subjects` | `face_verification/api.py:285` | API key; optional `limit`, `offset` | Paginated opaque subject metadata |
| GET | `/subjects/{subject_id}` | `face_verification/api.py:293` | API key; path subject ID | Subject status and metadata; no embeddings |
| DELETE | `/subjects/{subject_id}` | `face_verification/api.py:301` | API key; path subject ID | Deletes subject, embeddings, related sessions/tokens; audit entry remains |
| POST | `/subjects/{subject_id}/unlock` | `face_verification/api.py:308` | API key; JSON reason code | Reactivates subject, clears failed count, records audit event |
| POST | `/tokens/introspect` | `face_verification/api.py:319` | API key; JSON JWT and optional `consume` | Token validity/claims or `active: false`; optionally consumes token once |
| GET | `/health` | `face_verification/api.py:334` | API key | Authenticated health status and configured model |
| GET | `/config` | `face_verification/api.py:339` | API key | Safe model/profile/challenge/anti-spoof settings and effective match threshold |

API-level controls include `X-API-Key` checks with constant-time comparison, session expiry/terminal checks, optional kiosk IP allowlist, a request-body size cap, preview and per-IP/subject throttles, response security headers, and consistent logged request IDs. Browser routes authorize by session ID, not API key.

### Demo gateway (`demo/demo_backend.py`)

| Method | URL | File / line | Input | Output and responsibility |
|---|---|---|---|---|
| GET | `/` | `demo/demo_backend.py:51` | Browser navigation | Home page |
| GET | `/enroll` | `demo/demo_backend.py:56` | Browser navigation | Enrollment page |
| GET | `/verify` | `demo/demo_backend.py:61` | Browser navigation | Identify page presented as “Verify” |
| GET | `/client/face-capture.js` | `demo/demo_backend.py:46` | Browser request | Serves shared camera module from `client/` |
| POST | `/service-api/sessions` | `demo/demo_backend.py:83` | Browser JSON with purpose and optional subject ID | Server-side authenticated creation of face-service session; returns session ID and API base URL |
| GET | `/service-api/sessions/{session_id}/result` | `demo/demo_backend.py:100` | Browser session ID | Proxies detailed service result for the demo |
| POST | `/service-api/tokens/introspect` | `demo/demo_backend.py:113` | Browser JSON token | Server-side introspection with `consume=true` |

The gateway uses `client/python_client.py`; `demo_backend.main()` mounts the face API under `/service` with `DispatcherMiddleware` and serves both through Waitress on port 5050 by default.

### Legacy API (`legacy/app.py`)

This is a separate older application, not the current API. It has no API-key/session authentication layer and saves names alongside its index.

| Method | URL | File / line | Input | Output and responsibility |
|---|---|---|---|---|
| GET | `/` | `legacy/app.py:95` | Browser navigation | Legacy landing page |
| GET | `/register` | `legacy/app.py:100` | Browser navigation | Legacy registration page |
| GET | `/verify` | `legacy/app.py:105` | Browser navigation | Legacy verification page |
| POST | `/register` | `legacy/app.py:110` | JSON with `name` and base64 image(s) | Detects/checks face, adds embeddings to legacy FAISS index, saves index/name mapping |
| POST | `/detect_face` | `legacy/app.py:162` | JSON with base64 `image` | Returns whether a centered face is detected |
| POST | `/verify` | `legacy/app.py:191` | JSON with base64 image(s) | Checks quality/movement, searches legacy index, returns match name/score or no match |

### Important internal functions and module boundaries

| Function / class | File / line | Role |
|---|---|---|
| `create_app()` / `create_blueprint()` | `face_verification/api.py:68,353` | Compose the Flask application and dependency objects; routes delegate work instead of implementing matching inline |
| `FaceService.enroll()` | `face_verification/service.py:74` | Enrollment workflow and decisions |
| `FaceService.verify()` | `face_verification/service.py:148` | Claimed-subject 1:1 matching only |
| `FaceService.identify()` | `face_verification/service.py:272` | Active-subject 1:N search |
| `FaceService._analyze_frames()` / `_score_liveness()` | `face_verification/service.py:386,400` | Shared per-frame quality/embedding and passive liveness pipeline |
| `FaceEngine.analyze()` / `embed()` / `detect_face()` | `face_verification/engine.py:66,72,89` | Detector, normalized embedding, lightweight preview readiness |
| `quality.assess()` | `face_verification/quality.py:40` | Rejects poor or unsafe-to-use frames |
| `ChallengeManager.issue()` / `consume()` | `face_verification/liveness.py:270,291` | Creates and atomically uses a randomized prompt |
| `FaceStore` methods | `face_verification/store.py:72-370` | Database persistence, FAISS build/search, tokens, challenges, audit |
| `TokenManager.issue()` / `introspect()` | `face_verification/tokens.py:27,47` | Signs/verifies HS256 JWT and checks one-use DB state |
| `ServiceClient` | `client/python_client.py:19` | Server-side HTTP client used by demo and consuming Python integrations |

## 5. External APIs and libraries

### Analogy first

There is no remote “face recognition cloud” in this app. The model is like equipment installed in the room: inference runs on the same machine. The demo's HTTP requests are conversations between its own browser, gateway, and local service.

### Technical detail: services and network calls

| Name | Where used | Why / data sent and received | Credentials / cost |
|---|---|---|---|
| FaceVerify HTTP API (internal) | `client/face-capture.js` and `demo/static/app.js`; backend routes in `face_verification/api.py` | Browser sends base64 camera frames, session/challenge IDs, and gets guidance/decisions. Demo server uses Python `ServiceClient` to send API key and session/token requests to its own mounted service URL. | No third-party credentials. The server uses `FACE_SERVICE_API_KEY`; data goes to the local FaceVerify service. |
| InsightFace model package download/cache (conditional) | `face_verification/engine.py` creates `FaceAnalysis` lazily; README documents first-use download | Local model initialization; may download model assets if the InsightFace cache does not already contain them. Face images are inference input, not sent as an API request to InsightFace. | No application API key configured. Model licensing restrictions apply; see README. Bandwidth/storage cost depends on package/model cache, not a per-request paid API. |
| MiniFASNet ONNX inference (local) | `face_verification/liveness.py:AntiSpoofEnsemble` | Loads two configured local model files and passes face crops to ONNX Runtime; receives logits/probabilities. | No service or API key, no usage quota; model files must be present. |
| Browser camera API (local device) | `client/face-capture.js`, `legacy/static/js/camera.js` | `navigator.mediaDevices.getUserMedia()` reads camera video; frames are encoded and posted to the app. | Browser permission required; browsers normally require localhost or HTTPS. |
| No cloud, maps, email, SMS, or external hosted AI endpoint | Project-wide code path inspection | No code call to a cloud service/provider was found. | None identified. |

### Main packages

| Package | Use in this repository |
|---|---|
| Flask | Routes, JSON responses, page rendering, request/response hooks, test client |
| Waitress | WSGI server for demo and API-only deployment |
| InsightFace | Face detection, landmarks, and ArcFace-style 512-element recognition embedding |
| ONNX Runtime | Executes local MiniFASNet anti-spoof models and supports InsightFace model inference |
| OpenCV (`cv2`) | JPEG decode/encode-related processing, resizing, grayscale/blur/lighting measures, face crops, geometry and offline plots |
| NumPy | Embedding normalization, vector similarity, statistics, tensor/array shaping |
| FAISS CPU | Exact inner-product nearest-neighbor index for active embeddings and the legacy index |
| `cryptography` | Fernet encryption/decryption of SQLite embedding blobs when configured |
| PyJWT | HS256 JWT issuance and validation |
| pytest | Test runner; `requirements-dev.txt` adds it |
| Python standard library | SQLite, JSON, base64, threading, HMAC, secrets, HTTP `urllib`, file/path operations, timing, and argparse |

The PyTorch exporter is a developer utility and imports `torch` only when run; PyTorch is not listed in `requirements.txt`. The repository contains `.pth` source weights and ONNX runtime weights. No unrelated runtime package was confidently identified as unused.

## 6. Process-wise API mapping

### Analogy first

Every flow uses a temporary numbered ticket. The ticket proves which browser is allowed to send the camera frames. Enrollment files a new reference card; claimed verification compares against one card; identify searches all active cards. A challenge asks a randomized action, and liveness/model checks are the guards before a result is accepted.

### Technical flow table

| Process | Step-by-step flow | Internal APIs/libraries and storage | Failure behavior |
|---|---|---|---|
| Start app | `demo.demo_backend.main()` loads settings, creates the Face API app, mounts it at `/service`, then starts Waitress | `config.load_settings` reads `.env`; `create_app`; `DispatcherMiddleware`; SQLite schema open | Missing service key/token secret or required anti-spoof model files stops startup with an error |
| Create session | UI calls demo `/service-api/sessions`; demo `ServiceClient` calls `/api/v1/sessions`; API validates purpose/ID/TTL, rate-limits and stores session | `demo/static/app.js`; `demo_backend.create_session`; `ServiceClient.create_session`; `FaceStore.create_session`; `sessions` table | Invalid request `400`, bad API key `401`, rate limit `429`, missing config/server error returns mapped gateway response |
| Start camera / preview | `FaceCapture.start()` gets browser camera; starts periodic preview. Preview frame is POSTed with session header | Browser camera API; `/session/preview`; `decode_image`; `FaceEngine.detect_face`; OpenCV/InsightFace | Camera permission/device failure displayed in UI; malformed image or detector error gives guidance and `400`; stale/invalid session rejected |
| Issue liveness challenge | Browser GETs `/session/challenge`; manager selects two distinct actions and persists one-use challenge | `ChallengeManager.issue`; `FaceStore.create_challenge`; `challenges` table; keyed client-IP binding | Challenge issuance errors return `500`; absent challenge later fails challenge validation |
| Capture and verify claimed ID (1:1) | Browser captures baseline/action frames and POSTs `/session/submit`; service loads active subject, quality-checks frames, scores liveness, consumes challenge, compares embeddings against that subject, checks runner-up impersonation margin and votes | `FaceCapture.captureVerification`; API `submit`; `FaceService.verify`; `FaceEngine`, `quality.assess`, `AntiSpoofEnsemble`; `subjects`, `embeddings`, `challenges`, `tokens`, `audit_log` | Bad quality returns retry; missing subject/repeated failures reject or lock; anti-spoof/config/challenge/mismatch failures reject; unexpected exceptions become logged `500` |
| Identify without ID (1:N) | Same camera flow but `/verify` page sends `purpose=identify`; service analyzes frames, validates liveness/challenge, searches the FAISS index for active subjects and aggregates votes | `FaceService.identify`; `FaceStore.search_topk`; in-memory FAISS rebuilt from `embeddings`; audit result | Empty database, weak score, insufficient votes, or close runner-up yields `no_match`; good match yields `identified_subject_id` |
| Enroll face | UI validates opaque ID and consent; browser sends enrollment session and multiple frames; service quality-checks, anti-spoofs, optionally challenges, removes inconsistent vectors, checks duplicates, then writes subject and embeddings | `/sessions`, `/session/submit`; `FaceService.enroll`; `FaceStore.add_subject_with_embeddings`; `subjects`, `embeddings`, `audit_log`; NumPy/FAISS | No consent, poor/insufficient frames, spoof, duplicate, or already-enrolled ID returns a reason-specific rejection/retry; missing required models prevents service use |
| Verify result/token | API issues a short-lived signed token after successful 1:1 verification; consuming backend reads terminal result and introspects/consumes the token | `TokenManager.issue/introspect`; `tokens` and `sessions`; PyJWT, HMAC-SHA256 | Invalid/expired/already-consumed token returns `active:false`; token-issuance failure returns retry |
| Manage a subject | Trusted caller lists/reads metadata, deletes profile or unlocks a locked subject with reason code | `/subjects*`; `FaceStore` CRUD/status methods; `FaceService.unlock_subject`; `audit_log` | Missing subjects return `404`; invalid unlock reason returns `400`; delete removes embeddings/sessions/tokens for that subject |
| Calibrate match threshold | Operator runs calibration script; it reads active DB embeddings, forms genuine/impostor pairs, calculates rates, writes selected JSON threshold | `tools/calibrate_from_db.py`; `FaceStore.get_embeddings`; NumPy; `face_data/threshold.json` | Empty/small sample warns; if no threshold meets target, no suggestion is written |
| Legacy registration/verification | Old browser polls `/detect_face`, posts image arrays to `/register` or `/verify`; embeddings and names are saved in a separate FAISS index | `legacy/static/js/camera.js`; `legacy/app.py`; `legacy/face_utils.py`; `legacy/vector_store.py`; `database/faces.index`, `database/names.json` | Legacy uses simpler quality/movement checks, returns JSON errors; no current API session or server API-key protection is present |

## 7. End-to-end data-flow example: browser “Verify” / identify

### Analogy first

The user walks to the desk without giving an ID. The service checks whether the live visitor is really present, then searches the stored profile cards and reports the best sufficiently strong match—or says that no confident match exists.

### Technical sequence

1. The user opens `GET /verify`. `demo/demo_backend.py:61` renders `demo/verify.html`. Its `<body data-mode="identify">` is read by `demo/static/app.js`.
2. The user clicks Start camera. `app.js` calls the demo gateway `POST /service-api/sessions` with `{ "purpose": "identify" }`. This is an **ID-free identification** flow, even though the page calls itself “Verify”.
3. `demo_backend.create_session()` calls `ServiceClient.create_session()` in `client/python_client.py`. The server sends `POST /service/api/v1/sessions` with `X-API-Key`; the browser never receives that key.
4. `face_verification/api.py:create_session()` creates a random session ID. For identify, it stores an internal sentinel subject ID and stores purpose as `verify` because the SQLite schema permits only `enroll`/`verify`. `FaceStore.create_session()` inserts into `sessions`; the API returns session ID and expiry.
5. `FaceCapture.start()` in `client/face-capture.js` requests browser camera access and rejects device labels matching configured virtual-camera terms. Its preview loop repeatedly POSTs a reduced JPEG data URL to `/service/api/v1/session/preview` with `X-Session-Id`.
6. `api.preview()` validates the session, then calls `decode_image()` and `FaceEngine.detect_face()`. OpenCV decodes the base64 JPEG to BGR; InsightFace's detector checks face count, confidence, size, and centering. The preview result is guidance only. No image is saved.
7. When the user clicks the action button, `FaceCapture.captureVerification()` GETs `/service/api/v1/session/challenge`. `ChallengeManager.issue()` chooses two distinct actions from blink/turn-left/turn-right, saves action/timing/session/IP-binding metadata in `challenges`, and returns the prompts.
8. The browser captures one baseline frame plus the configured burst of frames per action (default: 3 frames for each of 2 actions), then POSTs the base64 frames and challenge ID to `/service/api/v1/session/submit`.
9. `api.submit()` checks frame list, increments session submit count, decodes frames, rate-limits, sees the identify sentinel, and calls `FaceService.identify()`.
10. `FaceService._analyze_frames()` runs `FaceEngine.analyze()`, `quality.assess()`, and `FaceEngine.embed()` per frame. Quality checks include face count/size, detector confidence, blur, brightness/contrast, and configured optional pose/eye gates. Invalid frames are excluded.
11. `FaceService._score_liveness()` invokes the two local MiniFASNet ONNX models in `AntiSpoofEnsemble.is_live()` for each candidate. `evaluate_challenge_actions()` checks blink/head-turn movement on landmarks. The challenge must be unused, belong to this session and IP binding, and satisfy time bounds.
12. The service keeps the best quality frames, requires their embeddings to be mutually consistent, then calls `FaceStore.search_topk()` for each vector. `FaceStore` searches its in-memory FAISS `IndexFlatIP`; the index was built from SQLite embedding BLOBs when the store opened and is rebuilt after writes/deletes.
13. `FaceService.identify()` aggregates votes and scores, ignores non-active profiles, applies the similarity threshold and runner-up margin, and records an `identified` audit event if successful.
14. `api.submit()` stores the result JSON and terminal flag in the `sessions` row. It returns only the browser-safe decision/message/retry fields plus `identified_subject_id` on a successful match. `demo/static/app.js` shows “Face matched: …” and stops the camera.
15. No photograph is persisted. The persistent biometric output is the embedding BLOB from enrollment; identification only reads it. Unlike claimed-subject 1:1 verification, this identify flow does not issue a JWT.

### Important distinction for a viva

There are two different “verification” meanings in the code:

- **1:1 verify API purpose:** a backend supplies a claimed `subject_id`; `FaceService.verify()` compares only against that person's profile and a successful result can create a one-use JWT.
- **Current demo `/verify` page:** sets `data-mode="identify"` and sends `purpose="identify"`; `FaceService.identify()` searches all active profiles and returns an ID. It is 1:N identification, not a claimed-ID 1:1 verification.

## 8. Problems and improvements

### Analogy first

The current service has sensible locks and checks, but a production review should also check whether the records cabinet is actually locked, whether the container carries all its examination equipment, and whether the filing/search method will still be fast when the cabinet becomes enormous. The legacy room is a separate, older design and should not be mistaken for the secured service.

### Technical observations

| Area | What the code shows | Improvement / review point |
|---|---|---|
| Secret management | `.env.example` uses placeholder API/JWT secrets; `FACE_DB_KEY` is blank there. `config.py` accepts secrets from environment or local `.env`. | Use unique high-entropy production secrets from a secret manager; do not deploy placeholders. Configure and back up Fernet key securely if at-rest embedding encryption is required. The real `.env` was not read, so whether real secrets are present/strong is **unsure**. |
| Database encryption | Embeddings are encrypted only when `FACE_DB_KEY` is set. Metadata, sessions, tokens, audit details, and WAL files are not encrypted by this code. | Decide and document encryption-at-rest and database/WAL backup policy; restrict filesystem access. Consider OS/disk/database encryption for metadata too. |
| Legacy exposure | `legacy/app.py` has direct `/register`, `/verify`, and `/detect_face` routes, saves names in JSON, and has no API-key or session guard. | Do not expose the legacy server on an untrusted network. Prefer removing/isolating it after confirming it is no longer needed, or add proper authentication/privacy controls before use. |
| Demo gateway access | `/service-api/*` routes have no user login or CSRF protection. The API key remains server-side, but anyone able to reach the demo can request a session and interact with it. | Treat the demo as local-only; for deployment add consuming-application authorization, origin/CSRF controls, HTTPS, and abuse controls appropriate to the deployment. |
| Network transport | `.env.example` defaults to loopback; `Dockerfile` exposes the service on `0.0.0.0` and HTTP. Kiosk mode only warns if HTTPS termination is not declared. | Terminate TLS at a trusted proxy/load balancer; bind/firewall the service appropriately; set `HTTPS_TERMINATED` only when that is actually true. Never send API keys or frames over public plaintext HTTP. |
| Docker model availability | `.dockerignore` excludes `models/`, but the default service startup requires both configured anti-spoof `.onnx` files and the Dockerfile does not mount them. | Decide how licensed models are supplied at runtime (secure volume/secret artifact or controlled image inclusion) and document the required mount paths; verify a clean container can start. |
| Request frame work | `/session/submit` decodes every item in `frames` before `FaceService` checks maximum enrollment/verification frame counts. Flask caps the whole body at 16 MiB, but a large list can still do unnecessary decode work. | Reject over-limit frame counts before decoding, and consider per-frame encoded-size limits in addition to request-size and pixel limits. |
| Per-process throttling | API `RateLimiter` and preview timestamps are in-memory; comments/README note multiple workers do not coordinate. Preview timestamps are keyed by IP and do not show an expiry cleanup. | Use a shared rate-limit store for multi-process deployments; bound/expire in-memory bookkeeping. Keep single-process deployment constraint explicit if retained. |
| Database growth | Session/challenge/token rows have expiries, but the reviewed code does not show a scheduled purge for expired rows. Audit log is intentionally append-only. | Define retention and privacy policy; add safe cleanup for expired workflow rows while preserving required audit history. |
| FAISS scaling | Current store uses exact `IndexFlatIP`, rebuilds the full index from SQLite at startup and after enrollment/deletion, and identification asks for all subject candidates per frame. | Benchmark with expected enrollment size; use batching/cached metadata and an approximate/index-ID strategy if scale requires it. Keep thresholds validated against real test populations. |
| Local project size | The workspace scan found `.venv` at about **1.09 GB**; it contains large native PyTorch/OpenCV/FAISS/ONNX binaries. `.gitignore` and `.dockerignore` exclude `.venv`, so it is not source data needed by the application. | Do not include/copy the virtual environment when sharing or packaging the source. Recreate it from pinned requirements. PyTorch is optional exporter tooling, not a runtime requirement. |
| Python version mismatch | The workspace venv contains CPython 3.14-tagged native extensions, while README and requirements comment recommend Python 3.11/3.12. | Verify the selected environment and native-wheel compatibility against the documented versions; recreate under the intended version if reproducibility is important. |
| Legacy vs current data | Current service uses `face_data/face.db`; legacy app uses `database/faces.index` and `database/names.json`. They are not automatically synchronized. | Make the active entry point/data directory unambiguous. Use `tools/import_legacy.py` only for confirmed test data; it explicitly requires `--confirm-test-data-only`. |
| Model/threshold reliability | README notes InsightFace model licensing restrictions and calibration scripts warn when sample sizes are inadequate. Eye landmark indices are explicitly documented as an assumption. | Validate model/data licenses, landmark mapping, false-accept/false-reject rates, and spoof performance on representative consented data before production. Do not treat test calibration as a security guarantee. |
| Tests | Nine pytest modules cover API, service, store, engine, quality, liveness, calibration, legacy vector store, and web gateway. | Run `pytest` in the supported Python environment before a review/release; add runtime/container smoke tests and test coverage for operational cleanup/deployment paths if those are introduced. |

### Good design choices worth explaining in a review

- Frames are processed in memory; the store API accepts vectors, not image pixels.
- Subject IDs are constrained to opaque safe characters, and the subject “extra” filter rejects common personal-data field names.
- Browser camera routes use short-lived session IDs; backend administration uses a separate API key.
- Challenges are randomized, short-lived, session/IP bound, and consumed atomically.
- Verification tokens are signed, short-lived, and backed by database state so they can be consumed once.
- SQLite writes are transaction-wrapped, foreign keys are enabled, and audit rows are protected from update/delete by triggers.
- The service uses dependency injection in `create_app()` and injectable clocks/model factories in important logic, which makes it testable.

## Quick viva questions to practice

1. **Why embeddings instead of saved photos?** An embedding is a compact numerical representation used for similarity comparison; the submitted frames are transient. It is still sensitive biometric data and must be protected.
2. **What is the difference between 1:1 and 1:N?** 1:1 checks a claimed subject ID against that subject's vectors; 1:N searches all active subjects and returns the best qualifying identity.
3. **Why normalize vectors?** For normalized vectors, inner product is cosine similarity, making similarity scores comparable by direction rather than vector magnitude.
4. **Why have both passive and active liveness?** MiniFASNet scores whether a face crop looks live; the challenge asks for observable movement. They provide different signals and neither is a perfect guarantee by itself.
5. **Why is there a session table?** It keeps the browser flow short-lived and ties browser requests, challenge state, expiry, submit count, and the final result together without exposing the backend API key.
6. **What does FAISS store here?** A process-local search index reconstructed from the SQLite vectors; the current app's durable source of truth remains SQLite.
7. **What happens when a subject is deleted?** The service writes a deletion audit event, removes related sessions/tokens and the subject (foreign-key cascade removes embeddings), then rebuilds the FAISS index.
8. **What does the demo's Verify button actually do?** It performs ID-free 1:N identification because `verify.html` sets `data-mode="identify"`; true claimed-ID 1:1 verification exists in the service API.
