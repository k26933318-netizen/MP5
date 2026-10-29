# VoteLedger

VoteLedger is an electronic-voting prototype with voter, administrator, election-officer, and candidate accounts. The local development profile uses MongoDB for application records and a local hash-linked vote ledger. Hyperledger Fabric is available when an operator supplies a running network, deployed chaincode, and valid identities. Voter face matching is performed by a separate Python service using a RetinaFace MobileNet detector and ArcFace embeddings.

> **Prototype only:** this project is not suitable for a real public election without independent security, biometric, privacy, accessibility, legal, and election-system validation. Face matching, active head-movement checks, and passive presentation-attack detection are separate probabilistic controls; none alone proves a person's legal identity or prevents every attack. The local hash-linked ledger is not equivalent to a distributed blockchain. Biometric data needs strong organizational, technical, and legal controls.

## Features

- Voter registration captures three face frames and checks positioning, image quality, stability, presentation-attack scores, and duplicate enrollment before account creation; a white background is not required, and only the encrypted template is retained.
- Admin approval; role-based sign-in for voters, admins, election officers, and candidates.
- Party, alliance, candidate, officer, and election administration.
- Election lifecycle rules and voting-date validation.
- Live camera feedback uses RetinaFace bounding boxes to move a tracking box independently of the fixed positioning guide. Enrollment allows a wider position range and does not require a white background; voting verification retains its stricter position and background checks.
- Recognition requires a backend-issued, randomized head-turn-and-return challenge, multiple ordered camera frames, and the two-model MiniFASNet passive presentation-attack check. Challenge use is bound to voter, election, browser session, expiration, and one-time consumption.
- RetinaFace requires exactly one face and supplies five landmarks for ArcFace-compatible alignment.
- ArcFace produces a 512-dimensional embedding, L2-normalized before cosine-similarity comparison.
- AES-256-GCM encrypted embedding storage in MongoDB; images and embeddings are not put in votes, receipts, or the ledger.
- Short-lived verification authorization bound to voter, election, browser session, and enrollment version; stored as a hash and consumed once.
- MongoDB unique vote claims protect against concurrent duplicate votes.
- Vote receipts, Fabric transaction lookup, local-ledger integrity checks, and audit events.
- Election and candidate records are synchronized to Fabric before voting; Fabric mode does not silently fall back to the local ledger.

## Architecture

```text
Browser camera
    -> Node.js / Express API
       -> Python FastAPI face service (internal service key)
          -> RetinaFace MobileNet detection and five-point landmarks
          -> InsightFace ArcFace alignment, embedding and L2 normalization
          -> background, positioning, multi-frame stability and quality checks
          -> randomized head-movement challenge and local MiniFASNet PAD models
       -> MongoDB (voters, encrypted face embeddings, elections,
                   hashed one-time authorizations, unique vote claims, audit logs)
       -> Local development ledger OR explicitly configured Hyperledger Fabric
```

The face service uses the 1.7 MB RetinaFace MobileNet-0.25 ONNX detector and the 13.6 MB InsightFace `buffalo_sc` ArcFace MobileFaceNet weights to reduce the model footprint for Render Free. The ArcFace archive is bundled and extracted at startup. Changing from the former `buffalo_l` pipeline changes the embedding model version, so previously enrolled voters must re-enroll. The two MiniFASNet ONNX anti-spoof models are expected under `FRS2/face-recognition-app/models` for local execution and are copied into `face-service/models` for deployment. Review licenses and provenance for all model files; the InsightFace pretrained weights are for non-commercial research use. The training `.pth` files and InsightFace cache remain outside version control.

### Face model contract

- Detector: the bundled RetinaFace MobileNet-0.25 ONNX model that returns boxes and five landmarks.
- Recognizer: the InsightFace `buffalo_sc` **ArcFace MobileFaceNet** model whose output is 512-dimensional.
- Alignment: InsightFace ArcFace five-point similarity transform to the model input size (typically 112×112), from BGR image pixels.
- Quality checks: minimum image/face dimensions, detector confidence, landmark geometry, face position/size, roll/yaw proxy, blur, brightness, and frame-to-frame stability. Voting verification also checks the near-white background; enrollment does not require it.
- Background policy: for voting verification, the service estimates the person region from the detected face and samples surrounding pixels. It is not semantic human segmentation, so camera framing and threshold calibration matter.
- Presentation-attack detection: the two configured MiniFASNet ONNX models score every submitted frame. The service fails startup if PAD is required (the default) and either model is unavailable. The PAD score is an additional probabilistic check, not a guarantee against sophisticated attacks.
- Similarity: cosine similarity on L2-normalized ArcFace embeddings; the same configured threshold is used for duplicate enrollment and voting verification. The default `0.50` is a starting point only and must be validated against representative genuine/impostor samples for the exact model and deployment; this repository's synthetic unit-test embeddings do not calibrate biometric accuracy.
- Duplicate enrollment checks compare against every currently enrolled encrypted template, including the requesting voter's existing template. Registration checks and persistence are serialized by a MongoDB-backed expiring lock shared across backend processes; the in-process queue reduces contention but is not the cross-process safeguard.

The service loads its face-recognition and anti-spoof models once at startup and fails closed when required models or the internal service key are unavailable. It will not silently substitute another detector or recognizer. It reuses FRS2's cached recognition/detection weights and anti-spoof ONNX files, not FRS2's separate SQLite enrollment database.

## Prerequisites

- Node.js 20 or newer and npm.
- MongoDB 7 or a compatible MongoDB deployment.
- Python 3.11 or 3.12 for the pinned face-service dependencies.
- The configured ArcFace weights under `INSIGHTFACE_MODEL_ROOT` for local legacy profiles; the `buffalo_sc` profile uses the bundled, checksum-verified MobileFaceNet ArcFace archive.
- The two MiniFASNet ONNX files (`minifasnet_a.onnx` and `minifasnet_b.onnx`) under `FRS2/face-recognition-app/models` for local execution; deploys use the copies under `face-service/models`.
- A modern browser with camera access. Camera access requires HTTPS, except for localhost.

## Configuration

Copy `.env.example` to `.env`, then set strong unique secrets. Generate secrets with:

```powershell
node -e "console.log(require('crypto').randomBytes(32).toString('base64'))"
```

Generate two different random values: set one as `FACE_ENCRYPTION_KEY` (a 32-byte key, base64 or 64-character hex) and the other as `FACE_SERVICE_API_KEY` (a long random string). Do not reuse keys for different purposes. Configure `FACE_ENCRYPTION_KEY_VERSION` before rotating keys; existing embeddings must be re-encrypted or voters re-enrolled during a controlled key rotation.

Required face-related settings:

```dotenv
FACE_SERVICE_URL=http://127.0.0.1:8000
FACE_SERVICE_API_KEY=replace-with-a-long-random-secret
FACE_ENCRYPTION_KEY=replace-with-a-generated-32-byte-key
FACE_ENCRYPTION_KEY_VERSION=1
FACE_SIMILARITY_THRESHOLD=0.50
INSIGHTFACE_MODEL_NAME=buffalo_sc
INSIGHTFACE_MODEL_ROOT=C:\Users\YOUR_USERNAME\.insightface
FACE_DETECTION_SIZE=640
FACE_MIN_DETECTION_SCORE=0.70
FACE_BACKGROUND_MIN_BRIGHTNESS=190
FACE_BACKGROUND_MAX_STDDEV=24
FACE_BACKGROUND_MIN_WHITE_FRACTION=0.60
FACE_DIAGNOSTICS_ENABLED=true
FACE_CENTER_X_TOLERANCE=0.14
FACE_CENTER_Y_TOLERANCE=0.18
FACE_MIN_WIDTH_FRACTION=0.24
FACE_MAX_WIDTH_FRACTION=0.62
FACE_MAX_ROLL_DEGREES=12
FACE_MAX_PITCH_DEVIATION=0.22
FACE_MAX_YAW_PROXY=0.18
FACE_STABILITY_TOLERANCE=0.045
FACE_CHALLENGE_TTL_SECONDS=90
FACE_CHALLENGE_FRAMES_PER_STAGE=3
FACE_CHALLENGE_MIN_DURATION_MS=1500
FACE_CHALLENGE_YAW_DELTA=0.14
FACE_CHALLENGE_RETURN_TOLERANCE=0.08
FACE_ANTISPOOF_THRESHOLD=0.70
FACE_ANTISPOOF_MIN_FRAME=0.50
FACE_REQUIRE_ANTISPOOF=true
```

The service extracts `buffalo_sc/w600k_mbf.onnx` from the checksum-verified archive bundled under `face-service/models/` into the system temporary directory. RetinaFace is loaded from the bundled `face-service/models/retinaface-mnet025.onnx`. Other configured model names use `INSIGHTFACE_MODEL_ROOT`. The health endpoint reports database, face-service, and configured blockchain backend status without exposing secrets.

Background checks use OpenCV GrabCut foreground segmentation initialized with face landmarks' detection box, a head/hair estimate, and a shoulder/body prior; pixels marked uncertain foreground are excluded from the denominator. The configured threshold is applied to near-white pixels divided by valid estimated background pixels. `FACE_BACKGROUND_MIN_WHITE_FRACTION` defaults to 0.60 and can be configured from 0.50 to 0.75; `FACE_BACKGROUND_MIN_BRIGHTNESS` and `FACE_BACKGROUND_MAX_STDDEV` tune near-white classification and uniformity. `FACE_DIAGNOSTICS_ENABLED` writes structured validation diagnostics to the face-service development logs; keep it disabled for production. The mask is an estimate, not semantic segmentation, and still needs calibration on representative webcams. The server returns only generic user guidance and does not expose background scores or thresholds through the API.

The voting face service requires both PAD models. For local execution, its defaults resolve to the two `.onnx` files in `FRS2/face-recognition-app/models`. Native Render deploys use the tracked copies in `face-service/models`; Docker Compose mounts the local model directory read-only at `/anti-spoof`. `FACE_REQUIRE_ANTISPOOF` must remain `true`; the service fails startup rather than treating head movement alone as presentation-attack detection.

## Local installation and running

From the project root in PowerShell:

```powershell
Copy-Item .env.example .env
npm install
```

Edit `.env` with MongoDB, admin-password, face-service API, and encryption-key settings. The checked-in local profile uses `BLOCKCHAIN_BACKEND=local` so the vote flow does not depend on an unavailable Fabric network. Set `BLOCKCHAIN_BACKEND=fabric` only when a peer, channel, upgraded chaincode, TLS certificate, signing certificate, and private key are configured. Set `INSIGHTFACE_MODEL_ROOT` to the FRS2 model cache. Start MongoDB separately, then run the face service in one terminal:

```powershell
cd face-service
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, from the project root:

```powershell
npm start
```

Open `http://localhost:3000`. The camera preview shows a fixed positioning guide and a separate tracking box that follows the detected face. A registrant keeps one face visible and steady for three frames; the background does not need to be white. The server repeats quality, position, stability, and PAD checks, compares against existing enrollments, and only then creates a pending account and stores the encrypted template. The voter must be approved before login.

For voting, the signed-in voter selects a candidate, passes live position/background checks, and requests a random server-stored left- or right-turn challenge. The browser captures three baseline frames, three movement frames, and three return-to-center frames. The server checks challenge expiry and one-time use, validates the sequence, quality, background, PAD and face match, then issues the existing short-lived authorization bound to the voter, election, browser session, and enrollment version. The vote API independently consumes that authorization; browser-reported success is not sufficient.

The face service reads its settings from the project `.env`. Python 3.11 or 3.12 is recommended for the pinned computer-vision dependencies. Do not use sample/demo voter credentials or seeded demo data in a real deployment.

## Docker development

Configure `.env` and ensure the FRS2 recognition cache exists at `INSIGHTFACE_MODEL_ROOT` and `FACE_ANTISPOOF_MODEL_DIR` points to the two MiniFASNet model files, then run:

```powershell
docker compose -f docker-compose.app.yml up --build
```

The base Compose file runs the local ledger and mounts both the configured InsightFace cache and PAD model directory read-only. To run Fabric in Compose, set the Fabric peer endpoint, TLS host alias, certificates, and signing key in `.env`, then run:

```powershell
docker compose -f docker-compose.app.yml -f docker-compose.fabric.yml up --build
```

The face service is only exposed to the Docker network, not published on a host port. MongoDB data is persisted in a named volume. The compose setup is for development; it does not configure production TLS, backups, key management, monitoring, or network policies.

## Render deployment (native runtimes)

The root `render.yaml` defines two native Render web services; Docker is not used:

- `mp5-backend` runs the Node.js/Express API and serves the frontend.
- `mp5-face-service` runs the Python FastAPI recognition service from `face-service/`.

In Render, create or update a Blueprint from this repository and review its environment variables before deploying. Because both services use the free plan, `FACE_SERVICE_URL` references the face service's public HTTPS URL. The face API requires `FACE_SERVICE_API_KEY`; set the same strong, random value on both services. Set `MONGODB_URI` to the Atlas driver URI on `mp5-backend`, grant its database user read/write access, and configure the Atlas IP access list for the backend's Render outbound IP ranges. Set `FACE_ENCRYPTION_KEY` to a separate random 32-byte key and provide a strong `ADMIN_USERNAME` and `ADMIN_PASSWORD`.

The two runtime MiniFASNet ONNX files, RetinaFace detector, and compact ArcFace archive are included under `face-service/models/`; training `.pth` files, virtual environments, caches, and secrets are not. Model extraction uses ephemeral storage on Render, so cold starts repeat it after an instance restarts. The free Render plan is for testing only: face inference may exceed its available CPU, and its filesystem is ephemeral.

The Blueprint defaults to `BLOCKCHAIN_BACKEND=local` for testing. Its file-backed ledger is lost when a free instance restarts or redeploys, so do not use this mode for real votes. For Fabric, change the Blueprint to `BLOCKCHAIN_BACKEND=fabric`, provide a reachable peer, and configure the peer endpoint, TLS host alias, and secret certificate/key files. Durable local-ledger storage requires a paid Render persistent disk and a configured `VOTES_LEDGER_PATH`.

The root `.env` is for local development only. Never paste it into Render or commit it; enter each required value in the Render dashboard as a secret.

## API

All application endpoints are under `/api/blockchain`:

| Method | Endpoint | Access | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | Public | MongoDB, face service, and backend health |
| `POST` | `/api/blockchain/auth/login` | Public | Role-specific login for voters, admins, election officers, and candidates |
| `POST` | `/api/blockchain/voters` | Public | Multipart `images` frames plus voter details and explicit face-template consent; server validates the sequence and rejects duplicate faces before account creation |
| `GET` | `/api/blockchain/voters` | Admin, election officer | List voters and approval statuses |
| `PATCH` | `/api/blockchain/voters/:voterId/approve` | Admin, election officer | Approve or reject voter registrations |
| `POST` | `/api/blockchain/candidate-accounts` | Admin, election officer | Create candidate accounts |
| `GET` | `/api/blockchain/candidate-accounts` | Admin, election officer | List candidate accounts |
| `POST` | `/api/blockchain/parties` | Admin | Multipart party details and optional JPEG, PNG, or WebP logo (2 MB maximum) |
| `PATCH` | `/api/blockchain/parties/:partyId/logo` | Admin | Replace the logo for an existing party |
| `GET` | `/api/blockchain/parties` | Authenticated | Party data, including logos shown on the ballot |
| `POST` | `/api/blockchain/face/assess` | Public, rate limited | Multipart `image`; server-side single-frame guidance only, never an authorization |
| `POST` | `/api/blockchain/face/enroll` | Approved voter | Multipart `images`; validate multi-frame position, quality, stability, PAD, duplicate face, and securely store enrollment |
| `POST` | `/api/blockchain/face/challenge` | Approved voter | JSON `electionId`; issue a random, session-bound, expiring head-turn challenge |
| `GET` | `/api/blockchain/face/status` | Voter | Return enrollment status only |
| `POST` | `/api/blockchain/face/verify` | Approved voter | Multipart nine ordered `images`, `electionId`, and `challengeId`; return verified flag and one-time token only after server-side identity, challenge, and PAD checks |
| `POST` | `/api/blockchain/vote` | Voter | JSON `electionId`, `candidateId`, and one-time `verificationToken` |
| `GET` | `/api/blockchain/results/:electionId` | Admin | Election totals by candidate |
| `GET` | `/api/blockchain/election/:electionId/votes` | Admin | Anonymized vote records for the admin transaction explorer |
| `GET` | `/api/blockchain/transactions?electionId=...` | Admin | Anonymized transaction list |
| `POST` | `/api/blockchain/receipt/verify` | Authenticated user | Verify a receipt by transaction ID |
| `GET` | `/api/blockchain/ledger/:electionId/integrity` | Admin | Verify local-ledger links/hashes (local backend only; Fabric auditing uses peer/orderer tools) |

`POST /api/blockchain/voters` accepts multipart `images` and a JSON `registration` field containing voter details and `biometricConsent: true`. The service independently validates the captured sequence before checking for duplicate face matches and creating the voter. A close match is rejected with HTTP 409 and `DUPLICATE_FACE`, without returning another voter's identity. The duplicate check uses the configured cosine-similarity threshold; calibrate it for the selected model and deployment because face similarity is probabilistic, not a legal identity proof. Registration comparisons and persistence are protected by a MongoDB-backed expiring lock shared across backend processes. Voter and face-template records are rolled back if persistence or audit logging fails.

The administrator dashboard includes an election results and transaction panel. It shows candidate totals and recorded vote transactions, refreshes every 15 seconds, and includes prior records. Voter identifiers, voter hashes, and biometric data are excluded from explorer responses. Face enrollment reset is not exposed by the application.

Party logos uploaded by administrators are saved in MongoDB and associated with ballot candidates by party name. Administrators can replace logos for existing parties, including seeded parties. Logos are limited to JPEG, PNG, and WebP; server-side checks validate image signatures as well as upload MIME type.

With Fabric selected, the backend synchronizes missing election and candidate records from MongoDB to the configured channel before ballot retrieval or vote submission. Deploy the updated chaincode with the Fabric operator's normal lifecycle/upgrade process before using this backend. Votes are recorded only through Fabric; a failed submission releases its temporary MongoDB claim so the voter can retry. Fabric's on-chain duplicate-vote key remains the final guard. Fabric peer/orderer tools are required for block-level audits; the local-ledger integrity endpoint is not a Fabric block verifier.

The old `/voters/:voterId/face` route remains as an enrollment alias, but it accepts the same multi-frame `images` upload and always uses the authenticated voter's own ID. The browser never calls the Python service directly.

The Python face service exposes `/health`, `/detect`, `/embedding`, `/verify`, `/assess`, `/frames/enroll`, and `/frames/verify`; all endpoints require `X-Face-Service-Key`. Keep it on a private interface/network.

## Data and privacy

Face enrollment records are stored separately from voters and contain the AES-GCM ciphertext, IV, authentication tag, key version, model version, 512-dimension metadata, and timestamps. Voter records include a face-consent timestamp. API responses never include the ciphertext or embedding. Original registration images are processed in memory and are not retained. Existing legacy face records without the new enrollment schema are marked not enrolled and their legacy `encryptedEmbedding` field is removed on repository initialization; those accounts cannot participate in duplicate-face checks until they enroll a face.

Vote records contain a one-way voter reference hash, election/candidate IDs, and Fabric transaction metadata—not face data or a raw voter ID. MongoDB keeps a unique `(electionId, voterHash)` vote claim to prevent concurrent repeat submissions.

## Testing

From the project root:

```powershell
npm test
```

For the face-service unit tests:

```powershell
cd face-service
python -m unittest discover -s tests
```

Integration testing needs a running MongoDB, both correctly licensed model files, and the face service configured with the same service key as the Node backend. Check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health -Headers @{ 'x-face-service-key' = $env:FACE_SERVICE_API_KEY }
Invoke-RestMethod http://127.0.0.1:3000/api/health
```

Exercise enrollment, wrong-person verification, no-face/multiple-face images, invalid/expired/reused verification tokens, cross-voter/session/election token use, inactive elections, invalid candidates, and concurrent duplicate vote requests before relying on a deployment.

The automated face-service workflow tests mock detector, recognizer, and PAD outputs; they test the validation rules but are not real-camera or biometric-accuracy evaluations. The configured InsightFace and MiniFASNet assets can be loaded locally with:

```powershell
cd face-service
python -c "from app.services.detection_service import create_runtime; from app.services.antispoof import anti_spoof; create_runtime(); anti_spoof.load(); print('face and PAD models loaded')"
```

## Limitations and remaining deployment work

- RetinaFace/ArcFace matching uses a compact ONNX model pair. InsightFace pretrained ArcFace weights are restricted to non-commercial research use; review licensing before production.
- Thresholds and quality cutoffs are configurable engineering defaults; they require demographic and environmental validation for the selected model/camera.
- Background validation combines GrabCut with detector bounds and a conservative head/shoulder prior; it is not a dedicated semantic person-segmentation model. Hair, clothing, scene composition, camera auto-exposure, and framing can influence results. Calibrate `FACE_BACKGROUND_MIN_WHITE_FRACTION` against intended webcams and lighting.
- Active head-turn checks and MiniFASNet PAD scoring are independent, probabilistic signals. Neither guarantees resistance to high-quality replay, masks, deepfakes, or other presentation attacks. The included tests mock model outputs and do not establish biometric accuracy or security.
- Face-position guidance is obtained from a backend assessment request at intervals; UI readiness is advisory. The final server endpoints re-run required checks against submitted frames and do not trust frontend flags.
- The challenge is bound to a MongoDB record, voter, election, session, expiry, and single use. It relies on the backend process clock and submitted ordered still frames; production deployments still need adversarial testing for frame replay/timing manipulation and a stronger validated liveness/PAD design.
- The local profile is a development demonstration, not a distributed ledger. Fabric requires a real network, channel, upgraded chaincode, identities, and TLS files; neither backend silently falls back to the other.
- The optional local ledger is a single-machine hash-linked file, not a distributed blockchain or consensus network. Its append/check process is not a substitute for a production ledger.
- Fabric block-level integrity checking is handled by Fabric peer/orderer operational tools; this sample does not independently verify Fabric block hashes.
- Sessions are currently in-memory and are lost on restart; horizontal scaling requires a shared session store.
- Key rotation, secure HSM/KMS integration, production transport security, privacy/legal review, formal threat modeling, accessibility testing, operational recovery, and election certification remain necessary.
"# MP5" 
