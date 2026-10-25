# Face service

This Python FastAPI service runs independently from the Node.js backend. From this directory, install `requirements.txt` and start it with:

```text
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Set `FACE_SERVICE_API_KEY` to a strong secret of at least 32 characters. Both MiniFASNet ONNX files are included in `models/` and are required at startup. InsightFace downloads the `buffalo_l` model the first time it starts and caches it under `~/.insightface` unless `INSIGHTFACE_MODEL_ROOT` is set.

For native Render deployment, the repository-root `render.yaml` defines this service and the Node backend. The backend's `FACE_SERVICE_API_KEY` must match this service's key, and its `FACE_SERVICE_URL` must point to this service's Render URL.
