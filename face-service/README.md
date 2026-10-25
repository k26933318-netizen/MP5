# Face service

This Python FastAPI service runs independently from the Node.js backend. From this directory, install `requirements.txt` and start it with:

```text
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Set `FACE_SERVICE_API_KEY` to a strong secret of at least 32 characters. Both MiniFASNet ONNX files, the 1.7 MB RetinaFace MobileNet-0.25 ONNX detector, and the 16 MB InsightFace `buffalo_sc` ArcFace MobileFaceNet archive are included in `models/`. At startup, the service checks the archive SHA-256 and extracts only the required package files into the configured InsightFace model cache. The detector runs on CPU with five-point landmarks. The detector comes from [clibdev/Pytorch_Retinaface](https://github.com/clibdev/Pytorch_Retinaface), which publishes a SHA-256 checksum; the bundled model matches `4aa128919a621c913b3bf78befa33eaaba086a7a7e142cdb739839a9340f2420`.

Changing detector and recognition weights changes the embedding model version. Voters enrolled with the former `buffalo_l` pipeline must re-enroll before they can pass face verification with this lighter pipeline. The InsightFace pretrained model weights are for non-commercial research use; review their terms before any production election use.

For native Render deployment, the repository-root `render.yaml` defines this service and the Node backend. The backend's `FACE_SERVICE_API_KEY` must match this service's key, and its `FACE_SERVICE_URL` must point to this service's Render URL.
