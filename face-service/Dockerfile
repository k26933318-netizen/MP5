FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACE_ANTISPOOF_MODEL_A_PATH=/service/models/minifasnet_a.onnx \
    FACE_ANTISPOOF_MODEL_B_PATH=/service/models/minifasnet_b.onnx

WORKDIR /service
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt
COPY app ./app
COPY models ./models

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
