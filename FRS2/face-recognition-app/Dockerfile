FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    FACE_DATA_DIR=/var/lib/face-data

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 faceverify \
    && mkdir -p /var/lib/face-data \
    && chown -R faceverify:faceverify /var/lib/face-data

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chown -R faceverify:faceverify /app

USER 10001:10001
VOLUME ["/var/lib/face-data"]
EXPOSE 5000
CMD ["python", "serve.py"]