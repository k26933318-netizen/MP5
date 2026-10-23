import base64
import binascii
import logging
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    from .face_utils import (
        detect_face,
        embedding_from_face,
        get_face,
        has_liveness_movement,
        is_good_quality,
        quality_error_message,
        warm_up,
    )
    from .vector_store import VectorStore
except ImportError:
    from face_utils import (
        detect_face,
        embedding_from_face,
        get_face,
        has_liveness_movement,
        is_good_quality,
        quality_error_message,
        warm_up,
    )
    from vector_store import VectorStore

MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.5"))
MATCH_MARGIN = float(os.getenv("MATCH_MARGIN", "0.0"))
# Tune this against representative enrollment captures before using outlier rejection in production.
OUTLIER_MIN_SIM = float(os.getenv("OUTLIER_MIN_SIM", "0.4"))
LIVENESS_IDENTITY_MIN = float(os.getenv("LIVENESS_IDENTITY_MIN", "0.5"))
MAX_REGISTER_IMAGES = min(5, max(1, int(os.getenv("MAX_REGISTER_IMAGES", "5"))))
MAX_VERIFY_IMAGES = min(2, max(1, int(os.getenv("MAX_VERIFY_IMAGES", "2"))))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.logger.setLevel(logging.INFO)
DATABASE_DIR = Path(__file__).resolve().parent.parent / "database"
vector_store = None
_runtime_lock = threading.Lock()


@app.before_request
def initialize_legacy_runtime():
    global vector_store
    if vector_store is not None:
        return
    with _runtime_lock:
        if vector_store is None:
            vector_store = VectorStore(DATABASE_DIR)
            warm_up()


def _log_stage_timing(stage, started_at):
    app.logger.info(
        "request=%s stage=%s duration_ms=%.2f",
        request.path,
        stage,
        (time.perf_counter() - started_at) * 1000,
    )


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith(("/register", "/verify", "/detect_face")):
        return jsonify(success=False, face_detected=False, message="API route not found."), 404
    return error


@app.errorhandler(405)
def method_not_allowed(error):
    if request.path.startswith(("/register", "/verify", "/detect_face")):
        return jsonify(success=False, face_detected=False, message="HTTP method not allowed."), 405
    return error


@app.errorhandler(500)
def internal_server_error(error):
    if request.path.startswith(("/register", "/verify", "/detect_face")):
        app.logger.exception("Unhandled API error")
        return jsonify(success=False, face_detected=False, message="The server could not process the request."), 500
    return error


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/register")
def register_page():
    return render_template("register.html")


@app.get("/verify")
def verify_page():
    return render_template("verify.html")


@app.post("/register")
def register():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    images = payload.get("images") or ([payload["image"]] if payload.get("image") else [])
    if not name or not images:
        return jsonify(success=False, message="A name and captured images are required."), 400

    try:
        decode_started = time.perf_counter()
        try:
            decoded_images = [decode_image(image) for image in images[:MAX_REGISTER_IMAGES]]
        finally:
            _log_stage_timing("image_decode", decode_started)

        embeddings = []
        face_started = time.perf_counter()
        try:
            for image in decoded_images:
                face = get_face(image)
                if not is_good_quality(image, face):
                    return jsonify(success=False, message=quality_error_message(image, face)), 400
                embeddings.append(embedding_from_face(face))
        finally:
            _log_stage_timing("face_detection_embedding", face_started)

        if len(embeddings) >= 3:
            similarities = np.asarray(embeddings) @ np.asarray(embeddings).T
            mean_similarities = (similarities.sum(axis=1) - np.diag(similarities)) / (len(embeddings) - 1)
            embeddings = [
                embedding
                for embedding, mean_similarity in zip(embeddings, mean_similarities)
                if mean_similarity >= OUTLIER_MIN_SIM
            ]
            if len(embeddings) < 2:
                return jsonify(
                    success=False,
                    message="Registered frames are inconsistent - please try again with only your face in view.",
                ), 400

        for embedding in embeddings:
            vector_store.add(embedding, name)
        vector_store.save()
    except (ValueError, binascii.Error) as error:
        return jsonify(success=False, message=str(error)), 400
    except Exception as error:
        app.logger.exception("Registration failed")
        return jsonify(success=False, message=f"Registration failed: {error}"), 500

    return jsonify(success=True, message=f"Registered: {name}")


@app.post("/detect_face")
def detect_face_route():
    payload = request.get_json(silent=True) or {}
    if not payload.get("image"):
        return jsonify(face_detected=False, message="A captured image is required."), 400

    try:
        decode_started = time.perf_counter()
        try:
            image = decode_image(payload["image"])
        finally:
            _log_stage_timing("image_decode", decode_started)
        detection_started = time.perf_counter()
        try:
            detected = detect_face(image)
        finally:
            _log_stage_timing("face_detection", detection_started)
    except (ValueError, binascii.Error) as error:
        return jsonify(face_detected=False, message=str(error)), 400
    except Exception as error:
        app.logger.exception("Face detection failed")
        return jsonify(face_detected=False, message=f"Face detection failed: {error}"), 500

    return jsonify(
        face_detected=detected,
        message="Face detected." if detected else "No centered face detected.",
    )


@app.post("/verify")
def verify():
    payload = request.get_json(silent=True) or {}
    images = payload.get("images") or ([payload["image"]] if payload.get("image") else [])
    if not images:
        return jsonify(success=False, message="Captured images are required."), 400

    try:
        decode_started = time.perf_counter()
        try:
            decoded_images = [decode_image(image) for image in images[:MAX_VERIFY_IMAGES]]
        finally:
            _log_stage_timing("image_decode", decode_started)

        face_started = time.perf_counter()
        try:
            faces = [get_face(image) for image in decoded_images]
            quality_passes = [
                is_good_quality(image, face)
                for image, face in zip(decoded_images, faces)
            ]
            if not any(quality_passes):
                return jsonify(
                    success=False,
                    message=quality_error_message(decoded_images[0], faces[0]),
                ), 400

            if len(decoded_images) >= 2:
                if not has_liveness_movement(
                    decoded_images[0], decoded_images[1], faces[0], faces[1]
                ):
                    return jsonify(success=False, message="Liveness check failed - please try again"), 400
                first_embedding = embedding_from_face(faces[0])
                second_embedding = embedding_from_face(faces[1])
                if float(np.dot(first_embedding, second_embedding)) < LIVENESS_IDENTITY_MIN:
                    return jsonify(success=False, message="Liveness check failed - please try again"), 400

            embeddings = [
                embedding_from_face(face)
                for face, passes_quality in zip(faces, quality_passes)
                if passes_quality
            ]
            embedding = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                raise ValueError("The face embedding was empty.")
            embedding = embedding / norm
        finally:
            _log_stage_timing("face_detection_embedding", face_started)

        search_started = time.perf_counter()
        try:
            if MATCH_MARGIN > 0:
                name, score, runner_up_score = vector_store.search_with_runner_up(embedding)
                margin_passes = runner_up_score is None or score - runner_up_score >= MATCH_MARGIN
            else:
                name, score = vector_store.search(embedding)
                margin_passes = True
        finally:
            _log_stage_timing("vector_search", search_started)
    except (ValueError, binascii.Error) as error:
        return jsonify(success=False, message=str(error)), 400
    except Exception as error:
        app.logger.exception("Verification failed")
        return jsonify(success=False, message=f"Verification failed: {error}"), 500

    if score > MATCH_THRESHOLD and margin_passes:
        return jsonify(success=True, match=name, score=score, message=f"Match: {name} (score: {score:.2f})")
    return jsonify(success=True, match=None, message="No match found")


def decode_image(data_url):
    """Decode a browser data URL into an OpenCV BGR image."""
    if not isinstance(data_url, str):
        raise ValueError("Image must be a base64 string.")
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("The captured image is not valid base64 data.") from error
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The captured image could not be decoded.")
    return image


if __name__ == "__main__":
    app.run(debug=FLASK_DEBUG)
