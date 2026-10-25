from contextlib import asynccontextmanager
import hmac
import json
import logging
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas.face_schema import DetectionResponse, EmbeddingResponse, VerificationResponse
from app.services.antispoof import anti_spoof
from app.services.detection_service import create_runtime
from app.services.embedding_service import create_embedding
from app.services.verification_service import verify_embedding
from app.utils.image_utils import decode_image


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("face-recognition")
runtime = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global runtime
    if not settings.service_api_key or settings.service_api_key.startswith("replace-with-") or len(settings.service_api_key) < 32:
        raise RuntimeError("FACE_SERVICE_API_KEY must be configured with a strong non-placeholder secret")
    logger.info("Loading InsightFace detection and recognition models")
    if not settings.require_antispoof:
        raise RuntimeError("FACE_REQUIRE_ANTISPOOF must remain enabled for voting workflows")
    anti_spoof.load()
    runtime = create_runtime()
    logger.info("Face models loaded: %s", runtime.model_version)
    yield
    runtime = None


app = FastAPI(title="VoteLedger Face Recognition", version="1.0.0", lifespan=lifespan)


def authorize(service_key: str) -> None:
    if not settings.service_api_key or not hmac.compare_digest(service_key, settings.service_api_key):
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED",
            "message": "Face service authentication failed.",
        })


async def read_upload(image: UploadFile) -> bytes:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail={
            "code": "UNSUPPORTED_IMAGE_TYPE",
            "message": "Upload a JPEG, PNG, or WebP image.",
        })
    content = await image.read(settings.max_upload_bytes + 1)
    if not content or len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail={
            "code": "IMAGE_TOO_LARGE",
            "message": "The image is empty or exceeds the upload size limit.",
        })
    return content


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {
        "code": "FACE_SERVICE_ERROR",
        "message": str(exc.detail),
    }
    return JSONResponse(status_code=exc.status_code, content={
        "success": False,
        "code": detail["code"],
        "message": detail["message"],
        **({"state": detail["state"]} if "state" in detail else {}),
    })


@app.get("/health")
async def health(x_face_service_key: str = Header()):
    authorize(x_face_service_key)
    if runtime is None:
        return JSONResponse(status_code=503, content={
            "status": "unavailable",
            "service": "face-recognition",
        })
    return {
        "status": "ok",
        "service": "face-recognition",
        "modelVersion": runtime.model_version,
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_face(
    image: Annotated[UploadFile, File()],
    x_face_service_key: Annotated[str, Header()],
):
    authorize(x_face_service_key)
    if runtime is None:
        raise HTTPException(status_code=503, detail={
            "code": "FACE_SERVICE_UNAVAILABLE",
            "message": "Face models are unavailable.",
        })
    content = await read_upload(image)
    decoded = decode_image(content, image.content_type or "")
    runtime.detect_faces(decoded)
    return DetectionResponse(face_count=1, model_version=runtime.model_version)


@app.post("/embedding", response_model=EmbeddingResponse)
async def embedding(
    image: Annotated[UploadFile, File()],
    x_face_service_key: Annotated[str, Header()],
):
    authorize(x_face_service_key)
    if runtime is None:
        raise HTTPException(status_code=503, detail={
            "code": "FACE_SERVICE_UNAVAILABLE",
            "message": "Face models are unavailable.",
        })
    content = await read_upload(image)
    values = create_embedding(runtime, content, image.content_type or "")
    return EmbeddingResponse(
        embedding=values,
        dimension=len(values),
        model_version=runtime.model_version,
    )


@app.post("/verify", response_model=VerificationResponse)
async def verify(
    image: Annotated[UploadFile, File()],
    enrolled_embedding: Annotated[str, Form()],
    x_face_service_key: Annotated[str, Header()],
):
    authorize(x_face_service_key)
    if runtime is None:
        raise HTTPException(status_code=503, detail={
            "code": "FACE_SERVICE_UNAVAILABLE",
            "message": "Face models are unavailable.",
        })
    try:
        reference = json.loads(enrolled_embedding)
        if not isinstance(reference, list) or len(reference) != 512:
            raise ValueError("Expected a 512-dimensional enrolled embedding")
        if any(not isinstance(value, (int, float)) for value in reference):
            raise ValueError("Embedding contains invalid values")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={
            "code": "INVALID_EMBEDDING",
            "message": "The enrolled embedding is invalid.",
        }) from exc

    content = await read_upload(image)
    candidate = create_embedding(runtime, content, image.content_type or "")
    return VerificationResponse(
        verified=verify_embedding(candidate, reference, settings.similarity_threshold),
        model_version=runtime.model_version,
    )


@app.post("/assess")
async def assess_position(
    image: Annotated[UploadFile, File()],
    x_face_service_key: Annotated[str, Header()],
    stage: Annotated[str, Form()] = "position",
):
    authorize(x_face_service_key)
    if runtime is None:
        raise HTTPException(status_code=503, detail={
            "code": "FACE_SERVICE_UNAVAILABLE",
            "message": "Face models are unavailable.",
        })
    try:
        content = await read_upload(image)
        decoded = decode_image(content, image.content_type or "")
    except HTTPException:
        if settings.face_diagnostics_enabled:
            logger.info("face_validation %s", json.dumps({
                "face_detected": False,
                "face_count": 0,
                "face_bbox": None,
                "valid_background_pixels": None,
                "near_white_ratio": None,
                "configured_threshold": settings.background_min_white_fraction,
                "uniformity_stddev": None,
                "failure_condition": "invalid_image_input",
            }, sort_keys=True))
        raise
    try:
        if stage not in {"position", "turn", "return"}:
            raise HTTPException(status_code=400, detail={
                "code": "INVALID_ASSESSMENT_STAGE",
                "message": "The face assessment stage is invalid.",
            })
        return runtime.assess_position(decoded, require_frontal=stage != "turn")
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        return JSONResponse(status_code=200, content={
            "ready": False,
            "state": detail.get("state", "red"),
            "message": detail.get("message", "Please reposition and try again."),
            **({"face": detail["face"]} if "face" in detail else {}),
            **({"faceCount": detail["faceCount"]} if "faceCount" in detail else {}),
        })


async def read_frames(images: list[UploadFile]) -> list:
    if not 1 <= len(images) <= 15:
        raise HTTPException(status_code=400, detail={
            "code": "INVALID_FRAME_COUNT",
            "message": "The submitted face sequence is incomplete.",
        })
    decoded = []
    total_size = 0
    for image in images:
        content = await read_upload(image)
        total_size += len(content)
        if total_size > settings.max_upload_bytes * 4:
            raise HTTPException(status_code=413, detail={
                "code": "IMAGE_TOO_LARGE",
                "message": "The captured sequence exceeds the upload limit.",
            })
        decoded.append(decode_image(content, image.content_type or ""))
    return decoded


@app.post("/frames/enroll")
async def enroll_frames(
    images: Annotated[list[UploadFile], File()],
    x_face_service_key: Annotated[str, Header()],
):
    authorize(x_face_service_key)
    if runtime is None:
        raise HTTPException(status_code=503, detail={
            "code": "FACE_SERVICE_UNAVAILABLE",
            "message": "Face models are unavailable.",
        })
    return runtime.analyze_frames(await read_frames(images), "enroll")


@app.post("/frames/verify")
async def verify_frames(
    images: Annotated[list[UploadFile], File()],
    challenge_action: Annotated[str, Form()],
    x_face_service_key: Annotated[str, Header()],
):
    authorize(x_face_service_key)
    if runtime is None:
        raise HTTPException(status_code=503, detail={
            "code": "FACE_SERVICE_UNAVAILABLE",
            "message": "Face models are unavailable.",
        })
    return runtime.analyze_frames(
        await read_frames(images),
        "verify",
        challenge_action,
    )
