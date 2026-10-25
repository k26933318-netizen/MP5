from app.services.detection_service import FaceRecognitionRuntime
from app.utils.image_utils import decode_image


def create_embedding(runtime: FaceRecognitionRuntime, image_bytes: bytes, content_type: str) -> list[float]:
    image = decode_image(image_bytes, content_type)
    return runtime.embedding(image)
