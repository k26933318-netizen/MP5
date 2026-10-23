from pydantic import BaseModel, Field


class DetectionResponse(BaseModel):
    success: bool = True
    face_count: int = Field(ge=0)
    model_version: str


class EmbeddingResponse(BaseModel):
    success: bool = True
    embedding: list[float]
    dimension: int
    model_version: str


class VerificationResponse(BaseModel):
    success: bool = True
    verified: bool
    model_version: str
