from app.utils.similarity import cosine_similarity


def verify_embedding(
    candidate: list[float],
    enrolled: list[float],
    threshold: float,
) -> bool:
    return cosine_similarity(candidate, enrolled) >= threshold
