import math


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embeddings must have the same non-zero dimension")
    if not all(math.isfinite(value) for value in left + right):
        raise ValueError("Embeddings must contain only finite values")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Embeddings must have non-zero magnitude")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
