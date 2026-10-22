import numpy as np
import pytest

from legacy.vector_store import VectorStore


def _normalized_random_vectors(seed, count, dimension):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dimension)).astype("float32")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_search_matches_brute_force_argmax(tmp_path):
    vectors = _normalized_random_vectors(11, 20, 16)
    names = [f"person-{index // 4}" for index in range(len(vectors))]
    query = _normalized_random_vectors(12, 1, 16)[0]
    store = VectorStore(tmp_path)
    for vector, name in zip(vectors, names):
        store.add(vector, name)

    expected_position = int(np.argmax(vectors @ query))
    name, score = store.search(query)

    assert name == names[expected_position]
    assert score == pytest.approx(float((vectors @ query).max()))


def test_save_load_round_trip(tmp_path):
    vectors = _normalized_random_vectors(21, 5, 8)
    store = VectorStore(tmp_path)
    for index, vector in enumerate(vectors):
        store.add(vector, f"person-{index}")
    store.save()

    loaded = VectorStore(tmp_path)

    assert loaded.names == store.names
    for vector in vectors:
        assert loaded.search(vector) == pytest.approx(store.search(vector))


def test_search_with_runner_up_uses_a_different_name(tmp_path):
    vectors = _normalized_random_vectors(31, 14, 12)
    query = vectors[0].copy()
    vectors[:11] = query
    names = ["alpha"] * 11 + ["beta"] * 2 + ["gamma"]
    store = VectorStore(tmp_path)
    for vector, name in zip(vectors, names):
        store.add(vector, name)

    best_name, best_score, runner_up_score = store.search_with_runner_up(query)
    scores = vectors @ query
    expected_runner_up = max(
        float(score)
        for score, name in zip(scores, names)
        if name != best_name
    )

    assert best_name == names[int(np.argmax(scores))]
    assert best_score == pytest.approx(float(scores.max()))
    assert runner_up_score == pytest.approx(expected_runner_up)


@pytest.mark.parametrize("present_file", ["faces.index", "names.json"])
def test_load_rejects_incomplete_index_pair(tmp_path, present_file):
    (tmp_path / present_file).touch()

    with pytest.raises(ValueError, match="must both exist"):
        VectorStore(tmp_path)