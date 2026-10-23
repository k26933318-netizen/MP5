import unittest

from app.utils.similarity import cosine_similarity
from app.services.verification_service import verify_embedding


class SimilarityTests(unittest.TestCase):
    def test_normalized_equal_embeddings_have_similarity_one(self):
        self.assertAlmostEqual(cosine_similarity([0.6, 0.8], [0.6, 0.8]), 1.0)

    def test_orthogonal_embeddings_have_similarity_zero(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0])

    def test_rejects_zero_vectors(self):
        with self.assertRaises(ValueError):
            cosine_similarity([0.0], [1.0])

    def test_verifies_a_match_and_rejects_a_different_embedding(self):
        self.assertTrue(verify_embedding([1.0, 0.0], [1.0, 0.0], 0.5))
        self.assertFalse(verify_embedding([1.0, 0.0], [0.0, 1.0], 0.5))


if __name__ == "__main__":
    unittest.main()
