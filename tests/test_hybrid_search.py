import unittest

import numpy as np

from hybrid_search import hybrid_search


class FakeEmbeddingModel:
    def encode(
        self,
        text: str,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        return np.array([1.0, 0.0])


class HybridSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = {
            "embeddings": np.array([[0.6, 0.0], [0.7, 0.0]]),
            "page_numbers": np.array([1, 2]),
            "chunk_indexes": np.array([0, 0]),
            "texts": np.array(["reset device", "other instructions"]),
            "model_name": "fake-model",
        }

    def test_exact_term_can_raise_a_result_above_semantic_only_result(self) -> None:
        results = hybrid_search(
            self.index,
            "reset",
            FakeEmbeddingModel(),
            semantic_weight=0.7,
        )

        self.assertEqual(results[0]["page_number"], 1)
        self.assertEqual(results[0]["lexical_score"], 1.0)
        self.assertAlmostEqual(results[0]["combined_score"], 0.72)

    def test_semantic_only_weight_matches_semantic_ranking(self) -> None:
        results = hybrid_search(
            self.index,
            "reset",
            FakeEmbeddingModel(),
            semantic_weight=1.0,
        )

        self.assertEqual(results[0]["page_number"], 2)

    def test_rejects_weight_outside_zero_to_one(self) -> None:
        with self.assertRaises(ValueError):
            hybrid_search(
                self.index,
                "reset",
                FakeEmbeddingModel(),
                semantic_weight=1.1,
            )


if __name__ == "__main__":
    unittest.main()
