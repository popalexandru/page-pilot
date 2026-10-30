import tempfile
import unittest
from pathlib import Path

import numpy as np

from semantic_search import semantic_search
from semantic_index import build_semantic_index, load_semantic_index, save_semantic_index


class FakeEmbeddingModel:
    """Return predictable vectors without downloading a real model."""

    embeddings = {"mașină": np.array([0.9, 0.1])}

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        return self.embeddings[str(texts)]


class FakeIndexModel:
    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class SemanticSearchTests(unittest.TestCase):
    def test_semantic_search_can_rank_text_without_exact_query_word(self) -> None:
        index = {
            "embeddings": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "page_numbers": np.array([1, 2]),
            "chunk_indexes": np.array([0, 0]),
            "texts": np.array(["automobil pe șosea", "instrucțiuni aparat"]),
            "model_name": "fake-model",
        }

        results = semantic_search(index, "mașină", FakeEmbeddingModel(), top_k=2)

        self.assertEqual(results[0]["page_number"], 1)
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_semantic_search_rejects_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            semantic_search({}, "   ", FakeEmbeddingModel())

    def test_semantic_search_rejects_invalid_top_k(self) -> None:
        with self.assertRaises(ValueError):
            semantic_search({}, "query", FakeEmbeddingModel(), top_k=0)

    def test_index_round_trip_preserves_embeddings_and_metadata(self) -> None:
        chunks = [
            {"page_number": 1, "chunk_index": 0, "text": "first"},
            {"page_number": 2, "chunk_index": 1, "text": "second"},
        ]
        index = build_semantic_index(chunks, FakeIndexModel(), "fake-model")

        with tempfile.TemporaryDirectory() as temporary_directory:
            index_path = Path(temporary_directory) / "index.npz"
            save_semantic_index(index, index_path)
            loaded_index = load_semantic_index(index_path)

        np.testing.assert_array_equal(loaded_index["embeddings"], index["embeddings"])
        np.testing.assert_array_equal(loaded_index["texts"], index["texts"])
        self.assertEqual(loaded_index["model_name"], "fake-model")


if __name__ == "__main__":
    unittest.main()
