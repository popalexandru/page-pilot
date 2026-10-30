import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_retrieval import evaluate_cases, load_evaluation_cases


class FakeEmbeddingModel:
    def encode(
        self,
        text: str,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        return np.array([1.0, 0.0])


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = {
            "embeddings": np.array([[0.9, 0.0], [0.2, 0.0]]),
            "page_numbers": np.array([3, 8]),
            "chunk_indexes": np.array([0, 0]),
            "texts": np.array(["blood group test", "unrelated text"]),
            "model_name": "fake-model",
        }

    def test_evaluate_cases_marks_expected_page_as_hit(self) -> None:
        cases = [
            {
                "question": "blood group",
                "expected_pages": [3],
                "relevant_chunks": [{"page_number": 3, "chunk_index": 0}],
            }
        ]

        results = evaluate_cases(
            cases,
            self.index,
            FakeEmbeddingModel(),
            top_k=1,
            semantic_weight=0.7,
        )

        self.assertTrue(results[0]["page_hit"])
        self.assertEqual(results[0]["retrieved_pages"], [3])
        self.assertEqual(results[0]["recall"], 1.0)
        self.assertEqual(results[0]["precision"], 1.0)

    def test_evaluate_cases_marks_missing_page_as_failure(self) -> None:
        cases = [
            {
                "question": "blood group",
                "expected_pages": [8],
                "relevant_chunks": [{"page_number": 8, "chunk_index": 0}],
            }
        ]

        results = evaluate_cases(
            cases,
            self.index,
            FakeEmbeddingModel(),
            top_k=1,
            semantic_weight=0.7,
        )

        self.assertFalse(results[0]["page_hit"])
        self.assertEqual(results[0]["recall"], 0.0)
        self.assertEqual(results[0]["precision"], 0.0)

    def test_load_evaluation_cases_rejects_invalid_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "question": "Question",
                            "expected_pages": [],
                            "relevant_chunks": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_evaluation_cases(path)

    def test_load_evaluation_cases_rejects_invalid_chunk_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "question": "Question",
                            "expected_pages": [1],
                            "relevant_chunks": [
                                {"page_number": 1, "chunk_index": -1}
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_evaluation_cases(path)


if __name__ == "__main__":
    unittest.main()
