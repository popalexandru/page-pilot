import json
import tempfile
import unittest
from pathlib import Path

from search_chunks import load_chunks, search_chunks, tokenize


class SearchChunksTests(unittest.TestCase):
    def test_tokenize_handles_case_diacritics_and_punctuation(self) -> None:
        self.assertEqual(tokenize("INSTRUCȚIUNI, utilizare!"), ["instrucțiuni", "utilizare"])

    def test_search_ranks_chunk_with_more_query_terms_first(self) -> None:
        chunks = [
            {"page_number": 1, "chunk_index": 0, "text": "Reset the device."},
            {
                "page_number": 2,
                "chunk_index": 0,
                "text": "Reset the device after an error.",
            },
        ]

        results = search_chunks(chunks, "reset error")

        self.assertEqual(results[0]["page_number"], 2)
        self.assertEqual(results[0]["score"], 1.0)
        self.assertEqual(results[1]["score"], 0.5)

    def test_search_returns_only_top_k_matches(self) -> None:
        chunks = [
            {"page_number": page, "chunk_index": 0, "text": "manual"}
            for page in range(1, 5)
        ]

        results = search_chunks(chunks, "manual", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual([result["page_number"] for result in results], [1, 2])

    def test_search_rejects_query_without_words(self) -> None:
        with self.assertRaises(ValueError):
            search_chunks([], "... !!!")

    def test_load_chunks_rejects_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chunks_path = Path(temporary_directory) / "chunks.json"
            chunks_path.write_text(json.dumps([{"text": "incomplete"}]), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_chunks(chunks_path)


if __name__ == "__main__":
    unittest.main()
