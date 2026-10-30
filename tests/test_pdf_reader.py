import json
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from pdf_reader import chunk_pages, extract_pages, make_preview, save_json


class PdfReaderTests(unittest.TestCase):
    def test_make_preview_keeps_short_text(self) -> None:
        self.assertEqual(make_preview("short text", limit=20), "short text")

    def test_make_preview_truncates_long_text(self) -> None:
        self.assertEqual(make_preview("abcdefghij", limit=7), "abcd...")

    def test_extract_pages_numbers_blank_pages_from_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pdf_path = Path(temporary_directory) / "blank-pages.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            writer.add_blank_page(width=200, height=200)
            writer.write(pdf_path)

            pages = extract_pages(pdf_path)

        self.assertEqual(
            pages,
            [
                {"page_number": 1, "text": ""},
                {"page_number": 2, "text": ""},
            ],
        )

    def test_chunk_pages_adds_overlap_and_metadata(self) -> None:
        pages = [{"page_number": 3, "text": "one two three four five six"}]

        chunks = chunk_pages(pages, chunk_size=4, overlap=2)

        self.assertEqual(
            chunks,
            [
                {
                    "page_number": 3,
                    "chunk_index": 0,
                    "text": "one two three four",
                },
                {
                    "page_number": 3,
                    "chunk_index": 1,
                    "text": "three four five six",
                },
            ],
        )

    def test_chunk_pages_rejects_overlap_equal_to_chunk_size(self) -> None:
        with self.assertRaises(ValueError):
            chunk_pages([], chunk_size=10, overlap=10)

    def test_save_json_writes_utf8(self) -> None:
        pages = [{"page_number": 1, "text": "Instrucțiuni"}]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "pages.json"
            save_json(pages, output_path)
            saved_pages = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved_pages, pages)


if __name__ == "__main__":
    unittest.main()
