import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import ManualStatus, app, process_uploaded_manual


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_home_serves_demo_interface(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Page Pilot", response.text)
        self.assertIn("Ask the manual", response.text)

    def test_manual_status_reports_active_manual(self) -> None:
        status = ManualStatus(
            ready=True,
            filename="product.pdf",
            page_count=12,
            chunk_count=34,
        )

        with patch("api.read_manual_status", return_value=status):
            response = self.client.get("/manual")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), status.model_dump())

    def test_upload_manual_processes_a_pdf(self) -> None:
        status = ManualStatus(
            ready=True,
            filename="product.pdf",
            page_count=2,
            chunk_count=5,
        )

        with patch("api.process_uploaded_manual", return_value=status) as process_mock:
            response = self.client.post(
                "/manual",
                files={"file": ("product.pdf", b"%PDF-1.7 fake", "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), status.model_dump())
        self.assertEqual(process_mock.call_args.args, (b"%PDF-1.7 fake", "product.pdf"))

    def test_upload_manual_rejects_non_pdf_content(self) -> None:
        response = self.client.post(
            "/manual",
            files={"file": ("product.pdf", b"not a pdf", "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("not a PDF", response.json()["detail"])

    def test_process_uploaded_manual_writes_active_artifacts(self) -> None:
        pages = [{"page_number": 1, "text": "Useful manual text"}]
        chunks = [
            {"page_number": 1, "chunk_index": 0, "text": "Useful manual text"}
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            paths = {
                "ACTIVE_PDF_PATH": data_dir / "active_manual.pdf",
                "ACTIVE_PAGES_PATH": data_dir / "pages.json",
                "ACTIVE_CHUNKS_PATH": data_dir / "chunks.json",
                "ACTIVE_INDEX_PATH": data_dir / "semantic_index.npz",
                "MANUAL_METADATA_PATH": data_dir / "manual.json",
            }

            with (
                patch("api.DATA_DIR", data_dir),
                patch.multiple("api", **paths),
                patch("api.extract_pages", return_value=pages),
                patch("api.chunk_pages", return_value=chunks),
                patch("api.get_embedding_model", return_value=object()),
                patch("api.build_semantic_index", return_value={"fake": "index"}),
                patch(
                    "api.save_semantic_index",
                    side_effect=lambda _index, path: path.write_bytes(b"index"),
                ),
            ):
                status = process_uploaded_manual(b"%PDF-1.7 fake", "product.pdf")

            self.assertTrue(status.ready)
            self.assertEqual(status.page_count, 1)
            self.assertEqual(status.chunk_count, 1)
            self.assertEqual(paths["ACTIVE_PDF_PATH"].read_bytes(), b"%PDF-1.7 fake")
            self.assertTrue(paths["ACTIVE_INDEX_PATH"].is_file())

    def test_ask_returns_answer_and_source_pages(self) -> None:
        fake_result = {
            "answer": "The manual describes this on page 3. [page 3]",
            "answerable": True,
            "source_pages": [3],
            "retrieved_pages": [1, 3],
        }

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch(
                "api.read_manual_status",
                return_value=ManualStatus(ready=True),
            ),
            patch("api.get_retrieval_resources", return_value=({}, object())),
            patch("api.answer_question", return_value=fake_result) as answer_mock,
        ):
            response = self.client.post(
                "/ask",
                json={"question": "How do I check it?", "top_k": 3},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), fake_result)
        self.assertEqual(answer_mock.call_args.kwargs["question"], "How do I check it?")
        self.assertEqual(answer_mock.call_args.kwargs["top_k"], 3)

    def test_ask_rejects_empty_question(self) -> None:
        response = self.client.post("/ask", json={"question": "", "top_k": 3})

        self.assertEqual(response.status_code, 422)

    def test_ask_reports_missing_api_key(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "api.read_manual_status",
                return_value=ManualStatus(ready=True),
            ),
        ):
            response = self.client.post(
                "/ask",
                json={"question": "How do I check it?"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENAI_API_KEY", response.json()["detail"])

    def test_ask_requires_an_active_manual(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch(
                "api.read_manual_status",
                return_value=ManualStatus(ready=False),
            ),
        ):
            response = self.client.post(
                "/ask",
                json={"question": "How do I check it?"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("Upload", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
