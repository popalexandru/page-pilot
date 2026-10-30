import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app


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

    def test_ask_returns_answer_and_source_pages(self) -> None:
        fake_result = {
            "answer": "The manual describes this on page 3. [page 3]",
            "answerable": True,
            "source_pages": [3],
            "retrieved_pages": [1, 3],
        }

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
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
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post(
                "/ask",
                json={"question": "How do I check it?"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENAI_API_KEY", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
