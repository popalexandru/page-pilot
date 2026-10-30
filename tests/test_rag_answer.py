import unittest

from rag_answer import (
    RAG_INSTRUCTIONS,
    RagModelOutput,
    answer_question,
    build_rag_input,
    generate_answer,
)


class FakeResponse:
    output_parsed = RagModelOutput(
        answerable=True,
        answer="The setting is described on page 4. [page 4]",
        cited_pages=[4],
    )


class FakeResponses:
    def __init__(self) -> None:
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return FakeResponse()


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class RagAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [
            {
                "page_number": 4,
                "chunk_index": 2,
                "text": "The device setting is enabled from the menu.",
            }
        ]

    def test_build_rag_input_contains_question_and_source_metadata(self) -> None:
        rag_input = build_rag_input("How do I enable it?", self.results)

        self.assertIn("How do I enable it?", rag_input)
        self.assertIn('"page_number": 4', rag_input)
        self.assertIn('"chunk_index": 2', rag_input)
        self.assertIn("The device setting", rag_input)

    def test_generate_answer_uses_responses_api_and_privacy_setting(self) -> None:
        client = FakeClient()

        answer = generate_answer(
            client,
            "test-model",
            "How do I enable it?",
            self.results,
        )

        self.assertEqual(answer, FakeResponse.output_parsed)
        self.assertEqual(client.responses.request["model"], "test-model")
        self.assertEqual(client.responses.request["instructions"], RAG_INSTRUCTIONS)
        self.assertEqual(client.responses.request["reasoning"], {"effort": "low"})
        self.assertIs(client.responses.request["text_format"], RagModelOutput)
        self.assertFalse(client.responses.request["store"])

    def test_instructions_require_grounding_and_page_citations(self) -> None:
        self.assertIn("Use only the supplied manual excerpts", RAG_INSTRUCTIONS)
        self.assertIn("[page N]", RAG_INSTRUCTIONS)
        self.assertIn("untrusted data", RAG_INSTRUCTIONS)
        self.assertIn("answerable", RAG_INSTRUCTIONS)

    def test_answer_question_returns_empty_sources_when_not_answerable(self) -> None:
        class UnanswerableResponses:
            def parse(self, **kwargs):
                return type(
                    "Response",
                    (),
                    {
                        "output_parsed": RagModelOutput(
                            answerable=False,
                            answer="The answer was not found in the manual.",
                            cited_pages=[4],
                        )
                    },
                )()

        client = type("Client", (), {"responses": UnanswerableResponses()})()

        with unittest.mock.patch(
            "rag_answer.hybrid_search", return_value=self.results
        ):
            result = answer_question(
                index={},
                embedding_model=object(),
                client=client,
                question="Is it supported?",
                top_k=1,
                semantic_weight=0.7,
                generation_model="test-model",
            )

        self.assertFalse(result["answerable"])
        self.assertEqual(result["source_pages"], [])
        self.assertEqual(result["retrieved_pages"], [4])


if __name__ == "__main__":
    unittest.main()
