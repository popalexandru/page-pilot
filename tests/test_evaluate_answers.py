import json
import tempfile
import unittest
from pathlib import Path

from evaluate_answers import load_answer_cases, score_prediction


class AnswerEvaluationTests(unittest.TestCase):
    def test_score_prediction_accepts_correct_abstention(self) -> None:
        case = {
            "question": "Unknown feature?",
            "expected_answerable": False,
            "expected_pages": [],
        }
        result = {
            "answer": "Not found in the manual.",
            "answerable": False,
            "source_pages": [],
        }

        score = score_prediction(case, result)

        self.assertTrue(score["answerability_correct"])
        self.assertTrue(score["citation_correct"])

    def test_score_prediction_checks_expected_citation(self) -> None:
        case = {
            "question": "Known feature?",
            "expected_answerable": True,
            "expected_pages": [3],
        }
        result = {
            "answer": "It is included. [page 4]",
            "answerable": True,
            "source_pages": [4],
        }

        score = score_prediction(case, result)

        self.assertTrue(score["answerability_correct"])
        self.assertFalse(score["citation_correct"])

    def test_load_answer_cases_rejects_pages_for_unanswerable_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "answers.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "question": "Unknown?",
                            "expected_answerable": False,
                            "expected_pages": [1],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_answer_cases(path)


if __name__ == "__main__":
    unittest.main()
