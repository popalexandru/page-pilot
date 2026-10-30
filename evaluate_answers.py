"""Evaluate answerability decisions and cited pages with real RAG calls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI, OpenAIError

from rag_answer import DEFAULT_GENERATION_MODEL, answer_question
from semantic_index import load_semantic_index
from semantic_search import load_embedding_model


def load_answer_cases(path: Path) -> list[dict[str, object]]:
    """Load and validate end-to-end answer evaluation cases."""
    with path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, list) or not data:
        raise ValueError("answer evaluation file must contain a non-empty JSON list")

    cases = []
    for case_index, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"answer case {case_index} must be an object")

        question = case.get("question")
        expected_answerable = case.get("expected_answerable")
        expected_pages = case.get("expected_pages")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"answer case {case_index} has an invalid question")
        if not isinstance(expected_answerable, bool):
            raise ValueError(
                f"answer case {case_index} has invalid expected_answerable"
            )
        if not isinstance(expected_pages, list) or not all(
            isinstance(page, int) and page > 0 for page in expected_pages
        ):
            raise ValueError(f"answer case {case_index} has invalid expected_pages")
        if expected_answerable and not expected_pages:
            raise ValueError(
                f"answer case {case_index} must include expected source pages"
            )
        if not expected_answerable and expected_pages:
            raise ValueError(
                f"unanswerable case {case_index} cannot include source pages"
            )

        cases.append(
            {
                "question": question,
                "expected_answerable": expected_answerable,
                "expected_pages": expected_pages,
            }
        )

    return cases


def score_prediction(
    case: dict[str, object], result: dict[str, object]
) -> dict[str, object]:
    """Compare one structured RAG result with the expected behavior."""
    expected_answerable = bool(case["expected_answerable"])
    actual_answerable = bool(result["answerable"])
    expected_pages = {int(page) for page in case["expected_pages"]}
    actual_pages = {int(page) for page in result["source_pages"]}

    answerability_correct = actual_answerable == expected_answerable
    citation_correct = (
        bool(expected_pages & actual_pages)
        if expected_answerable and actual_answerable
        else not expected_answerable and not actual_pages
    )
    return {
        "question": str(case["question"]),
        "expected_answerable": expected_answerable,
        "actual_answerable": actual_answerable,
        "answerability_correct": answerability_correct,
        "citation_correct": citation_correct,
        "source_pages": sorted(actual_pages),
        "answer": str(result["answer"]),
    }


def run(
    index_path: Path,
    evaluation_path: Path,
    top_k: int,
    semantic_weight: float,
    generation_model: str,
) -> int:
    """Run end-to-end RAG evaluations and print answerability metrics."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    try:
        cases = load_answer_cases(evaluation_path)
        index = load_semantic_index(index_path)
        embedding_model = load_embedding_model(str(index["model_name"]))
        client = OpenAI()
        evaluations = []
        for case in cases:
            result = answer_question(
                index=index,
                embedding_model=embedding_model,
                client=client,
                question=str(case["question"]),
                top_k=top_k,
                semantic_weight=semantic_weight,
                generation_model=generation_model,
            )
            evaluations.append(score_prediction(case, result))
    except (FileNotFoundError, json.JSONDecodeError, OSError, OpenAIError, ValueError) as error:
        print(f"Error: answer evaluation failed: {error}", file=sys.stderr)
        return 1

    for item in evaluations:
        status = "PASS" if item["answerability_correct"] else "FAIL"
        expected = "yes" if item["expected_answerable"] else "no"
        actual = "yes" if item["actual_answerable"] else "no"
        pages = ", ".join(map(str, item["source_pages"])) or "[none]"
        print(f"[{status}] {item['question']}")
        print(f"  Expected answerable: {expected}")
        print(f"  Actual answerable: {actual}")
        print(f"  Source pages: {pages}")
        print(f"  Answer: {item['answer']}")

    answerable_cases = [item for item in evaluations if item["expected_answerable"]]
    unanswerable_cases = [
        item for item in evaluations if not item["expected_answerable"]
    ]
    answerable_accuracy = sum(
        bool(item["actual_answerable"]) for item in answerable_cases
    ) / len(answerable_cases)
    abstention_accuracy = sum(
        not bool(item["actual_answerable"]) for item in unanswerable_cases
    ) / len(unanswerable_cases)
    citation_accuracy = sum(
        bool(item["citation_correct"]) for item in evaluations
    ) / len(evaluations)

    print(f"\nAnswerable accuracy: {answerable_accuracy:.1%}")
    print(f"Abstention accuracy: {abstention_accuracy:.1%}")
    print(f"Citation/empty-source accuracy: {citation_accuracy:.1%}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate answerability and citations with real OpenAI calls."
    )
    parser.add_argument("index_path", type=Path, help="Path to semantic index NPZ")
    parser.add_argument("evaluation_path", type=Path, help="Answer evaluation JSON")
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of retrieved chunks per question (default: 2)",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.7,
        help="Semantic share of the hybrid score (default: 0.7)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GENERATION_MODEL,
        help=f"OpenAI generation model (default: {DEFAULT_GENERATION_MODEL})",
    )
    args = parser.parse_args()
    return run(
        args.index_path,
        args.evaluation_path,
        args.top_k,
        args.semantic_weight,
        args.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
