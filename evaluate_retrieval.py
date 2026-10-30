"""Evaluate whether hybrid retrieval finds the expected manual pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hybrid_search import hybrid_search
from semantic_index import load_semantic_index
from semantic_search import load_embedding_model


def load_evaluation_cases(path: Path) -> list[dict[str, object]]:
    """Load and validate retrieval evaluation cases from JSON."""
    with path.open(encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, list) or not data:
        raise ValueError("evaluation file must contain a non-empty JSON list")

    cases = []
    for case_index, case in enumerate(data, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"evaluation case {case_index} must be an object")

        question = case.get("question")
        expected_pages = case.get("expected_pages")
        relevant_chunks = case.get("relevant_chunks")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"evaluation case {case_index} has an invalid question")
        if (
            not isinstance(expected_pages, list)
            or not expected_pages
            or not all(isinstance(page, int) and page > 0 for page in expected_pages)
        ):
            raise ValueError(
                f"evaluation case {case_index} has invalid expected_pages"
            )
        if not isinstance(relevant_chunks, list) or not relevant_chunks:
            raise ValueError(
                f"evaluation case {case_index} has invalid relevant_chunks"
            )

        validated_chunks = []
        for chunk in relevant_chunks:
            if (
                not isinstance(chunk, dict)
                or not isinstance(chunk.get("page_number"), int)
                or chunk["page_number"] <= 0
                or not isinstance(chunk.get("chunk_index"), int)
                or chunk["chunk_index"] < 0
            ):
                raise ValueError(
                    f"evaluation case {case_index} has an invalid relevant chunk"
                )
            validated_chunks.append(
                {
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                }
            )

        cases.append(
            {
                "question": question,
                "expected_pages": expected_pages,
                "relevant_chunks": validated_chunks,
            }
        )

    return cases


def evaluate_cases(
    cases: list[dict[str, object]],
    index: dict[str, object],
    embedding_model: object,
    top_k: int,
    semantic_weight: float,
) -> list[dict[str, object]]:
    """Run every question and record whether an expected page was retrieved."""
    evaluations = []

    for case in cases:
        results = hybrid_search(
            index,
            str(case["question"]),
            embedding_model,
            top_k=top_k,
            semantic_weight=semantic_weight,
        )
        retrieved_pages = [int(result["page_number"]) for result in results]
        expected_pages = [int(page) for page in case["expected_pages"]]
        retrieved_chunks = [
            (int(result["page_number"]), int(result["chunk_index"]))
            for result in results
        ]
        relevant_chunks = {
            (int(chunk["page_number"]), int(chunk["chunk_index"]))
            for chunk in case["relevant_chunks"]
        }
        relevant_retrieved = set(retrieved_chunks) & relevant_chunks
        recall = len(relevant_retrieved) / len(relevant_chunks)
        precision = (
            len(relevant_retrieved) / len(retrieved_chunks)
            if retrieved_chunks
            else 0.0
        )
        page_hit = bool(set(retrieved_pages) & set(expected_pages))
        evaluations.append(
            {
                "question": str(case["question"]),
                "expected_pages": expected_pages,
                "retrieved_pages": retrieved_pages,
                "retrieved_chunks": retrieved_chunks,
                "relevant_chunks": sorted(relevant_chunks),
                "page_hit": page_hit,
                "recall": recall,
                "precision": precision,
            }
        )

    return evaluations


def run(
    index_path: Path,
    evaluation_path: Path,
    top_k: int,
    semantic_weight: float,
) -> int:
    """Load resources, evaluate retrieval, and print a compact report."""
    try:
        cases = load_evaluation_cases(evaluation_path)
        index = load_semantic_index(index_path)
        embedding_model = load_embedding_model(str(index["model_name"]))
        evaluations = evaluate_cases(
            cases,
            index,
            embedding_model,
            top_k,
            semantic_weight,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        print(f"Error: evaluation failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: embedding model failed: {error}", file=sys.stderr)
        return 1

    hit_count = sum(bool(item["page_hit"]) for item in evaluations)
    for item in evaluations:
        status = "PASS" if item["page_hit"] else "FAIL"
        expected = ", ".join(map(str, item["expected_pages"]))
        retrieved = ", ".join(map(str, item["retrieved_pages"]))
        print(f"[{status}] {item['question']}")
        print(f"  Expected pages: {expected}")
        print(f"  Retrieved pages: {retrieved}")
        print(f"  Chunk recall@{top_k}: {item['recall']:.1%}")
        print(f"  Chunk precision@{top_k}: {item['precision']:.1%}")

    hit_rate = hit_count / len(evaluations)
    average_recall = sum(float(item["recall"]) for item in evaluations) / len(
        evaluations
    )
    average_precision = sum(
        float(item["precision"]) for item in evaluations
    ) / len(evaluations)
    print(
        f"\nPage hit rate@{top_k}: "
        f"{hit_count}/{len(evaluations)} ({hit_rate:.1%})"
    )
    print(f"Mean chunk recall@{top_k}: {average_recall:.1%}")
    print(f"Mean chunk precision@{top_k}: {average_precision:.1%}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure whether retrieval finds expected manual pages."
    )
    parser.add_argument("index_path", type=Path, help="Path to semantic index NPZ")
    parser.add_argument(
        "evaluation_path",
        type=Path,
        help="Path to evaluation questions JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks per question (default: 3)",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.7,
        help="Semantic share of the hybrid score (default: 0.7)",
    )
    args = parser.parse_args()
    return run(
        args.index_path,
        args.evaluation_path,
        args.top_k,
        args.semantic_weight,
    )


if __name__ == "__main__":
    raise SystemExit(main())
