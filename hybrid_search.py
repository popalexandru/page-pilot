"""Combine lexical and semantic scores for PDF chunk retrieval."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from search_chunks import make_preview, tokenize
from semantic_index import load_semantic_index
from semantic_search import load_embedding_model, semantic_search


def hybrid_search(
    index: dict[str, object],
    query: str,
    model: SentenceTransformer,
    top_k: int = 3,
    semantic_weight: float = 0.7,
) -> list[dict[str, object]]:
    """Rank chunks using weighted lexical and semantic scores."""
    if top_k <= 0:
        raise ValueError("top-k must be greater than 0")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic weight must be between 0 and 1")

    query_terms = set(tokenize(query))
    if not query_terms:
        raise ValueError("query must contain at least one word")

    item_count = len(np.asarray(index["embeddings"]))
    if item_count == 0:
        return []

    semantic_results = semantic_search(index, query, model, top_k=item_count)
    lexical_weight = 1.0 - semantic_weight
    results = []

    for result in semantic_results:
        chunk_terms = set(tokenize(str(result["text"])))
        matched_terms = sorted(query_terms & chunk_terms)
        lexical_score = len(matched_terms) / len(query_terms)
        semantic_score = max(0.0, float(result["score"]))
        combined_score = (
            lexical_weight * lexical_score + semantic_weight * semantic_score
        )

        results.append(
            {
                **result,
                "lexical_score": lexical_score,
                "semantic_score": semantic_score,
                "combined_score": combined_score,
                "matched_terms": matched_terms,
            }
        )

    results.sort(
        key=lambda result: (
            -float(result["combined_score"]),
            int(result["page_number"]),
            int(result["chunk_index"]),
        )
    )
    return results[:top_k]


def run(
    index_path: Path,
    query: str,
    top_k: int,
    semantic_weight: float,
) -> int:
    """Load the index and print hybrid search results."""
    try:
        index = load_semantic_index(index_path)
    except FileNotFoundError:
        print(f"Error: semantic index does not exist: {index_path}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Error: could not load semantic index: {error}", file=sys.stderr)
        return 1

    try:
        model_name = str(index["model_name"])
        print(f"Loading embedding model: {model_name}")
        model = load_embedding_model(model_name)
        start_time = time.perf_counter()
        results = hybrid_search(index, query, model, top_k, semantic_weight)
        search_time = time.perf_counter() - start_time
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: could not run embedding model: {error}", file=sys.stderr)
        return 1

    if not results:
        print("No chunks available in the semantic index.")
        return 0

    lexical_weight = 1.0 - semantic_weight
    print(
        f"Weights: lexical={lexical_weight:.2f}, semantic={semantic_weight:.2f}"
    )
    print(f"Hybrid search time: {search_time:.3f} seconds")
    print(f"Results: {len(results)}")

    for rank, result in enumerate(results, start=1):
        matched_terms = ", ".join(str(term) for term in result["matched_terms"])
        print(
            f"\n{rank}. Combined: {float(result['combined_score']):.3f} | "
            f"Lexical: {float(result['lexical_score']):.3f} | "
            f"Semantic: {float(result['semantic_score']):.3f} | "
            f"Page: {result['page_number']} | Chunk: {result['chunk_index']}"
        )
        print(f"Matched terms: {matched_terms or '[none]'}")
        print(f"Text: {make_preview(str(result['text']))}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine lexical and semantic PDF chunk search."
    )
    parser.add_argument("index_path", type=Path, help="Path to semantic index NPZ")
    parser.add_argument("query", help="Hybrid search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of results (default: 3)",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.7,
        help="Semantic share of the combined score, from 0 to 1 (default: 0.7)",
    )
    args = parser.parse_args()
    return run(args.index_path, args.query, args.top_k, args.semantic_weight)


if __name__ == "__main__":
    raise SystemExit(main())
