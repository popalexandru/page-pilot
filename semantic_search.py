"""Search PDF chunks with local multilingual embeddings."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from search_chunks import make_preview
from semantic_index import load_semantic_index


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Use a cached model first, downloading it only when it is missing."""
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except OSError:
        print("Model not found in the local cache; downloading it now...")
        return SentenceTransformer(model_name)


def semantic_search(
    index: dict[str, object],
    query: str,
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[dict[str, object]]:
    """Rank chunks by cosine similarity to the query embedding."""
    if top_k <= 0:
        raise ValueError("top-k must be greater than 0")
    if not query.strip():
        raise ValueError("query must contain text")
    embeddings = np.asarray(index["embeddings"])
    if len(embeddings) == 0:
        return []

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    scores = embeddings @ np.asarray(query_embedding)
    ranked_indexes = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "page_number": int(np.asarray(index["page_numbers"])[item_index]),
            "chunk_index": int(np.asarray(index["chunk_indexes"])[item_index]),
            "text": str(np.asarray(index["texts"])[item_index]),
            "score": float(scores[item_index]),
        }
        for item_index in ranked_indexes
    ]


def run(index_path: Path, query: str, top_k: int) -> int:
    """Load a persistent index and calculate only the query embedding."""
    try:
        load_start = time.perf_counter()
        index = load_semantic_index(index_path)
        load_time = time.perf_counter() - load_start
    except FileNotFoundError:
        print(f"Error: semantic index does not exist: {index_path}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Error: could not load semantic index: {error}", file=sys.stderr)
        return 1

    model_name = str(index["model_name"])
    try:
        print(f"Loading embedding model: {model_name}")
        model = load_embedding_model(model_name)
        search_start = time.perf_counter()
        results = semantic_search(index, query, model, top_k)
        search_time = time.perf_counter() - search_start
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: could not run embedding model: {error}", file=sys.stderr)
        return 1

    if not results:
        print("No chunks available in the semantic index.")
        return 0

    print(f"Index load time: {load_time:.3f} seconds")
    print(f"Query search time: {search_time:.3f} seconds")
    print(f"Results: {len(results)}")
    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. Similarity: {float(result['score']):.3f} | "
            f"Page: {result['page_number']} | Chunk: {result['chunk_index']}"
        )
        print(f"Text: {make_preview(str(result['text']))}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find semantically similar PDF chunks with a local model."
    )
    parser.add_argument("index_path", type=Path, help="Path to semantic index NPZ")
    parser.add_argument("query", help="Semantic search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of results (default: 3)",
    )
    args = parser.parse_args()
    return run(args.index_path, args.query, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
