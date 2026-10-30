"""Search extracted PDF chunks using exact word overlap."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """Return lowercase Unicode words without punctuation."""
    return re.findall(r"\w+", text.casefold())


def load_chunks(chunks_path: Path) -> list[dict[str, int | str]]:
    """Load and validate chunk records from a JSON file."""
    with chunks_path.open(encoding="utf-8") as chunks_file:
        chunks = json.load(chunks_file)

    if not isinstance(chunks, list):
        raise ValueError("chunks JSON must contain a list")

    required_fields = {"page_number", "chunk_index", "text"}
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not required_fields.issubset(chunk):
            raise ValueError(f"invalid chunk record at list index {index}")
        if not isinstance(chunk["page_number"], int):
            raise ValueError(f"invalid page_number at list index {index}")
        if not isinstance(chunk["chunk_index"], int):
            raise ValueError(f"invalid chunk_index at list index {index}")
        if not isinstance(chunk["text"], str):
            raise ValueError(f"invalid text at list index {index}")

    return chunks


def search_chunks(
    chunks: list[dict[str, int | str]], query: str, top_k: int = 3
) -> list[dict[str, object]]:
    """Rank chunks by the proportion of distinct query terms they contain."""
    if top_k <= 0:
        raise ValueError("top-k must be greater than 0")

    query_terms = set(tokenize(query))
    if not query_terms:
        raise ValueError("query must contain at least one word")

    results = []
    for chunk in chunks:
        chunk_terms = set(tokenize(str(chunk["text"])))
        matched_terms = sorted(query_terms & chunk_terms)
        if not matched_terms:
            continue

        results.append(
            {
                **chunk,
                "score": len(matched_terms) / len(query_terms),
                "matched_terms": matched_terms,
            }
        )

    results.sort(
        key=lambda result: (
            -float(result["score"]),
            int(result["page_number"]),
            int(result["chunk_index"]),
        )
    )
    return results[:top_k]


def make_preview(text: str, limit: int = 240) -> str:
    """Return a single-line preview for a search result."""
    preview = " ".join(text.split())
    if len(preview) > limit:
        return preview[: limit - 3] + "..."
    return preview


def run(chunks_path: Path, query: str, top_k: int) -> int:
    """Load chunks, run lexical search, and print ranked results."""
    try:
        chunks = load_chunks(chunks_path)
        results = search_chunks(chunks, query, top_k)
    except FileNotFoundError:
        print(f"Error: chunks file does not exist: {chunks_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as error:
        print(f"Error: invalid JSON in '{chunks_path}': {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if not results:
        print("No matching chunks found.")
        return 0

    print(f"Results: {len(results)}")
    for rank, result in enumerate(results, start=1):
        matched_terms = ", ".join(str(term) for term in result["matched_terms"])
        print(
            f"\n{rank}. Score: {float(result['score']):.2f} | "
            f"Page: {result['page_number']} | Chunk: {result['chunk_index']}"
        )
        print(f"Matched terms: {matched_terms}")
        print(f"Text: {make_preview(str(result['text']))}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find PDF chunks that contain words from a query."
    )
    parser.add_argument("chunks_path", type=Path, help="Path to chunks JSON")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of results (default: 3)",
    )
    args = parser.parse_args()
    return run(args.chunks_path, args.query, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
