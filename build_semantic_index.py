"""Create a persistent semantic index from extracted PDF chunks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from search_chunks import load_chunks
from semantic_index import build_semantic_index, save_semantic_index
from semantic_search import DEFAULT_MODEL, load_embedding_model


def run(chunks_path: Path, output_path: Path, model_name: str) -> int:
    """Load chunks, calculate embeddings once, and save the index."""
    try:
        chunks = load_chunks(chunks_path)
    except FileNotFoundError:
        print(f"Error: chunks file does not exist: {chunks_path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as error:
        print(f"Error: invalid JSON in '{chunks_path}': {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        print(f"Loading embedding model: {model_name}")
        model = load_embedding_model(model_name)
        start_time = time.perf_counter()
        index = build_semantic_index(chunks, model, model_name)
        embedding_time = time.perf_counter() - start_time
        save_semantic_index(index, output_path)
    except (OSError, ValueError) as error:
        print(f"Error: could not build semantic index: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Error: could not run embedding model: {error}", file=sys.stderr)
        return 1

    embeddings = index["embeddings"]
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Embedding time: {embedding_time:.3f} seconds")
    print(f"Saved semantic index: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a persistent semantic index from chunks JSON."
    )
    parser.add_argument("chunks_path", type=Path, help="Path to chunks JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("semantic_index.npz"),
        help="Output index path (default: semantic_index.npz)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence Transformers model (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    return run(args.chunks_path, args.output, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
