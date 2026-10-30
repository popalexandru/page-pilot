"""Build, save, and load a small persistent semantic index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def build_semantic_index(
    chunks: list[dict[str, int | str]],
    model: SentenceTransformer,
    model_name: str,
) -> dict[str, object]:
    """Create normalized embeddings and metadata for all chunks."""
    if not chunks:
        raise ValueError("cannot build an index without chunks")

    texts = [str(chunk["text"]) for chunk in chunks]
    embeddings = np.asarray(
        model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("embedding model returned an unexpected shape")

    return {
        "embeddings": embeddings,
        "page_numbers": np.array(
            [int(chunk["page_number"]) for chunk in chunks], dtype=np.int32
        ),
        "chunk_indexes": np.array(
            [int(chunk["chunk_index"]) for chunk in chunks], dtype=np.int32
        ),
        "texts": np.array(texts, dtype=np.str_),
        "model_name": model_name,
    }


def save_semantic_index(index: dict[str, object], output_path: Path) -> None:
    """Save embeddings and metadata in a compressed NumPy archive."""
    with output_path.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            embeddings=index["embeddings"],
            page_numbers=index["page_numbers"],
            chunk_indexes=index["chunk_indexes"],
            texts=index["texts"],
            model_name=np.array(index["model_name"], dtype=np.str_),
        )


def load_semantic_index(index_path: Path) -> dict[str, object]:
    """Load and validate a semantic index without allowing pickled objects."""
    with np.load(index_path, allow_pickle=False) as saved_index:
        required_fields = {
            "embeddings",
            "page_numbers",
            "chunk_indexes",
            "texts",
            "model_name",
        }
        if not required_fields.issubset(saved_index.files):
            raise ValueError("semantic index is missing required data")

        embeddings = saved_index["embeddings"].copy()
        page_numbers = saved_index["page_numbers"].copy()
        chunk_indexes = saved_index["chunk_indexes"].copy()
        texts = saved_index["texts"].astype(str)
        model_name_array = saved_index["model_name"]

    if embeddings.ndim != 2:
        raise ValueError("semantic index embeddings must be a matrix")

    item_count = embeddings.shape[0]
    if not (
        len(page_numbers) == item_count
        and len(chunk_indexes) == item_count
        and len(texts) == item_count
    ):
        raise ValueError("semantic index metadata length does not match embeddings")
    if model_name_array.size != 1:
        raise ValueError("semantic index must contain one model name")

    return {
        "embeddings": embeddings,
        "page_numbers": page_numbers,
        "chunk_indexes": chunk_indexes,
        "texts": texts,
        "model_name": str(model_name_array.item()),
    }
