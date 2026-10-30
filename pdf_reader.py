"""Extract and chunk page text from a PDF manual."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError


def make_preview(text: str, limit: int = 50) -> str:
    """Collapse whitespace and keep only a short preview."""
    preview = " ".join(text.split())
    if len(preview) > limit:
        return preview[: limit - 3] + "..."
    return preview


def extract_pages(pdf_path: Path) -> list[dict[str, int | str]]:
    """Return the page number and extracted text for every PDF page."""
    reader = PdfReader(pdf_path)
    pages = []

    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1
        try:
            text = page.extract_text() or ""
        except Exception as error:  # A damaged page should not hide other pages.
            print(
                f"Warning: text extraction failed on page {page_number}: {error}",
                file=sys.stderr,
            )
            text = ""

        pages.append({"page_number": page_number, "text": text})

    return pages


def chunk_pages(
    pages: list[dict[str, int | str]],
    chunk_size: int = 120,
    overlap: int = 20,
) -> list[dict[str, int | str]]:
    """Split page text into overlapping, word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk size must be greater than 0")
    if overlap < 0:
        raise ValueError("chunk overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        words = str(page["text"]).split()
        start = 0
        chunk_index = 0

        while start < len(words):
            chunk_words = words[start : start + chunk_size]
            chunks.append(
                {
                    "page_number": int(page["page_number"]),
                    "chunk_index": chunk_index,
                    "text": " ".join(chunk_words),
                }
            )

            if start + chunk_size >= len(words):
                break

            start += step
            chunk_index += 1

    return chunks


def save_json(records: list[dict[str, int | str]], output_path: Path) -> None:
    """Save records as readable UTF-8 JSON."""
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def run(
    pdf_path: Path,
    output_path: Path | None = None,
    chunks_output_path: Path | None = None,
    chunk_size: int = 120,
    chunk_overlap: int = 20,
) -> int:
    """Extract a PDF, create chunks, and optionally write JSON files."""
    try:
        pages = extract_pages(pdf_path)
    except FileNotFoundError:
        print(f"Error: PDF file does not exist: {pdf_path}", file=sys.stderr)
        return 1
    except (OSError, PyPdfError) as error:
        print(f"Error: could not read PDF '{pdf_path}': {error}", file=sys.stderr)
        return 1

    print(f"Pages: {len(pages)}")
    for page in pages:
        preview = make_preview(str(page["text"]))
        if preview:
            print(f"Page {page['page_number']}: {preview}")
        else:
            print(f"Page {page['page_number']}: [no extractable text]")

    try:
        chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Chunks: {len(chunks)}")

    if output_path is not None:
        try:
            save_json(pages, output_path)
        except OSError as error:
            print(f"Error: could not write JSON '{output_path}': {error}", file=sys.stderr)
            return 1
        print(f"Saved pages JSON: {output_path}")

    if chunks_output_path is not None:
        try:
            save_json(chunks, chunks_output_path)
        except OSError as error:
            print(
                f"Error: could not write JSON '{chunks_output_path}': {error}",
                file=sys.stderr,
            )
            return 1
        print(f"Saved chunks JSON: {chunks_output_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show a short text preview for every page in a PDF."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF manual")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the extracted page data in JSON format",
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        help="Optional path for the chunk data in JSON format",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=120,
        help="Maximum words per chunk (default: 120)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="Words repeated between consecutive chunks (default: 20)",
    )
    args = parser.parse_args()
    return run(
        args.pdf_path,
        args.output,
        args.chunks_output,
        args.chunk_size,
        args.chunk_overlap,
    )


if __name__ == "__main__":
    raise SystemExit(main())
