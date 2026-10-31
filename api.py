"""Expose the Page Pilot RAG pipeline through a small FastAPI application."""

from __future__ import annotations

import os
import json
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from pypdf.errors import PyPdfError
from starlette.concurrency import run_in_threadpool

from pdf_reader import chunk_pages, extract_pages, save_json
from rag_answer import DEFAULT_GENERATION_MODEL, answer_question
from semantic_index import build_semantic_index, load_semantic_index, save_semantic_index
from semantic_search import DEFAULT_MODEL, load_embedding_model


FRONTEND_DIR = Path(__file__).parent / "frontend"
DATA_DIR = Path(os.environ.get("PAGE_PILOT_DATA_DIR", "data"))
ACTIVE_PDF_PATH = DATA_DIR / "active_manual.pdf"
ACTIVE_PAGES_PATH = DATA_DIR / "pages.json"
ACTIVE_CHUNKS_PATH = DATA_DIR / "chunks.json"
ACTIVE_INDEX_PATH = DATA_DIR / "semantic_index.npz"
MANUAL_METADATA_PATH = DATA_DIR / "manual.json"
MAX_PDF_BYTES = 20 * 1024 * 1024
manual_processing_lock = Lock()

app = FastAPI(
    title="Page Pilot API",
    description="Answer questions using excerpts retrieved from a product manual.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class AskRequest(BaseModel):
    """JSON body accepted by POST /ask."""

    question: str = Field(min_length=1)
    top_k: int = Field(default=2, ge=1, le=10)


class AskResponse(BaseModel):
    """Grounded answer and pages of the retrieved excerpts."""

    answer: str
    answerable: bool
    source_pages: list[int]
    retrieved_pages: list[int]


class ManualStatus(BaseModel):
    """Current locally indexed manual."""

    ready: bool
    filename: str | None = None
    page_count: int = 0
    chunk_count: int = 0


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the Page Pilot demo interface."""
    return FileResponse(FRONTEND_DIR / "index.html")


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str) -> object:
    """Load each embedding model at most once per API process."""
    return load_embedding_model(model_name)


@lru_cache(maxsize=1)
def get_retrieval_resources() -> tuple[dict[str, object], object]:
    """Load the semantic index and embedding model once per API process."""
    index_path = Path(os.environ.get("PAGE_PILOT_INDEX", ACTIVE_INDEX_PATH))
    index = load_semantic_index(index_path)
    embedding_model = get_embedding_model(str(index["model_name"]))
    return index, embedding_model


def read_manual_status() -> ManualStatus:
    """Read metadata only when every active-manual artifact exists."""
    required_paths = (
        ACTIVE_PDF_PATH,
        ACTIVE_PAGES_PATH,
        ACTIVE_CHUNKS_PATH,
        ACTIVE_INDEX_PATH,
        MANUAL_METADATA_PATH,
    )
    if not all(path.is_file() for path in required_paths):
        return ManualStatus(ready=False)

    try:
        metadata = json.loads(MANUAL_METADATA_PATH.read_text(encoding="utf-8"))
        return ManualStatus(
            ready=True,
            filename=str(metadata["filename"]),
            page_count=int(metadata["page_count"]),
            chunk_count=int(metadata["chunk_count"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ManualStatus(ready=False)


def process_uploaded_manual(pdf_bytes: bytes, filename: str) -> ManualStatus:
    """Build active-manual artifacts, replacing old ones only after success."""
    if not manual_processing_lock.acquire(blocking=False):
        raise RuntimeError("another manual is already being processed")

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=DATA_DIR) as temporary_directory:
            temporary_dir = Path(temporary_directory)
            pdf_path = temporary_dir / "manual.pdf"
            pages_path = temporary_dir / "pages.json"
            chunks_path = temporary_dir / "chunks.json"
            index_path = temporary_dir / "semantic_index.npz"
            metadata_path = temporary_dir / "manual.json"

            pdf_path.write_bytes(pdf_bytes)
            pages = extract_pages(pdf_path)
            chunks = chunk_pages(pages)
            if not chunks:
                raise ValueError(
                    "the PDF contains no extractable text; OCR is not supported"
                )

            model = get_embedding_model(DEFAULT_MODEL)
            index = build_semantic_index(chunks, model, DEFAULT_MODEL)

            save_json(pages, pages_path)
            save_json(chunks, chunks_path)
            save_semantic_index(index, index_path)
            metadata_path.write_text(
                json.dumps(
                    {
                        "filename": filename,
                        "page_count": len(pages),
                        "chunk_count": len(chunks),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            for temporary_path, active_path in (
                (pdf_path, ACTIVE_PDF_PATH),
                (pages_path, ACTIVE_PAGES_PATH),
                (chunks_path, ACTIVE_CHUNKS_PATH),
                (index_path, ACTIVE_INDEX_PATH),
                (metadata_path, MANUAL_METADATA_PATH),
            ):
                temporary_path.replace(active_path)

        get_retrieval_resources.cache_clear()
        return ManualStatus(
            ready=True,
            filename=filename,
            page_count=len(pages),
            chunk_count=len(chunks),
        )
    finally:
        manual_processing_lock.release()


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the HTTP application is running."""
    return {"status": "ok"}


@app.get("/manual", response_model=ManualStatus)
def manual_status() -> ManualStatus:
    """Report whether a manual is ready for questions."""
    return read_manual_status()


@app.post("/manual", response_model=ManualStatus)
async def upload_manual(file: UploadFile = File(...)) -> ManualStatus:
    """Validate, extract, chunk, and index one active PDF manual."""
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    pdf_bytes = await file.read(MAX_PDF_BYTES + 1)
    await file.close()
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB limit")
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="File content is not a PDF")

    try:
        return await run_in_threadpool(process_uploaded_manual, pdf_bytes, filename)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (OSError, PyPdfError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process PDF: {error}",
        ) from error


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Run retrieval and answer generation for one customer question."""
    if not read_manual_status().ready and "PAGE_PILOT_INDEX" not in os.environ:
        raise HTTPException(
            status_code=503,
            detail="Upload and index a PDF manual before asking questions",
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set in the server environment",
        )

    try:
        index, embedding_model = get_retrieval_resources()
        result = answer_question(
            index=index,
            embedding_model=embedding_model,
            client=OpenAI(),
            question=request.question,
            top_k=request.top_k,
            semantic_weight=0.7,
            generation_model=DEFAULT_GENERATION_MODEL,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Semantic index not found: {error}",
        ) from error
    except (OSError, OpenAIError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return AskResponse(
        answer=str(result["answer"]),
        answerable=bool(result["answerable"]),
        source_pages=[int(page) for page in result["source_pages"]],
        retrieved_pages=[int(page) for page in result["retrieved_pages"]],
    )
