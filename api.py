"""Expose the Page Pilot RAG pipeline through a small FastAPI application."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

from rag_answer import DEFAULT_GENERATION_MODEL, answer_question
from semantic_index import load_semantic_index
from semantic_search import load_embedding_model


DEFAULT_INDEX_PATH = Path("semantic_index.npz")
FRONTEND_DIR = Path(__file__).parent / "frontend"

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


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the Page Pilot demo interface."""
    return FileResponse(FRONTEND_DIR / "index.html")


@lru_cache(maxsize=1)
def get_retrieval_resources() -> tuple[dict[str, object], object]:
    """Load the semantic index and embedding model once per API process."""
    index_path = Path(os.environ.get("PAGE_PILOT_INDEX", DEFAULT_INDEX_PATH))
    index = load_semantic_index(index_path)
    embedding_model = load_embedding_model(str(index["model_name"]))
    return index, embedding_model


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the HTTP application is running."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Run retrieval and answer generation for one customer question."""
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
