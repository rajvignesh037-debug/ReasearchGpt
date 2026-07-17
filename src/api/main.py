"""
Phase 6: FastAPI Layer
-------------------------
Exposes the RAGPipeline over HTTP. This is intentionally thin - all it does
is validate requests, call the pipeline, and shape responses. No PDF logic,
no ChromaDB calls, no OpenAI calls happen in this file directly.

Uploads are handled synchronously: the request blocks until ingestion
(extract -> embed -> store) finishes, then returns the result. Simple for
now; swap for background jobs + polling later if uploads get large/frequent.

Run with: uvicorn main:app --reload
"""

import shutil
import tempfile
from pathlib import Path
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Ensure the project `src` directory is on sys.path so sibling packages (core) can be imported
# This allows running `uvicorn main:app` from inside `src/api` during local development.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.rag_pipeline import RAGPipeline
from api.schemas import (
    CitationResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(title="Research Paper RAG API", version="0.1.0")

# Single shared pipeline instance for the life of the server.
# ChromaDB's PersistentClient is safe to share across requests.
pipeline = RAGPipeline()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """Catch-all so clients always get consistent JSON errors, never a raw stack trace."""
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health():
    """Report service status and how many chunks are currently stored."""
    return HealthResponse(status="ok", chunks_stored=pipeline.document_count())


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Persist an uploaded PDF to data/ and ingest any new PDFs from that folder."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    destination_path = data_dir / Path(file.filename).name

    try:
        with destination_path.open("wb") as destination_file:
            shutil.copyfileobj(file.file, destination_file)

        summary = pipeline.ingest_folder("data")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return JSONResponse(
        status_code=200,
        content={
            "source": file.filename,
            "summary": summary,
        },
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a question against the ingested document(s) and get a cited answer."""
    if pipeline.document_count() == 0:
        raise HTTPException(status_code=400, detail="No documents have been ingested yet. Upload one first.")

    result = pipeline.query(
        question=request.question,
        top_k=request.top_k,
        source_filter=request.source_filter,
    )

    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationResponse(marker=c.marker, source=c.source, page=c.page)
            for c in result.citations
        ],
    )