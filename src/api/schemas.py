"""
Pydantic schemas for the FastAPI layer.

Keeping these separate from main.py so request/response shapes are easy
to find and reuse if a UI or tests need to import them later.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    source: str
    chunks_created: int
    pages_processed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the document(s)")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    source_filter: Optional[str] = Field(default=None, description="Restrict search to one document")


class CitationResponse(BaseModel):
    marker: int
    source: str
    page: int


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationResponse]


class HealthResponse(BaseModel):
    status: str
    chunks_stored: int


class ErrorResponse(BaseModel):
    error: str