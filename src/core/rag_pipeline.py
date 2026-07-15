"""
Phase 5: Pipeline Orchestration
----------------------------------
Wires ingestion -> embeddings -> vector DB -> retrieval -> generation into
a single RAGPipeline class. This is the one interface the API layer (Phase 6)
and later the UI should ever call - they should never touch ChromaDB or
OpenAI directly.
"""

from dataclasses import dataclass
from typing import List, Optional

from core.ingestion import process_pdf
from core.embeddings import embed_chunks
from core.vector_db import VectorDB
from core.retrieval import retrieve, DEFAULT_TOP_K, DEFAULT_MAX_DISTANCE
from core.generation import generate_answer, GeneratedAnswer


@dataclass
class IngestResult:
    """Summary returned after ingesting a document - useful for API responses."""
    source: str
    chunks_created: int
    pages_processed: int


class RAGPipeline:
    """
    The single entry point for the whole RAG system.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest_document("paper.pdf")
        result = pipeline.query("What method did they use?")
    """

    def __init__(self, db_path: str = "./chroma_data"):
        self.db = VectorDB(db_path=db_path)

    def ingest_document(self, pdf_path: str) -> IngestResult:
        """
        Run a PDF through ingestion -> embedding -> storage.

        This is the only method needed to add a new document to the system.
        """
        chunks = process_pdf(pdf_path)
        embedded = embed_chunks(chunks)
        self.db.add_chunks(embedded)

        pages_processed = len(set(c.page for c in chunks))

        return IngestResult(
            source=chunks[0].source if chunks else pdf_path,
            chunks_created=len(chunks),
            pages_processed=pages_processed,
        )

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        source_filter: Optional[str] = None,
    ) -> GeneratedAnswer:
        """
        Run a question through retrieval -> generation.

        This is the only method needed to ask the system a question.
        """
        chunks = retrieve(
            question,
            self.db,
            top_k=top_k,
            max_distance=max_distance,
            source_filter=source_filter,
        )
        return generate_answer(question, chunks)

    def document_count(self) -> int:
        """Total chunks currently stored across all documents."""
        return self.db.count()

    def delete_document(self, source: str) -> None:
        """Remove a document (and all its chunks) from the system."""
        self.db.delete_by_source(source)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print('Usage: python rag_pipeline.py <path_to_pdf> "your question here"')
        sys.exit(1)

    pdf_file = sys.argv[1]
    question = sys.argv[2]

    pipeline = RAGPipeline()

    print(f"Ingesting '{pdf_file}'...")
    ingest_result = pipeline.ingest_document(pdf_file)
    print(f"  -> {ingest_result.chunks_created} chunks created from {ingest_result.pages_processed} pages\n")

    print(f"Question: {question}\n")
    result = pipeline.query(question)

    print("Answer:")
    print(result.answer)
    print("\nCitations:")
    for c in result.citations:
        print(f"  [{c.marker}] {c.source}, page {c.page}")