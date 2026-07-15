"""
Phase 2: Embeddings
--------------------
Turns text chunks (from ingestion.py) into vector embeddings using
OpenAI's embedding API. Batches requests to keep API calls efficient.

Requires OPENAI_API_KEY to be set in the environment (e.g. via a .env file).
"""

import os
import time
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from core.ingestion import Chunk

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # chunks per API call - OpenAI allows up to 2048 inputs per request

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class EmbeddedChunk:
    """A chunk paired with its vector embedding, ready for storage in ChromaDB."""
    chunk: Chunk
    embedding: List[float]


def embed_batch(texts: List[str], model: str = EMBEDDING_MODEL, max_retries: int = 3) -> List[List[float]]:
    """
    Get embeddings for a batch of texts in a single API call.

    Retries with exponential backoff on transient failures (rate limits, timeouts).
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=model, input=texts)
            # response.data is returned in the same order as the input list
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Embedding request failed after {max_retries} attempts: {e}")
            wait = 2 ** attempt
            print(f"Embedding call failed ({e}); retrying in {wait}s...")
            time.sleep(wait)


def embed_chunks(chunks: List[Chunk], model: str = EMBEDDING_MODEL, batch_size: int = BATCH_SIZE) -> List[EmbeddedChunk]:
    """
    Embed a list of Chunks, batching requests for efficiency.

    This is the function Phase 2's vector_db.py (and later rag_pipeline.py)
    will call after ingestion produces chunks.
    """
    if not chunks:
        return []

    embedded_chunks: List[EmbeddedChunk] = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        vectors = embed_batch(texts, model=model)

        for chunk, vector in zip(batch, vectors):
            embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=vector))

        print(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    return embedded_chunks


if __name__ == "__main__":
    import sys
    from core.ingestion import process_pdf

    if len(sys.argv) < 2:
        print("Usage: python embeddings.py <path_to_pdf>")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Create a .env file with OPENAI_API_KEY=sk-...")
        sys.exit(1)

    pdf_file = sys.argv[1]
    chunks = process_pdf(pdf_file)
    print(f"Loaded {len(chunks)} chunks from '{pdf_file}'\n")

    embedded = embed_chunks(chunks)

    print(f"\nDone. {len(embedded)} chunks embedded.")
    print(f"Embedding dimension: {len(embedded[0].embedding)}")
    print(f"Sample chunk id: {embedded[0].chunk.id}")