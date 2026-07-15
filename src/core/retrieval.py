"""
Phase 3: Retrieval
--------------------
Takes a user's question, embeds it, and retrieves the most relevant
chunks from the vector DB. This is the bridge between storage (Phase 2)
and generation (Phase 4).
"""

from dataclasses import dataclass
from typing import List, Optional

from core.embeddings import embed_batch, EMBEDDING_MODEL
from core.vector_db import VectorDB

DEFAULT_TOP_K = 5
# Cosine distance threshold - matches above this are considered too irrelevant to use.
# ChromaDB cosine distance ranges 0 (identical) to 2 (opposite); 0.8 is a reasonably loose cutoff.
DEFAULT_MAX_DISTANCE = 0.8


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval, ready to be fed into the generation prompt."""
    text: str
    source: str
    page: int
    distance: float


def embed_query(query: str) -> List[float]:
    """Embed a single query string using the same model as document chunks."""
    return embed_batch([query], model=EMBEDDING_MODEL)[0]


def retrieve(
    query: str,
    db: VectorDB,
    top_k: int = DEFAULT_TOP_K,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    source_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve the top_k most relevant chunks for a query.

    Filters out matches beyond max_distance so clearly irrelevant chunks
    don't get passed to the LLM (which could otherwise cause hallucinated
    answers stitched from unrelated text).
    """
    query_embedding = embed_query(query)
    raw_matches = db.query(query_embedding, top_k=top_k, source_filter=source_filter)

    results = []
    for match in raw_matches:
        if match["distance"] <= max_distance:
            results.append(
                RetrievedChunk(
                    text=match["text"],
                    source=match["metadata"]["source"],
                    page=match["metadata"]["page"],
                    distance=match["distance"],
                )
            )

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python retrieval.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    db = VectorDB()

    if db.count() == 0:
        print("Vector DB is empty. Run vector_db.py on a PDF first to ingest a document.")
        sys.exit(1)

    print(f"Query: {question}\n")
    chunks = retrieve(question, db)

    if not chunks:
        print("No sufficiently relevant chunks found (all matches exceeded the distance threshold).")
    else:
        print(f"Found {len(chunks)} relevant chunk(s):\n")
        for i, c in enumerate(chunks, start=1):
            print(f"[{i}] {c.source} (page {c.page}, distance={c.distance:.4f})")
            print(c.text[:200] + ("..." if len(c.text) > 200 else ""))
            print("-" * 60)