"""
Phase 2: Vector Database
--------------------------
Stores embedded chunks in ChromaDB and provides similarity search.

Uses a persistent local ChromaDB instance (data survives between runs),
stored under ./chroma_data by default.
"""

from typing import List, Optional

import chromadb

from core.embeddings import EmbeddedChunk

DB_PATH = "./chroma_data"
COLLECTION_NAME = "research_papers"


class VectorDB:
    """Thin wrapper around a ChromaDB collection for storing and querying chunks."""

    def __init__(self, db_path: str = DB_PATH, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for OpenAI embeddings
        )

    def add_chunks(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """Store a batch of embedded chunks. Re-adding the same chunk id overwrites it."""
        if not embedded_chunks:
            return

        self.collection.upsert(
            ids=[ec.chunk.id for ec in embedded_chunks],
            embeddings=[ec.embedding for ec in embedded_chunks],
            documents=[ec.chunk.text for ec in embedded_chunks],
            metadatas=[
                {"source": ec.chunk.source, "page": ec.chunk.page, "chunk_index": ec.chunk.chunk_index}
                for ec in embedded_chunks
            ],
        )

    def query(self, query_embedding: List[float], top_k: int = 5, source_filter: Optional[str] = None) -> List[dict]:
        """
        Find the top_k most similar chunks to a query embedding.

        Optionally filter to a single source document (for multi-doc mode later).
        Returns a list of dicts with text, metadata, and similarity distance.
        """
        where = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return matches

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self.collection.count()

    def delete_by_source(self, source: str) -> None:
        """Remove all chunks belonging to a given source document."""
        self.collection.delete(where={"source": source})


if __name__ == "__main__":
    import sys
    from core.ingestion import process_pdf
    from core.embeddings import embed_chunks

    if len(sys.argv) < 2:
        print("Usage: python vector_db.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]

    print("Ingesting...")
    chunks = process_pdf(pdf_file)

    print("Embedding...")
    embedded = embed_chunks(chunks)

    print("Storing in ChromaDB...")
    db = VectorDB()
    db.add_chunks(embedded)

    print(f"\nDone. Collection now has {db.count()} chunks total.")

    # Quick sanity check: use the first chunk's own embedding as a test query
    print("\nSanity check - searching for the most similar chunks to chunk[0]:")
    results = db.query(embedded[0].embedding, top_k=3)
    for r in results:
        print(f"  [{r['id']}] distance={r['distance']:.4f} (page {r['metadata']['page']})")
        print(f"  {r['text'][:150]}...")