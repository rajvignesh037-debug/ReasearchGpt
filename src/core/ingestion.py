"""
Phase 1: Document Ingestion
----------------------------
Extracts text from PDF research papers and splits it into overlapping
chunks with metadata (source file, page number, chunk index).

This module has no external dependencies beyond pdfplumber - no API keys,
no vector DB, no network calls. Test it standalone before moving to Phase 2.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber

# Defaults matching the project README
CHUNK_SIZE = 400     # characters per chunk
CHUNK_OVERLAP = 50    # characters shared between consecutive chunks


@dataclass
class Chunk:
    """A single chunk of text with metadata needed for retrieval and citation."""
    id: str
    text: str
    source: str        # original filename
    page: int          # 1-indexed page number this chunk came from
    chunk_index: int    # position of this chunk within the document


def extract_text_by_page(pdf_path: str) -> List[tuple[int, str]]:
    """
    Extract raw text from a PDF, one entry per page.

    Returns a list of (page_number, text) tuples. page_number is 1-indexed.
    Pages with no extractable text (e.g. pure image scans) are skipped.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append((i, text))

    if not pages:
        raise ValueError(
            f"No extractable text found in {pdf_path.name}. "
            "It may be a scanned/image-only PDF (needs OCR - not handled in Phase 1)."
        )

    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into fixed-size character chunks with overlap.

    Overlap ensures context isn't lost when a sentence/idea is split
    across a chunk boundary.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def process_pdf(pdf_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Full Phase 1 pipeline: PDF -> extracted pages -> chunks with metadata.

    This is the function later phases (embeddings.py) will call.
    """
    source_name = Path(pdf_path).name
    pages = extract_text_by_page(pdf_path)

    all_chunks: List[Chunk] = []
    chunk_index = 0

    for page_num, page_text in pages:
        page_chunks = chunk_text(page_text, chunk_size, overlap)
        for chunk_str in page_chunks:
            all_chunks.append(
                Chunk(
                    id=f"{source_name}_p{page_num}_c{chunk_index}",
                    text=chunk_str,
                    source=source_name,
                    page=page_num,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

    return all_chunks


def batch_ingest_folder(folder_path: str = "data", vector_db: Optional[Any] = None) -> Dict[str, Any]:
    """Scan a folder for PDFs, ingest new ones, and skip previously ingested files."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    manifest_path = folder / ".ingested_manifest.json"
    manifest: Dict[str, Dict[str, str]] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}

    if vector_db is None:
        from core.vector_db import VectorDB

        vector_db = VectorDB()

    ingested_files: List[str] = []
    skipped_files: List[str] = []
    failed_files: List[Dict[str, str]] = []

    for pdf_path in sorted(folder.glob("*.pdf")):
        try:
            file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            previous_record = manifest.get(pdf_path.name)
            if previous_record and previous_record.get("hash") == file_hash:
                skipped_files.append(pdf_path.name)
                continue

            chunks = process_pdf(str(pdf_path))
            if not chunks:
                raise ValueError(f"No extractable text found in {pdf_path.name}")

            from core.embeddings import embed_chunks

            embedded = embed_chunks(chunks)
            vector_db.add_chunks(embedded)

            manifest[pdf_path.name] = {
                "hash": file_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            ingested_files.append(pdf_path.name)
        except Exception as exc:
            failed_files.append({"file": pdf_path.name, "error": str(exc)})

    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "ingested_files": ingested_files,
        "skipped_files": skipped_files,
        "failed_files": failed_files,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    chunks = process_pdf(pdf_file)

    print(f"\nProcessed '{pdf_file}': {len(chunks)} chunks created\n")
    for c in chunks[:3]:
        print(f"[{c.id}] (page {c.page})")
        print(c.text[:200] + ("..." if len(c.text) > 200 else ""))
        print("-" * 60)