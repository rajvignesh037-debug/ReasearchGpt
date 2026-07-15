"""
Phase 4: Generation
---------------------
Takes a user's question and the chunks retrieved for it, builds a prompt
that forces the model to answer only from context and cite its sources,
then parses the response back into real source/page citations.
"""

import os
import re
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from core.retrieval import RetrievedChunk

load_dotenv()

GENERATION_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a research assistant that answers questions using ONLY the provided context from research papers.

Rules:
- Answer only using the numbered context blocks below. Do not use outside knowledge.
- Every claim you make must be followed by a citation marker like [1], [2], etc., referring to the context block it came from.
- If the answer is not contained in the context, say "I don't know based on the provided document(s)." Do not guess or make up information.
- Be concise and precise."""


@dataclass
class Citation:
    marker: int      # the [1], [2] etc. number used in the answer
    source: str
    page: int


@dataclass
class GeneratedAnswer:
    answer: str
    citations: List[Citation]


def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    """Build the user-turn prompt with numbered context blocks."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(f"[{i}] (source: {chunk.source}, page {chunk.page})\n{chunk.text}")

    context_str = "\n\n".join(context_blocks)

    return f"""Context:
{context_str}

Question: {query}

Answer the question using only the context above, with citation markers like [1], [2] after each claim."""


def extract_used_citation_numbers(answer_text: str) -> List[int]:
    """Find which [N] markers the model actually used in its answer, in order of first appearance."""
    found = re.findall(r"\[(\d+)\]", answer_text)
    seen = []
    for n in found:
        n = int(n)
        if n not in seen:
            seen.append(n)
    return seen


def generate_answer(query: str, chunks: List[RetrievedChunk], model: str = GENERATION_MODEL) -> GeneratedAnswer:
    """
    Generate a cited answer to `query` using `chunks` as context.

    If chunks is empty, returns a canned "I don't know" response without
    calling the API at all - no point spending a call on empty context.
    """
    if not chunks:
        return GeneratedAnswer(
            answer="I don't know based on the provided document(s). No relevant context was found.",
            citations=[],
        )

    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,  # deterministic, factual answers - not creative writing
    )

    answer_text = response.choices[0].message.content

    # Map [N] markers back to real source/page metadata
    used_numbers = extract_used_citation_numbers(answer_text)
    citations = []
    for n in used_numbers:
        if 1 <= n <= len(chunks):
            chunk = chunks[n - 1]
            citations.append(Citation(marker=n, source=chunk.source, page=chunk.page))

    return GeneratedAnswer(answer=answer_text, citations=citations)


if __name__ == "__main__":
    import sys
    from core.vector_db import VectorDB
    from core.retrieval import retrieve

    if len(sys.argv) < 2:
        print('Usage: python generation.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    db = VectorDB()

    if db.count() == 0:
        print("Vector DB is empty. Run vector_db.py on a PDF first to ingest a document.")
        sys.exit(1)

    print(f"Question: {question}\n")
    chunks = retrieve(question, db)
    result = generate_answer(question, chunks)

    print("Answer:")
    print(result.answer)
    print("\nCitations:")
    for c in result.citations:
        print(f"  [{c.marker}] {c.source}, page {c.page}")