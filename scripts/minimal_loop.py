"""Minimal embed-store-retrieve loop. Run this before the full pipeline.

Two known snippets go into a throwaway Chroma collection. A query that is
paraphrased (not a substring of either snippet) must rank the per-diem
snippet first. That is the proof the vector loop works, which the assignment
asks for before chunking, hybrid, or rerank are added.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rag.embeddings import encode_query, encode_texts  # noqa: E402
from rag.store import VectorStore  # noqa: E402
from rag.types import Chunk  # noqa: E402

SNIPPETS = [
    Chunk(
        chunk_id="snippet::pto::0",
        text="Employees may carry a maximum of 5 unused PTO days into the next year.",
        doc="snippet_pto.md",
        doc_version="0",
        status="demo",
        effective_date="2026-01-01",
        section_code="SNIP-PTO",
        section_title="Carryover",
        char_start=0,
        char_end=72,
        word_count=14,
        token_count=20,
    ),
    Chunk(
        chunk_id="snippet::per_diem::0",
        text="Employees travelling internationally receive a per diem of 95 USD per day.",
        doc="snippet_exp.md",
        doc_version="0",
        status="demo",
        effective_date="2026-01-01",
        section_code="SNIP-EXP",
        section_title="Per Diem",
        char_start=0,
        char_end=74,
        word_count=13,
        token_count=20,
    ),
]

QUERY = "What is the meal allowance for overseas travel?"


def run_loop(persist_dir: Path) -> list[tuple[str, float]]:
    store = VectorStore(persist_dir, "minimal")
    embeddings = encode_texts([chunk.embed_text() for chunk in SNIPPETS])
    store.upsert(SNIPPETS, embeddings)
    hits = store.dense_search(encode_query(QUERY), k=2)
    return [(hit.chunk_id, hit.distance) for hit in hits]


def main() -> int:
    print("minimal embed-store-retrieve loop")
    print(f"query: {QUERY}")
    print()
    with tempfile.TemporaryDirectory(prefix="minimal-chroma-") as tmp:
        ranked = run_loop(Path(tmp))
    print(f"{'rank':<6} {'chunk_id':<24} {'cosine distance':>16}  closer?")
    print("-" * 62)
    for index, (chunk_id, distance) in enumerate(ranked, start=1):
        closer = "YES  <-- expected" if index == 1 else "no"
        print(f"{index:<6} {chunk_id:<24} {distance:>16.4f}  {closer}")
    if not ranked or "per_diem" not in ranked[0][0]:
        print("FAIL: per-diem snippet was not rank 1", file=sys.stderr)
        return 1
    print()
    print("PASS: paraphrased query retrieved the per-diem snippet first.")
    print("cosine distance: lower is closer (hnsw:space=cosine).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
