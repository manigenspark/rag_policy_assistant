"""Compare dense vs hybrid retrieval. Finds queries where keyword recovers a miss."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import CANDIDATE_K, CHROMA_DIR, COLLECTION, FINAL_K, RRF_K  # noqa: E402
from rag.embeddings import encode_query  # noqa: E402
from rag.hybrid import fuse  # noqa: E402
from rag.keyword import keyword_search  # noqa: E402
from rag.rerank import rerank  # noqa: E402
from rag.store import VectorStore  # noqa: E402

DEFAULT_QUERIES = [
    "POL-EXP-004",
    "POL-RMT-006",
    "POL-PTO-003",
    "What does POL-EXP-004 say about pre-approval?",
    "POL-EXP-007 non-reimbursable items",
    "Form POL-RMT-005 hybrid attendance",
]


def _section(chunk_id: str) -> str:
    parts = chunk_id.split("::")
    return parts[1] if len(parts) > 1 else chunk_id


def _rank_of(ids: list[str], needle: str) -> int | None:
    for index, chunk_id in enumerate(ids, start=1):
        if needle in chunk_id:
            return index
    return None


def compare(query: str, store: VectorStore, target: str) -> None:
    dense = store.dense_search(encode_query(query), k=CANDIDATE_K)
    keyword = keyword_search(query, store.get_all(), k=CANDIDATE_K)
    fused = fuse(dense, keyword, rrf_k=RRF_K)
    reranked = rerank(query, fused, top_k=max(FINAL_K, 10), min_prob=0.0)
    dense_ids = [hit.chunk_id for hit in dense]
    keyword_ids = [hit.chunk_id for hit in keyword]
    hybrid_ids = [hit.chunk_id for hit in fused]
    rerank_ids = [hit.chunk_id for hit in reranked]
    d_rank = _rank_of(dense_ids, target)
    k_rank = _rank_of(keyword_ids, target)
    h_rank = _rank_of(hybrid_ids, target)
    r_rank = _rank_of(rerank_ids, target)
    print(f"\nQ: {query}")
    print(f"  target {target}")
    print(
        f"  dense={d_rank or 'miss':<4}  "
        f"keyword={k_rank or 'miss':<4}  "
        f"hybrid={h_rank or 'miss':<4}  "
        f"rerank={r_rank or 'miss'}"
    )
    print("  dense  top5 :", ", ".join(_section(i) for i in dense_ids[:5]))
    print("  hybrid top5 :", ", ".join(_section(i) for i in hybrid_ids[:5]))
    print("  rerank top5 :", ", ".join(_section(i) for i in rerank_ids[:5]))
    if reranked:
        top = reranked[0]
        print(f"  rerank #1 p={top.probability:.3f} logit={top.logit:+.2f} {top.chunk_id}")
    dense_miss = d_rank is None or d_rank > 5
    hybrid_hit = h_rank is not None and h_rank <= 5
    if dense_miss and hybrid_hit:
        print(
            "  DEMO: vector-only misses the LLM top-5; hybrid recovers it. "
            "This is the locked hybrid-wins query."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dense vs hybrid retrieval compare")
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--target", default="POL-EXP-004")
    args = parser.parse_args(argv)
    store = VectorStore(CHROMA_DIR, COLLECTION)
    if store.count() == 0:
        print("empty store; run python scripts/ingest.py", file=sys.stderr)
        return 2
    queries = args.queries or DEFAULT_QUERIES
    print(f"corpus size: {store.count()}  candidate_k: {CANDIDATE_K}")
    for query in queries:
        target = args.target
        codes = [part for part in query.replace("?", "").split() if part.startswith("POL-")]
        if codes:
            target = codes[0].upper()
        compare(query, store, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
