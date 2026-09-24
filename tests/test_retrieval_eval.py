"""Retrieval eval: recall@5, MRR, and hybrid vs dense on the code query."""
from __future__ import annotations

from typing import Any

from rag.pipeline import RAGPipeline
from rag.store import VectorStore
from rag.types import Source

TOP_K = 5


def _rank(sources: list[Source], case: dict[str, Any]) -> int | None:
    section = case.get("expected_section")
    doc = case.get("expected_doc")
    if not section:
        return None
    for index, source in enumerate(sources, start=1):
        if source.section_code != section:
            continue
        if doc and source.doc != doc:
            continue
        return index
    return None


def _metrics(ranks: list[int | None]) -> tuple[float, float]:
    hits = [1.0 if rank is not None and rank <= TOP_K else 0.0 for rank in ranks]
    rr = [1.0 / rank if rank is not None else 0.0 for rank in ranks]
    n = len(ranks) or 1
    return sum(hits) / n, sum(rr) / n


def test_hybrid_recall_and_mrr(
    vector_store: VectorStore,
    answerable_cases: list[dict[str, Any]],
) -> None:
    pipeline = RAGPipeline(store=vector_store)
    ranks: list[int | None] = []
    print("\nhybrid retrieval")
    for case in answerable_cases:
        sources = pipeline.retrieve(case["question"], k=TOP_K, mode="hybrid")
        rank = _rank(sources, case)
        ranks.append(rank)
        print(f"  {case['id']:<18} rank={rank or 'miss'}")
    recall, mrr = _metrics(ranks)
    print(f"  recall@{TOP_K}={recall:.2f}  MRR={mrr:.2f}")
    assert recall >= 0.80
    assert mrr >= 0.70


def test_hybrid_beats_dense_on_policy_code_query(vector_store: VectorStore) -> None:
    """Code-only query: dense misses POL-EXP-004; keyword recovers a copy of it.

    Two versions of the section exist (current v3.0 and planted v1.4). The
    hybrid win is recovering the section code into top-5, not picking the
    current file — the stale copy is a Phase 7 diagnosis issue.
    """
    pipeline = RAGPipeline(store=vector_store)
    question = "POL-EXP-004"
    case = {"expected_section": "POL-EXP-004"}
    dense_rank = _rank(pipeline.retrieve(question, k=20, mode="dense"), case)
    hybrid_rank = _rank(pipeline.retrieve(question, k=5, mode="hybrid"), case)
    assert dense_rank is None or dense_rank > 5, dense_rank
    assert hybrid_rank is not None and hybrid_rank <= 5, hybrid_rank


def test_gold_keywords_fit_in_one_expected_chunk(
    vector_store: VectorStore,
    answerable_cases: list[dict[str, Any]],
) -> None:
    corpus = vector_store.get_all()
    for case in answerable_cases:
        keywords = [str(item).lower() for item in case.get("expected_keywords") or []]
        section = case["expected_section"]
        doc = case.get("expected_doc")
        matching = [
            hit
            for hit in corpus
            if hit.metadata.get("section_code") == section
            and (not doc or hit.metadata.get("doc") == doc)
        ]
        assert matching, f"no chunk for {case['id']}"
        blob = " ".join(hit.text.lower() for hit in matching)
        for keyword in keywords:
            assert keyword.lower() in blob, f"{case['id']} missing {keyword!r}"
