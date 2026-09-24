"""Unit tests for keyword scoring and reciprocal rank fusion."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rag.hybrid import fuse, reciprocal_rank_fusion  # noqa: E402
from rag.keyword import KeywordHit, extract_codes, keyword_search  # noqa: E402
from rag.store import DenseHit  # noqa: E402


def _hit(chunk_id: str, text: str, code: str) -> DenseHit:
    return DenseHit(
        chunk_id=chunk_id,
        text=text,
        distance=0.4,
        metadata={"section_code": code, "section_title": "Test"},
    )


def test_extract_policy_codes() -> None:
    assert extract_codes("see POL-EXP-004 please") == ["POL-EXP-004"]


def test_keyword_ranks_exact_code_first() -> None:
    corpus = [
        _hit("a::POL-PTO-001::0", "leave accrual", "POL-PTO-001"),
        _hit(
            "a::POL-EXP-004::0",
            "POL-EXP-004 - Pre-Approval Thresholds\n500 USD",
            "POL-EXP-004",
        ),
        _hit("a::POL-RMT-001::0", "remote work", "POL-RMT-001"),
    ]
    hits = keyword_search("POL-EXP-004", corpus, k=3)
    assert hits[0].chunk_id.endswith("POL-EXP-004::0")
    assert hits[0].score >= 100


def test_rrf_rewards_agreement_over_a_single_high_rank() -> None:
    scores = reciprocal_rank_fusion(
        [
            ["dense_top", "both"],
            ["both", "kw_top"],
        ],
        k=60,
    )
    assert scores["both"] > scores["dense_top"]
    assert scores["both"] > scores["kw_top"]


def test_fuse_dedups_and_orders_by_rrf() -> None:
    dense = [
        _hit("doc::A::0", "aaa", "A"),
        _hit("doc::B::0", "bbb", "B"),
    ]
    keyword = [
        KeywordHit("doc::B::0", "bbb", 100.0, {"section_code": "B"}),
        KeywordHit("doc::C::0", "ccc", 10.0, {"section_code": "C"}),
    ]
    fused = fuse(dense, keyword, rrf_k=60)
    ids = [hit.chunk_id for hit in fused]
    assert ids[0] == "doc::B::0"
    assert set(ids) == {"doc::A::0", "doc::B::0", "doc::C::0"}


def test_code_only_query_hybrid_puts_target_in_top5() -> None:
    """Live index: POL-EXP-004 is the measured case where dense misses top-5."""
    from config import CANDIDATE_K, CHROMA_DIR, COLLECTION, RRF_K
    from rag.embeddings import encode_query
    from rag.store import VectorStore

    store = VectorStore(CHROMA_DIR, COLLECTION)
    if store.count() == 0:
        import pytest

        pytest.skip("run scripts/ingest.py first")
    query = "POL-EXP-004"
    dense = store.dense_search(encode_query(query), k=CANDIDATE_K)
    keyword = keyword_search(query, store.get_all(), k=CANDIDATE_K)
    fused = fuse(dense, keyword, rrf_k=RRF_K)
    dense_rank = next(
        (i for i, h in enumerate(dense, 1) if "POL-EXP-004" in h.chunk_id),
        None,
    )
    hybrid_rank = next(
        (i for i, h in enumerate(fused, 1) if "POL-EXP-004" in h.chunk_id),
        None,
    )
    assert dense_rank is None or dense_rank > 5
    assert hybrid_rank is not None and hybrid_rank <= 5
