"""Reciprocal Rank Fusion of dense and keyword ranked lists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rag.keyword import KeywordHit
from rag.store import DenseHit

from config import RRF_K


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    text: str
    metadata: dict[str, object]
    rrf: float
    dense_rank: int | None
    keyword_rank: int | None


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """score(id) = sum 1/(k + rank) over lists that contain the id. Ranks are 1-based."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse(
    dense: list[DenseHit],
    keyword: list[KeywordHit],
    *,
    rrf_k: int = RRF_K,
) -> list[FusedHit]:
    dense_ids = [hit.chunk_id for hit in dense]
    keyword_ids = [hit.chunk_id for hit in keyword]
    scores = reciprocal_rank_fusion([dense_ids, keyword_ids], k=rrf_k)
    dense_rank = {hit.chunk_id: index for index, hit in enumerate(dense, start=1)}
    keyword_rank = {hit.chunk_id: index for index, hit in enumerate(keyword, start=1)}
    by_id: dict[str, tuple[str, Mapping[str, object]]] = {}
    for dense_hit in dense:
        by_id[dense_hit.chunk_id] = (dense_hit.text, dense_hit.metadata)
    for keyword_hit in keyword:
        by_id.setdefault(keyword_hit.chunk_id, (keyword_hit.text, keyword_hit.metadata))
    fused = [
        FusedHit(
            chunk_id=chunk_id,
            text=by_id[chunk_id][0],
            metadata=dict(by_id[chunk_id][1]),
            rrf=score,
            dense_rank=dense_rank.get(chunk_id),
            keyword_rank=keyword_rank.get(chunk_id),
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda item: (-item.rrf, item.chunk_id))
    return fused
