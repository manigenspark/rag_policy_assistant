"""Cross-encoder rerank: score (query, chunk) pairs jointly, keep top-k."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from typing import cast

from sentence_transformers import CrossEncoder

from config import MIN_RERANK_PROB, RERANK_MODEL
from rag.hybrid import FusedHit
from rag.store import DenseHit

Pair = tuple[str, str]
Scorer = Callable[[Sequence[Pair]], Sequence[float]]


@dataclass(frozen=True)
class RerankedHit:
    chunk_id: str
    text: str
    metadata: dict[str, object]
    logit: float
    probability: float
    rrf: float | None = None


def sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp = math.exp(logit)
    return exp / (1.0 + exp)


@lru_cache(maxsize=2)
def load_reranker(model_id: str = RERANK_MODEL) -> CrossEncoder:
    return cast(CrossEncoder, CrossEncoder(model_id))


def _default_scorer(pairs: Sequence[Pair]) -> Sequence[float]:
    model = load_reranker()
    scores = model.predict(list(pairs), show_progress_bar=False)
    return [float(score) for score in scores]


def rerank(
    query: str,
    hits: Sequence[FusedHit | DenseHit | RerankedHit],
    *,
    top_k: int,
    min_prob: float = MIN_RERANK_PROB,
    scorer: Scorer | None = None,
) -> list[RerankedHit]:
    """Re-score candidates. Query is always the first member of each pair."""
    if not hits:
        return []
    score_fn = scorer or _default_scorer
    pairs: list[Pair] = [(query, hit.text) for hit in hits]
    logits = list(score_fn(pairs))
    if len(logits) != len(hits):
        raise ValueError("scorer must return one logit per hit")
    ranked: list[RerankedHit] = []
    for hit, logit in zip(hits, logits):
        rrf = hit.rrf if isinstance(hit, FusedHit) else None
        ranked.append(
            RerankedHit(
                chunk_id=hit.chunk_id,
                text=hit.text,
                metadata=dict(hit.metadata),
                logit=float(logit),
                probability=sigmoid(float(logit)),
                rrf=rrf,
            )
        )
    ranked.sort(key=lambda item: (-item.probability, item.chunk_id))
    filtered = [hit for hit in ranked if hit.probability >= min_prob]
    chosen = filtered if filtered else ranked[:1]
    return chosen[:top_k]
