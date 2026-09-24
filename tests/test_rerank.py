"""Tests for cross-encoder rerank (injected scorer, no model download)."""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rag.hybrid import FusedHit  # noqa: E402
from rag.rerank import rerank, sigmoid  # noqa: E402


def test_sigmoid_maps_zero_to_half() -> None:
    assert abs(sigmoid(0.0) - 0.5) < 1e-9
    assert sigmoid(10.0) > 0.99
    assert sigmoid(-10.0) < 0.01


def test_rerank_orders_by_injected_logits() -> None:
    hits = [
        FusedHit("low", "irrelevant text", {}, 0.02, 1, None),
        FusedHit("high", "the answer lives here", {}, 0.01, 2, None),
    ]

    def scorer(pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [3.0 if "answer" in text else -4.0 for _query, text in pairs]

    ranked = rerank("need approval?", hits, top_k=2, min_prob=0.0, scorer=scorer)
    assert ranked[0].chunk_id == "high"
    assert ranked[0].probability > ranked[1].probability
    assert ranked[0].logit == 3.0


def test_rerank_keeps_best_if_all_below_floor() -> None:
    hits = [
        FusedHit("a", "aaa", {}, 0.1, 1, None),
        FusedHit("b", "bbb", {}, 0.2, 2, None),
    ]

    def scorer(pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [-8.0] * len(pairs)

    ranked = rerank("q", hits, top_k=5, min_prob=0.5, scorer=scorer)
    assert len(ranked) == 1
    assert ranked[0].probability < 0.01
