"""Sparse keyword search: normalized terms plus exact policy-code match."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag.store import DenseHit

CODE = re.compile(r"POL-[A-Z]{3}-\d{3}", re.IGNORECASE)
NON_TOKEN = re.compile(r"[^a-z0-9]+")
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "at",
    "is", "are", "do", "does", "i", "my", "me", "we", "our", "what",
    "how", "when", "where", "which", "about", "with", "from", "this",
}


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, object]


def normalize(text: str) -> str:
    return NON_TOKEN.sub(" ", text.casefold()).strip()


def terms(text: str) -> list[str]:
    return [token for token in normalize(text).split() if token and token not in STOPWORDS]


def extract_codes(text: str) -> list[str]:
    return [match.group(0).upper() for match in CODE.finditer(text)]


def _score(query: str, hit: DenseHit) -> float:
    codes = extract_codes(query)
    haystack = f"{hit.text} {hit.metadata.get('section_code', '')}".upper()
    score = 0.0
    for code in codes:
        if code in haystack:
            score += 100.0
    query_terms = terms(query)
    if not query_terms:
        return score
    blob = normalize(hit.text)
    section = normalize(str(hit.metadata.get("section_title", "")))
    blob = f"{blob} {section}"
    overlap = sum(1 for term in query_terms if term in blob)
    score += float(overlap)
    return score


def keyword_search(query: str, corpus: list[DenseHit], k: int) -> list[KeywordHit]:
    scored: list[KeywordHit] = []
    for hit in corpus:
        value = _score(query, hit)
        if value <= 0:
            continue
        scored.append(
            KeywordHit(
                chunk_id=hit.chunk_id,
                text=hit.text,
                score=value,
                metadata=dict(hit.metadata),
            )
        )
    scored.sort(key=lambda item: (-item.score, item.chunk_id))
    return scored[:k]
