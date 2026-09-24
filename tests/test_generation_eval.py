"""Answer accuracy against recorded fixtures; live Ollama is opt-in."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from rag.generate import ollama_available
from rag.pipeline import RAGPipeline
from rag.store import VectorStore

FIXTURE = Path(__file__).parent / "fixtures" / "generated_answers.json"
MARKER = re.compile(r"\[S(\d+)\]")
REFUSAL_HINTS = (
    "cannot find",
    "not in the provided",
    "do not contain",
    "don't have",
    "does not contain",
    "no information",
    "not mentioned",
)


def _load_fixtures() -> dict[str, Any]:
    if not FIXTURE.exists():
        pytest.skip("no generated_answers.json; run scripts/record_answers.py")
    payload: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload


def test_fixture_answers_cover_key_facts_and_valid_citations(
    eval_cases: list[dict[str, Any]],
) -> None:
    payload = _load_fixtures()
    answers = payload["answers"]
    missing = [case["id"] for case in eval_cases if case["id"] not in answers]
    assert not missing, f"fixtures missing cases: {missing}"

    for case in eval_cases:
        record = answers[case["id"]]
        text = str(record["text"]).lower()
        markers = {int(item) for item in MARKER.findall(record["text"])}
        n_sources = int(record.get("n_sources") or 0)
        if case.get("expect_refusal"):
            assert any(hint in text for hint in REFUSAL_HINTS), case["id"]
            continue
        for keyword in case.get("expected_keywords") or []:
            assert str(keyword).lower() in text, f"{case['id']} missing {keyword!r}"
        assert markers, f"{case['id']} has no [S#] citations"
        assert all(1 <= marker <= n_sources for marker in markers), case["id"]


@pytest.mark.live_llm
def test_live_generation_matches_key_facts(
    vector_store: VectorStore,
    eval_cases: list[dict[str, Any]],
) -> None:
    if not ollama_available():
        pytest.skip("Ollama is not reachable")
    pipeline = RAGPipeline(store=vector_store)
    for case in eval_cases:
        answer = pipeline.answer(
            case["question"],
            k=5,
            mode=str(case.get("retrieval") or "hybrid"),
        )
        text = answer.text.lower()
        if case.get("expect_refusal"):
            assert any(hint in text for hint in REFUSAL_HINTS)
            continue
        for keyword in case.get("expected_keywords") or []:
            assert str(keyword).lower() in text
        cited = {int(item) for item in MARKER.findall(answer.text)}
        assert cited
        assert all(1 <= marker <= len(answer.sources) for marker in cited)
