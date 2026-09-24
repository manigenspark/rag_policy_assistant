"""Version-conflict detector: flag, never drop."""
from __future__ import annotations

from rag.lineage import (
    drop_archived_sources,
    find_version_conflicts,
    format_conflicts,
    needs_archived_context,
)
from rag.types import Source


def _source(
    marker: str,
    code: str,
    doc: str,
    version: str,
    status: str,
) -> Source:
    return Source(
        marker=marker,
        chunk_id=f"{doc}::{code}::0",
        doc=doc,
        doc_version=version,
        status=status,
        section_code=code,
        section_title="Test",
        char_start=0,
        char_end=10,
        distance=0.1,
        text="placeholder",
    )


def test_same_section_different_versions_is_a_conflict() -> None:
    sources = [
        _source("[S1]", "POL-EXP-006", "expense_policy.md", "3.0", "current"),
        _source("[S2]", "POL-EXP-006", "expense_policy_2022.md", "1.4", "archived"),
        _source("[S3]", "POL-EXP-001", "expense_policy.md", "3.0", "current"),
    ]
    conflicts = find_version_conflicts(sources)
    assert len(conflicts) == 1
    assert conflicts[0].section_code == "POL-EXP-006"
    assert set(conflicts[0].versions) == {"1.4", "3.0"}
    assert len(conflicts[0].sources) == 2


def test_same_version_is_not_a_conflict() -> None:
    sources = [
        _source("[S1]", "POL-EXP-006", "expense_policy.md", "3.0", "current"),
        _source("[S2]", "POL-PTO-001", "pto_policy.md", "2.0", "current"),
    ]
    assert find_version_conflicts(sources) == []


def test_format_does_not_imply_sources_were_dropped() -> None:
    sources = [
        _source("[S1]", "POL-EXP-004", "expense_policy.md", "3.0", "current"),
        _source("[S2]", "POL-EXP-004", "expense_policy_2022.md", "1.4", "archived"),
    ]
    warning = format_conflicts(find_version_conflicts(sources))
    assert warning is not None
    assert "were not dropped" in warning
    assert "expense_policy.md" in warning
    assert "expense_policy_2022.md" in warning
    assert format_conflicts([]) is None


def test_pipeline_warns_without_dropping_or_calling_a_filter(
    vector_store: object,
    monkeypatch: object,
) -> None:
    from rag.pipeline import RAGPipeline, format_answer
    from rag.store import VectorStore

    assert isinstance(vector_store, VectorStore)
    monkeypatch.setattr("rag.pipeline.complete", lambda question, sources: "95 USD")
    pipeline = RAGPipeline(store=vector_store)
    silent = pipeline.answer(
        "What is the international per diem for meals and incidentals?",
        k=5,
        mode="hybrid",
        detect_conflicts=False,
        prefer_current=False,
    )
    flagged = pipeline.answer(
        "What is the international per diem for meals and incidentals?",
        k=5,
        mode="hybrid",
        detect_conflicts=True,
        prefer_current=False,
    )
    assert silent.warning is None
    assert flagged.warning is not None
    assert "POL-EXP-006" in flagged.warning
    assert [source.chunk_id for source in silent.sources] == [
        source.chunk_id for source in flagged.sources
    ]
    assert len(flagged.sources) == 5
    rendered = format_answer(flagged)
    assert "LINEAGE WARNING" in rendered
    assert flagged.text == "95 USD"


def test_prefer_current_drops_archived_when_current_exists(
    vector_store: object,
) -> None:
    from rag.pipeline import RAGPipeline
    from rag.store import VectorStore

    assert isinstance(vector_store, VectorStore)
    pipeline = RAGPipeline(store=vector_store)
    question = "how much per day to cover meals and incidental expenses"
    mixed = pipeline.retrieve(question, k=5, mode="hybrid", prefer_current=False)
    current = pipeline.retrieve(question, k=5, mode="hybrid", prefer_current=True)
    assert any(source.status == "archived" for source in mixed)
    assert all(source.status != "archived" for source in current)
    assert any(source.section_code == "POL-EXP-006" for source in current)


def test_rate_question_strips_archive_before_the_llm(
    vector_store: object,
    monkeypatch: object,
) -> None:
    from rag.pipeline import RAGPipeline
    from rag.store import VectorStore

    assert isinstance(vector_store, VectorStore)
    seen: list[list[Source]] = []

    def fake_complete(question: str, sources: list[Source]) -> str:
        seen.append(sources)
        return "95 USD"

    monkeypatch.setattr("rag.pipeline.complete", fake_complete)
    RAGPipeline(store=vector_store).answer(
        "how much per day to cover meals and incidental expenses",
        k=5,
        mode="hybrid",
        prefer_current=True,
    )
    assert seen
    assert all(source.status != "archived" for source in seen[0])
    assert any(source.section_code == "POL-EXP-006" for source in seen[0])


def test_compare_question_keeps_archive_for_the_llm(
    vector_store: object,
    monkeypatch: object,
) -> None:
    from rag.pipeline import RAGPipeline
    from rag.store import VectorStore

    assert isinstance(vector_store, VectorStore)
    seen: list[list[Source]] = []

    def fake_complete(question: str, sources: list[Source]) -> str:
        seen.append(sources)
        return "95 vs 75"

    monkeypatch.setattr("rag.pipeline.complete", fake_complete)
    RAGPipeline(store=vector_store).answer(
        "whats the new and old policy of the meal and incidental coverage",
        k=5,
        mode="hybrid",
        prefer_current=True,
    )
    assert seen
    assert any(source.status == "archived" for source in seen[0])
    assert any(source.status == "current" for source in seen[0])
    assert all(source.section_code == seen[0][0].section_code for source in seen[0])


def test_needs_archived_context_detects_compare_not_rate() -> None:
    assert needs_archived_context("whats the new and old policy")
    assert needs_archived_context("when did it turn to 95")
    assert not needs_archived_context(
        "how much per day to cover meals and incidental expenses"
    )


def test_drop_archived_renumbers_markers() -> None:
    sources = [
        _source("[S1]", "POL-EXP-006", "expense_policy.md", "3.0", "current"),
        _source("[S2]", "POL-EXP-006", "expense_policy_2022.md", "1.4", "archived"),
    ]
    kept = drop_archived_sources(sources)
    assert [source.marker for source in kept] == ["[S1]"]
    assert kept[0].status == "current"
