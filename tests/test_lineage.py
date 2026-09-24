"""Version-conflict detector: flag, never drop."""
from __future__ import annotations

from rag.lineage import find_version_conflicts, format_conflicts
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
    )
    flagged = pipeline.answer(
        "What is the international per diem for meals and incidentals?",
        k=5,
        mode="hybrid",
        detect_conflicts=True,
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
