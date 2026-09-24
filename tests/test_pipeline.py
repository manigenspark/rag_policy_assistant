"""Context assembly for the vector-only RAG path (no LLM required)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rag.pipeline import hits_to_sources  # noqa: E402
from rag.store import DenseHit  # noqa: E402


def test_hits_to_sources_preserves_lineage_and_markers() -> None:
    hit = DenseHit(
        chunk_id="expense_policy.md::POL-EXP-006::0",
        text="Employees travelling internationally receive a per diem of 95 USD.",
        distance=0.22,
        metadata={
            "doc": "expense_policy.md",
            "doc_version": "3.0",
            "status": "current",
            "section_code": "POL-EXP-006",
            "section_title": "International Travel and Per Diem",
            "char_start": 100,
            "char_end": 200,
        },
    )
    sources = hits_to_sources([hit])
    assert sources[0].marker == "[S1]"
    assert sources[0].doc_version == "3.0"
    assert sources[0].section_code == "POL-EXP-006"
    assert "v3.0, current, chars 100-200" in sources[0].header()
