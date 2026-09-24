"""Phase 3 sanity check: two snippets, paraphrased query, correct one first."""
from __future__ import annotations

from pathlib import Path

from scripts.minimal_loop import QUERY, run_loop


def test_paraphrased_query_ranks_per_diem_first(tmp_path: Path) -> None:
    ranked = run_loop(tmp_path)
    assert len(ranked) == 2
    assert "per_diem" in ranked[0][0]
    assert "pto" in ranked[1][0]
    assert ranked[0][1] < ranked[1][1]
    assert "overseas" in QUERY.lower() or "meal" in QUERY.lower()
