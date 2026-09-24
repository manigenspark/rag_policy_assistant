"""Evidence dump for the planted stale-document failure.

Prints retrieval ranks, raw-file grep, a generation with the detector off
(the genuine flawed/conflicting answer), and the same generation with the
detector on (warning only — archived chunks stay in the prompt).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_DIR  # noqa: E402
from rag.generate import ollama_available  # noqa: E402
from rag.lineage import find_version_conflicts, format_conflicts  # noqa: E402
from rag.pipeline import RAGPipeline, format_answer, hits_to_sources  # noqa: E402


class Case(TypedDict):
    id: str
    question: str
    section: str
    needle: re.Pattern[str]


CASES: list[Case] = [
    {
        "id": "per_diem",
        "question": "What is the international per diem for meals and incidentals?",
        "section": "POL-EXP-006",
        "needle": re.compile(r"per diem of\s+\d+\s+USD per day"),
    },
    {
        "id": "pre_approval",
        "question": "What is the pre-approval threshold for a single expense?",
        "section": "POL-EXP-004",
        "needle": re.compile(r"exceeding [\d,]+\s+USD requires"),
    },
]


def _grep(needle: re.Pattern[str]) -> list[str]:
    hits: list[str] = []
    for path in sorted(RAW_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        for match in needle.finditer(raw):
            line_no = raw[: match.start()].count("\n") + 1
            snippet = re.sub(r"\s+", " ", match.group(0))
            hits.append(f"{path.name}:{line_no}: {snippet}")
    return hits


def _print_ranks(pipeline: RAGPipeline, question: str, section: str) -> None:
    for mode in ("hybrid", "rerank"):
        sources = pipeline.retrieve(question, k=5, mode=mode)
        print(f"  {mode}")
        for source in sources:
            mark = "  <-- gold section" if source.section_code == section else ""
            print(
                f"    {source.marker}  {source.section_code:12}  "
                f"v{source.doc_version:<4}  {source.status:<9}  "
                f"{source.doc:24}  dist={source.distance:.3f}{mark}"
            )
        warning = format_conflicts(find_version_conflicts(sources))
        if warning:
            print()
            print("\n".join(f"    {line}" for line in warning.splitlines()))


def main() -> int:
    pipeline = RAGPipeline()
    print("=" * 78)
    print("corpus-wide version collisions (every stored chunk, no query)")
    print("=" * 78)
    corpus_sources = hits_to_sources(pipeline.store.get_all())
    corpus_warning = format_conflicts(find_version_conflicts(corpus_sources))
    print(corpus_warning or "none")

    live = ollama_available()
    for case in CASES:
        print()
        print("=" * 78)
        print(f"case: {case['id']}")
        print(f"Q:    {case['question']}")
        print("=" * 78)
        print()
        print("1. retrieval — was the gold chunk found?")
        _print_ranks(pipeline, case["question"], case["section"])
        print()
        print("2. source data — grepped on disk, before any embedding")
        for hit in _grep(case["needle"]):
            print(f"  {hit}")
        if not live:
            print()
            print("3. generation skipped (Ollama unreachable)")
            continue
        print()
        print("3. generation with DETECT_CONFLICTS off (hybrid, eval path)")
        silent = pipeline.answer(
            case["question"],
            k=5,
            mode="hybrid",
            detect_conflicts=False,
        )
        print(format_answer(silent))
        print()
        print("3b. generation with DETECT_CONFLICTS off (rerank, ask.py default)")
        reranked = pipeline.answer(
            case["question"],
            k=5,
            mode="rerank",
            detect_conflicts=False,
        )
        print(format_answer(reranked))
        print()
        print("4. generation with DETECT_CONFLICTS on (warning, sources kept)")
        flagged = pipeline.answer(
            case["question"],
            k=5,
            mode="hybrid",
            detect_conflicts=True,
        )
        print(format_answer(flagged))
        kept = [source.chunk_id for source in silent.sources] == [
            source.chunk_id for source in flagged.sources
        ]
        print()
        print(f"sources identical across 3 and 4: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
