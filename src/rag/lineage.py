"""Flag version conflicts and decide when archived chunks reach the LLM.

Retrieval is supposed to see the planted stale document. Generation drops it
unless the question is actually about old vs new policy.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace

from rag.types import Source

# Comparative / historical questions need the archived copy in the prompt.
# A rate question ("how much per day") does not.
ARCHIVE_QUESTION = re.compile(
    r"(?i)("
    r"\bold\b|\bprevious\b|\bbefore\b|\barchived\b|\bformer\b|"
    r"used to|\bdifference\b|\bdiffer\b|\bcompare\b|\bversus\b|\bvs\.?\b|"
    r"when did|\bchanged\b|turn(?:ed)? to|new and old|old and new|"
    r"\bbefore and now\b|\band now\b|"
    r"v1\.4|\b2022\b"
    r")"
)


def needs_archived_context(question: str) -> bool:
    return bool(ARCHIVE_QUESTION.search(question))


def renumber_sources(sources: list[Source]) -> list[Source]:
    return [
        replace(source, marker=f"[S{index}]")
        for index, source in enumerate(sources, start=1)
    ]


def drop_archived_sources(sources: list[Source]) -> list[Source]:
    """Remove archived chunks. If that would empty the set, leave it alone."""
    kept = [source for source in sources if source.status != "archived"]
    if not kept:
        return sources
    return renumber_sources(kept)


def focus_on_top_section(sources: list[Source]) -> list[Source]:
    """Keep every version of the top-ranked section; drop off-topic extras."""
    if not sources:
        return sources
    code = sources[0].section_code
    kept = [source for source in sources if source.section_code == code]
    return renumber_sources(kept)


def sources_for_generation(
    question: str,
    sources: list[Source],
    *,
    prefer_current: bool,
) -> list[Source]:
    """Keep archived text only when the question needs a before/after compare."""
    if not prefer_current:
        return sources
    if needs_archived_context(question):
        return focus_on_top_section(sources)
    return drop_archived_sources(sources)


@dataclass(frozen=True)
class VersionConflict:
    section_code: str
    sources: tuple[Source, ...]

    @property
    def versions(self) -> tuple[str, ...]:
        return tuple(sorted({source.doc_version for source in self.sources}))


def find_version_conflicts(sources: list[Source]) -> list[VersionConflict]:
    """Sections that appear under more than one doc_version in `sources`."""
    by_code: dict[str, list[Source]] = defaultdict(list)
    for source in sources:
        if source.section_code:
            by_code[source.section_code].append(source)

    conflicts: list[VersionConflict] = []
    for code in sorted(by_code):
        group = by_code[code]
        versions = {source.doc_version for source in group}
        if len(versions) < 2:
            continue
        # Stable order: current first, then archived, then marker.
        ordered = sorted(
            group,
            key=lambda item: (
                0 if item.status == "current" else 1,
                item.marker,
                item.doc,
            ),
        )
        conflicts.append(VersionConflict(section_code=code, sources=tuple(ordered)))
    return conflicts


def format_conflicts(conflicts: list[VersionConflict]) -> str | None:
    if not conflicts:
        return None
    lines = [
        f"LINEAGE WARNING: {len(conflicts)} section(s) retrieved in more than "
        "one document version. Archived copies were not dropped.",
    ]
    for conflict in conflicts:
        lines.append(f"  {conflict.section_code}  versions {', '.join(conflict.versions)}")
        for source in conflict.sources:
            lines.append(
                f"    {source.marker}  {source.doc}  "
                f"v{source.doc_version}  {source.status}"
            )
    lines.append(
        "This is a corpus defect: the retriever returned contradictory policy. "
        "Prefer status=current, or remove the stale document from the source system."
    )
    return "\n".join(lines)
