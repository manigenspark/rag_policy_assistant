"""Flag version conflicts in a retrieved set. Never silently drop a source.

The planted defect is an archived expense policy that shares section codes
with the current one. Retrieval is supposed to return both; this module only
annotates the contradiction so the caller can see it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rag.types import Source


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
