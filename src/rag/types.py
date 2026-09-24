"""Shared dataclasses that every pipeline stage speaks.

The Chunk contract is the spine: text, identity, lineage, and provenance all
travel together so attribution and the Phase 7 diagnosis are data, not narrative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Section:
    """One ## heading and the body that follows it."""

    code: str
    title: str
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class PolicyDocument:
    """A source markdown file after front-matter and section parse."""

    path: str
    title: str
    doc_id: str
    version: str
    status: str
    effective_date: str
    owner: str
    raw: str
    sections: tuple[Section, ...]


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit, with lineage and a span back into the raw file."""

    chunk_id: str
    text: str
    doc: str
    doc_version: str
    status: str
    effective_date: str
    section_code: str
    section_title: str
    char_start: int
    char_end: int
    word_count: int
    token_count: int

    def embed_text(self) -> str:
        """Text actually sent to the embedding model.

        The heading is prepended so the vector and the keyword index both see
        the section code. The stored span still covers only the body, which is
        what validate_chunks() round-trips against the raw file.
        """
        return f"{self.section_code} - {self.section_title}\n{self.text}"

    def metadata(self) -> dict[str, Any]:
        """Chroma-safe metadata (str / int / float / bool only)."""
        return {
            "doc": self.doc,
            "doc_version": self.doc_version,
            "status": self.status,
            "effective_date": self.effective_date,
            "section_code": self.section_code,
            "section_title": self.section_title,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "word_count": self.word_count,
            "token_count": self.token_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Source:
    """One retrieved chunk as it is cited in an answer."""

    marker: str
    chunk_id: str
    doc: str
    doc_version: str
    status: str
    section_code: str
    section_title: str
    char_start: int
    char_end: int
    distance: float
    text: str

    def header(self) -> str:
        return (
            f"{self.marker} {self.doc} § {self.section_code} "
            f"\"{self.section_title}\" "
            f"(v{self.doc_version}, {self.status}, chars {self.char_start}-{self.char_end})"
        )


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: tuple[Source, ...]
    model: str
    retrieval: str = "dense"
    warning: str | None = None
