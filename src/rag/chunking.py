"""Section-aware windowing with overlap and ingest-time validation."""
from __future__ import annotations

import re
from collections.abc import Callable

from rag.types import Chunk, PolicyDocument, Section

WORD = re.compile(r"\S+")


class ChunkValidationError(ValueError):
    """Raised when a chunk would silently degrade retrieval."""


def word_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of each whitespace-separated token."""
    return [(match.start(), match.end()) for match in WORD.finditer(text)]


def _windows(n_words: int, size: int, overlap: int) -> list[tuple[int, int]]:
    if n_words == 0:
        return []
    if n_words <= size:
        return [(0, n_words)]
    stride = max(1, size - overlap)
    starts = list(range(0, n_words - size, stride))
    starts.append(n_words - size)
    # Dedup if the last window lands on the same start as the previous one.
    unique: list[tuple[int, int]] = []
    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        unique.append((start, start + size))
    return unique


def chunk_section(
    document: PolicyDocument,
    section: Section,
    *,
    window_words: int,
    overlap_words: int,
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    spans = word_spans(section.text)
    chunks: list[Chunk] = []
    for index, (word_start, word_end) in enumerate(
        _windows(len(spans), window_words, overlap_words)
    ):
        local_start = spans[word_start][0]
        local_end = spans[word_end - 1][1]
        text = section.text[local_start:local_end]
        char_start = section.char_start + local_start
        char_end = section.char_start + local_end
        heading = f"{section.code} - {section.title}\n"
        token_count = count_tokens(heading + text)
        chunks.append(
            Chunk(
                chunk_id=f"{document.path}::{section.code}::{index}",
                text=text,
                doc=document.path,
                doc_version=document.version,
                status=document.status,
                effective_date=document.effective_date,
                section_code=section.code,
                section_title=section.title,
                char_start=char_start,
                char_end=char_end,
                word_count=word_end - word_start,
                token_count=token_count,
            )
        )
    return chunks


def chunk_document(
    document: PolicyDocument,
    *,
    window_words: int,
    overlap_words: int,
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in document.sections:
        chunks.extend(
            chunk_section(
                document,
                section,
                window_words=window_words,
                overlap_words=overlap_words,
                count_tokens=count_tokens,
            )
        )
    return chunks


def validate_chunks(
    chunks: list[Chunk],
    documents: list[PolicyDocument],
    *,
    max_tokens: int,
) -> None:
    """Hard-fail on anything that would silently break retrieval or attribution."""
    by_path = {document.path: document for document in documents}
    errors: list[str] = []
    required = (
        "chunk_id",
        "text",
        "doc",
        "doc_version",
        "status",
        "effective_date",
        "section_code",
        "section_title",
    )
    for chunk in chunks:
        for field in required:
            if not str(getattr(chunk, field)).strip():
                errors.append(f"{chunk.chunk_id}: missing {field}")
        if not chunk.text.strip():
            errors.append(f"{chunk.chunk_id}: empty text")
        if chunk.token_count > max_tokens:
            errors.append(
                f"{chunk.chunk_id}: {chunk.token_count} tokens exceeds "
                f"model ceiling of {max_tokens} (silent truncation)"
            )
        source = by_path.get(chunk.doc)
        if source is None:
            errors.append(f"{chunk.chunk_id}: unknown source document {chunk.doc}")
            continue
        sliced = source.raw[chunk.char_start:chunk.char_end]
        if sliced != chunk.text:
            errors.append(
                f"{chunk.chunk_id}: span {chunk.char_start}:{chunk.char_end} "
                "does not round-trip against the raw file"
            )
    if errors:
        joined = "\n  ".join(errors)
        raise ChunkValidationError(f"{len(errors)} chunk validation error(s):\n  {joined}")


def word_histogram(chunks: list[Chunk]) -> list[tuple[str, int]]:
    """Bucket word counts so the size choice is visible, not implicit."""
    edges = (40, 80, 120, 160, 200, 10_000)
    labels = ["1-40", "41-80", "81-120", "121-160", "161-200", "201+"]
    counts = [0] * len(edges)
    for chunk in chunks:
        for index, edge in enumerate(edges):
            if chunk.word_count <= edge:
                counts[index] += 1
                break
    return list(zip(labels, counts))
