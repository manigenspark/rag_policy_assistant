"""Ingest-time invariants: token budget, span round-trip, size visibility."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORDS,
    RAW_DIR,
    EmbeddingModel,
    active_embedding_model,
)
from rag.chunking import chunk_document, validate_chunks, word_histogram  # noqa: E402
from rag.documents import load_documents  # noqa: E402
from rag.embeddings import count_tokens, current_model  # noqa: E402
from rag.types import Chunk, PolicyDocument  # noqa: E402


def _chunks() -> tuple[list[PolicyDocument], list[Chunk], EmbeddingModel]:
    spec, model = current_model()
    documents = load_documents(RAW_DIR)
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                window_words=CHUNK_WORDS,
                overlap_words=CHUNK_OVERLAP_WORDS,
                count_tokens=lambda text: count_tokens(text, model),
            )
        )
    return documents, chunks, spec


def test_every_chunk_stays_under_the_token_ceiling() -> None:
    documents, chunks, spec = _chunks()
    validate_chunks(chunks, documents, max_tokens=spec.max_tokens)
    assert chunks
    assert max(chunk.token_count for chunk in chunks) <= spec.max_tokens
    assert spec.max_tokens == active_embedding_model().max_tokens


def test_spans_round_trip_against_the_raw_file() -> None:
    documents, chunks, spec = _chunks()
    by_path = {document.path: document for document in documents}
    for chunk in chunks:
        sliced = by_path[chunk.doc].raw[chunk.char_start:chunk.char_end]
        assert sliced == chunk.text, chunk.chunk_id


def test_oversized_section_is_windowed_with_overlap() -> None:
    _documents, chunks, _spec = _chunks()
    windows = [chunk for chunk in chunks if chunk.section_code == "POL-RMT-004"]
    assert len(windows) == 2
    first, second = windows
    first_words = set(first.text.split())
    second_words = set(second.text.split())
    assert first_words & second_words, "adjacent windows must share overlap words"


def test_histogram_covers_every_chunk() -> None:
    _documents, chunks, _spec = _chunks()
    total = sum(count for _label, count in word_histogram(chunks))
    assert total == len(chunks)
    assert len(chunks) == 27
