"""Shared pytest fixtures: path setup, eval cases, and a session vector index."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, RAW_DIR  # noqa: E402
from rag.chunking import chunk_document, validate_chunks  # noqa: E402
from rag.documents import load_documents  # noqa: E402
from rag.embeddings import count_tokens, current_model, encode_texts  # noqa: E402
from rag.store import VectorStore  # noqa: E402

EVAL_PATH = Path(__file__).parent / "eval_set.yaml"


@pytest.fixture(scope="session")
def eval_cases() -> list[dict[str, Any]]:
    payload = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
    cases = list(payload["cases"])
    assert len(cases) >= 8
    return cases


@pytest.fixture(scope="session")
def answerable_cases(eval_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [case for case in eval_cases if not case.get("expect_refusal")]


@pytest.fixture(scope="session")
def vector_store(tmp_path_factory: pytest.TempPathFactory) -> VectorStore:
    spec, model = current_model()
    documents = load_documents(RAW_DIR)
    chunks = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                window_words=CHUNK_WORDS,
                overlap_words=CHUNK_OVERLAP_WORDS,
                count_tokens=lambda text: count_tokens(text, model),
            )
        )
    validate_chunks(chunks, documents, max_tokens=spec.max_tokens)
    store = VectorStore(tmp_path_factory.mktemp("chroma"), "eval")
    embeddings = encode_texts([chunk.embed_text() for chunk in chunks], model=model)
    store.upsert(chunks, embeddings)
    return store
