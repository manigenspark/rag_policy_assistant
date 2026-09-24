"""Embedding helpers: query prefix must not be left to the caller to remember."""
from __future__ import annotations

from config import EmbeddingModel
from rag import embeddings


def test_bge_query_prefix_is_prepended(monkeypatch: object) -> None:
    captured: list[str] = []

    def fake_encode(texts: list[str], model: object = None) -> list[list[float]]:
        captured.extend(texts)
        return [[0.0, 0.0, 0.0]]

    spec = EmbeddingModel(
        model_id="BAAI/bge-small-en-v1.5",
        max_tokens=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
    )
    monkeypatch.setattr(embeddings, "active_embedding_model", lambda: spec)
    monkeypatch.setattr(embeddings, "encode_texts", fake_encode)
    vector = embeddings.encode_query("per diem")
    assert vector == [0.0, 0.0, 0.0]
    assert captured == [
        "Represent this sentence for searching relevant passages: per diem"
    ]


def test_minilm_query_has_no_prefix(monkeypatch: object) -> None:
    captured: list[str] = []

    def fake_encode(texts: list[str], model: object = None) -> list[list[float]]:
        captured.extend(texts)
        return [[0.0]]

    spec = EmbeddingModel(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=256,
    )
    monkeypatch.setattr(embeddings, "active_embedding_model", lambda: spec)
    monkeypatch.setattr(embeddings, "encode_texts", fake_encode)
    embeddings.encode_query("per diem")
    assert captured == ["per diem"]
