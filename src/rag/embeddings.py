"""Cached SentenceTransformer loader, token counting, and encode helpers."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import EmbeddingModel, active_embedding_model


@lru_cache(maxsize=4)
def load_model(model_id: str) -> SentenceTransformer:
    return SentenceTransformer(model_id)


def current_model() -> tuple[EmbeddingModel, SentenceTransformer]:
    spec = active_embedding_model()
    return spec, load_model(spec.model_id)


def count_tokens(text: str, model: SentenceTransformer | None = None) -> int:
    """Count wordpieces the same way the embedding model will see them."""
    if model is None:
        _, model = current_model()
    encoded = model.tokenizer(text, add_special_tokens=True, truncation=False)
    return int(len(encoded["input_ids"]))


def encode_texts(texts: list[str], model: SentenceTransformer | None = None) -> list[list[float]]:
    if model is None:
        _, model = current_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def encode_query(query: str, model: SentenceTransformer | None = None) -> list[float]:
    spec = active_embedding_model()
    prefixed = f"{spec.query_prefix}{query}" if spec.query_prefix else query
    return encode_texts([prefixed], model=model)[0]
