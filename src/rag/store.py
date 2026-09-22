"""ChromaDB adapter: persist chunks and run dense cosine search."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import chromadb

from rag.types import Chunk


@dataclass(frozen=True)
class DenseHit:
    chunk_id: str
    text: str
    distance: float
    metadata: dict[str, Any]


class VectorStore:
    def __init__(self, persist_dir: Path, collection: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection_name = collection
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop and recreate so every ingest starts from an identical index."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=cast("list[Sequence[float]]", embeddings),
            documents=[chunk.embed_text() for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
        )
        return len(chunks)

    def count(self) -> int:
        return int(self._collection.count())

    def dense_search(self, query_embedding: list[float], k: int) -> list[DenseHit]:
        result = self._collection.query(
            query_embeddings=cast("list[Sequence[float]]", [query_embedding]),
            n_results=min(k, max(self.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0] if result["documents"] else [""] * len(ids)
        metadatas = result["metadatas"][0] if result["metadatas"] else [{}] * len(ids)
        distances = result["distances"][0] if result["distances"] else [0.0] * len(ids)
        hits: list[DenseHit] = []
        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            hits.append(
                DenseHit(
                    chunk_id=chunk_id,
                    text=text or "",
                    distance=float(distance),
                    metadata=dict(meta or {}),
                )
            )
        return hits
