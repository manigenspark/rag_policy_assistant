"""Vector-only RAG: embed question, dense search, grounded generation."""
from __future__ import annotations

from config import CHROMA_DIR, COLLECTION, FINAL_K, OLLAMA_MODEL
from rag.embeddings import encode_query
from rag.generate import complete
from rag.store import DenseHit, VectorStore
from rag.types import Answer, Source


def _meta_str(meta: dict[str, object], key: str, default: str = "") -> str:
    value = meta.get(key, default)
    return default if value is None else str(value)


def _meta_int(meta: dict[str, object], key: str, default: int = 0) -> int:
    value = meta.get(key, default)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def hits_to_sources(hits: list[DenseHit]) -> list[Source]:
    sources: list[Source] = []
    for index, hit in enumerate(hits, start=1):
        meta = hit.metadata
        sources.append(
            Source(
                marker=f"[S{index}]",
                chunk_id=hit.chunk_id,
                doc=_meta_str(meta, "doc"),
                doc_version=_meta_str(meta, "doc_version"),
                status=_meta_str(meta, "status"),
                section_code=_meta_str(meta, "section_code"),
                section_title=_meta_str(meta, "section_title"),
                char_start=_meta_int(meta, "char_start"),
                char_end=_meta_int(meta, "char_end"),
                distance=hit.distance,
                text=hit.text,
            )
        )
    return sources


class RAGPipeline:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore(CHROMA_DIR, COLLECTION)

    def retrieve(self, question: str, k: int = FINAL_K) -> list[Source]:
        if self.store.count() == 0:
            raise RuntimeError(
                "the vector store is empty; run `python scripts/ingest.py` first"
            )
        query_vec = encode_query(question)
        hits = self.store.dense_search(query_vec, k=k)
        return hits_to_sources(hits)

    def answer(self, question: str, k: int = FINAL_K) -> Answer:
        sources = self.retrieve(question, k=k)
        text = complete(question, sources)
        return Answer(
            question=question,
            text=text,
            sources=tuple(sources),
            model=OLLAMA_MODEL,
            retrieval="dense",
        )


def format_answer(answer: Answer) -> str:
    lines = [
        f"Q: {answer.question}",
        "",
        answer.text,
        "",
        f"model: {answer.model}   retrieval: {answer.retrieval}",
        "sources:",
    ]
    for source in answer.sources:
        lines.append(
            f"  {source.header()}  dist={source.distance:.3f}"
        )
    return "\n".join(lines)
