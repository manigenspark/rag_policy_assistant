"""RAG retrieve + generate: dense, hybrid RRF, or hybrid + cross-encoder rerank."""
from __future__ import annotations

from config import (
    CANDIDATE_K,
    CHROMA_DIR,
    COLLECTION,
    DETECT_VERSION_CONFLICTS,
    FINAL_K,
    OLLAMA_MODEL,
    RRF_K,
)
from rag.embeddings import encode_query
from rag.generate import complete
from rag.hybrid import FusedHit, fuse
from rag.keyword import keyword_search
from rag.lineage import find_version_conflicts, format_conflicts
from rag.rerank import rerank
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


def _fused_as_dense(hits: list[FusedHit]) -> list[DenseHit]:
    return [
        DenseHit(
            chunk_id=hit.chunk_id,
            text=hit.text,
            distance=1.0 - hit.rrf,
            metadata=dict(hit.metadata),
        )
        for hit in hits
    ]


class RAGPipeline:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore(CHROMA_DIR, COLLECTION)

    def _hybrid_candidates(self, question: str) -> list[FusedHit]:
        query_vec = encode_query(question)
        dense = self.store.dense_search(query_vec, k=CANDIDATE_K)
        keyword = keyword_search(question, self.store.get_all(), k=CANDIDATE_K)
        return fuse(dense, keyword, rrf_k=RRF_K)

    def retrieve(
        self,
        question: str,
        k: int = FINAL_K,
        mode: str = "dense",
    ) -> list[Source]:
        if self.store.count() == 0:
            raise RuntimeError(
                "the vector store is empty; run `python scripts/ingest.py` first"
            )
        if mode == "dense":
            query_vec = encode_query(question)
            hits = self.store.dense_search(query_vec, k=k)
            return hits_to_sources(hits)
        if mode == "hybrid":
            fused = self._hybrid_candidates(question)[:k]
            return hits_to_sources(_fused_as_dense(fused))
        if mode == "rerank":
            fused = self._hybrid_candidates(question)
            reranked = rerank(question, fused, top_k=k)
            dense_hits = [
                DenseHit(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    distance=1.0 - hit.probability,
                    metadata=dict(hit.metadata),
                )
                for hit in reranked
            ]
            return hits_to_sources(dense_hits)
        raise ValueError(f"unknown retrieval mode {mode!r}")

    def answer(
        self,
        question: str,
        k: int = FINAL_K,
        mode: str = "dense",
        *,
        detect_conflicts: bool | None = None,
    ) -> Answer:
        sources = self.retrieve(question, k=k, mode=mode)
        text = complete(question, sources)
        enabled = DETECT_VERSION_CONFLICTS if detect_conflicts is None else detect_conflicts
        warning = (
            format_conflicts(find_version_conflicts(sources)) if enabled else None
        )
        return Answer(
            question=question,
            text=text,
            sources=tuple(sources),
            model=OLLAMA_MODEL,
            retrieval=mode,
            warning=warning,
        )


def format_answer(answer: Answer) -> str:
    lines = [
        f"Q: {answer.question}",
        "",
        answer.text,
        "",
    ]
    if answer.warning:
        lines.extend([answer.warning, ""])
    lines.extend(
        [
            f"model: {answer.model}   retrieval: {answer.retrieval}",
            "sources:",
        ]
    )
    for source in answer.sources:
        lines.append(
            f"  {source.header()}  dist={source.distance:.3f}"
        )
    return "\n".join(lines)
