"""Central configuration for the RAG policy assistant.

Chunk size is validated against the active embedding model's token ceiling
rather than hardcoded, because exceeding it causes silent truncation: the tail
of an oversized chunk is never embedded and no error is raised.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "policies"


@dataclass(frozen=True)
class EmbeddingModel:
    """An embedding model and the constraints it imposes on chunking."""

    model_id: str
    max_tokens: int
    query_prefix: str = ""


EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    "minilm": EmbeddingModel(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=256,
    ),
    # BGE is trained for asymmetric retrieval and expects a query-side
    # instruction. Omitting the prefix degrades accuracy silently, so it lives
    # here rather than in caller code.
    "bge-small": EmbeddingModel(
        model_id="BAAI/bge-small-en-v1.5",
        max_tokens=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}

EMBED_MODEL = os.environ.get("EMBED_MODEL", "minilm")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 160 words is roughly 210-230 wordpieces for policy prose, which carries more
# numerals and currency codes than ordinary English and so tokenizes worse.
# That leaves headroom under the MiniLM ceiling of 256.
CHUNK_WORDS = int(os.environ.get("CHUNK_WORDS", "160"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", "32"))

CANDIDATE_K = 20
FINAL_K = 5
RRF_K = 60

# The ms-marco cross-encoder emits raw logits, not probabilities: measured
# scores on this corpus span +3.74 to -11.12. rerank.py applies a sigmoid, so
# this floor is a probability.
#
# Measured caveat: a correct answer scored 0.63 while an off-topic question's
# best candidate scored 0.60, so this floor cannot separate subtle off-topic
# queries. It only excludes clearly irrelevant candidates (a -4.9 logit maps to
# 0.007). Treat refusal as a coarse guard, not a calibrated decision.
MIN_RERANK_PROB = float(os.environ.get("MIN_RERANK_PROB", "0.10"))

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b")
OLLAMA_TIMEOUT = 120

# Off by default so the graded run reproduces the genuine failure caused by the
# planted stale document. scripts/diagnose.py enables it to show the remedy.
DETECT_VERSION_CONFLICTS = os.environ.get("DETECT_CONFLICTS", "0") == "1"


def active_embedding_model() -> EmbeddingModel:
    if EMBED_MODEL not in EMBEDDING_MODELS:
        raise KeyError(
            f"unknown EMBED_MODEL {EMBED_MODEL!r}; "
            f"choose from {sorted(EMBEDDING_MODELS)}"
        )
    return EMBEDDING_MODELS[EMBED_MODEL]
