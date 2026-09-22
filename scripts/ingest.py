"""Parse → chunk → validate → embed → upsert into ChromaDB."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    CHROMA_DIR,
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORDS,
    COLLECTION,
    RAW_DIR,
)
from rag.chunking import chunk_document, validate_chunks, word_histogram  # noqa: E402
from rag.documents import load_documents  # noqa: E402
from rag.embeddings import count_tokens, current_model, encode_texts  # noqa: E402
from rag.store import VectorStore  # noqa: E402


def ingest() -> int:
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

    print(f"embedding model : {spec.model_id}")
    print(f"token ceiling   : {spec.max_tokens}")
    print(f"window / overlap: {CHUNK_WORDS} / {CHUNK_OVERLAP_WORDS} words")
    print(f"documents       : {len(documents)}")
    print(f"sections        : {sum(len(document.sections) for document in documents)}")
    print(f"chunks          : {len(chunks)}")
    print()
    print("chunk word-count histogram")
    print("-" * 28)
    for label, count in word_histogram(chunks):
        bar = "#" * count
        print(f"  {label:<8} {count:>3}  {bar}")

    tokens = [chunk.token_count for chunk in chunks]
    print()
    print(
        f"tokens  min={min(tokens)}  mean={sum(tokens) / len(tokens):.1f}  "
        f"max={max(tokens)}  ceiling={spec.max_tokens}"
    )
    windowed = sum(1 for chunk in chunks if chunk.chunk_id.endswith("::1"))
    print(f"windowed sections: {windowed}")

    store = VectorStore(CHROMA_DIR, COLLECTION)
    store.reset()
    embeddings = encode_texts([chunk.embed_text() for chunk in chunks], model=model)
    store.upsert(chunks, embeddings)
    print(f"stored in chroma : {store.count()} vectors at {CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(ingest())
