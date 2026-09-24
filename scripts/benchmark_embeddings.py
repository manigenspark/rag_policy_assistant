"""3-cell embedding bake-off: isolate the model, then isolate chunk size.

Cells:
  1. minilm / 160 words   — current default
  2. bge-small / 160 words — same window, different encoder
  3. bge-small / 300 words — uses BGE's 512-token headroom

Each cell rebuilds a throwaway Chroma index so production data/chroma is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "tests" / "eval_set.yaml"
TOP_K = 5

CELLS = [
    {"name": "minilm-160", "embed_model": "minilm", "chunk_words": 160},
    {"name": "bge-160", "embed_model": "bge-small", "chunk_words": 160},
    {"name": "bge-300", "embed_model": "bge-small", "chunk_words": 300},
]


def _rank(sources: list[Any], case: dict[str, Any]) -> int | None:
    section = case.get("expected_section")
    doc = case.get("expected_doc")
    if not section:
        return None
    for index, source in enumerate(sources, start=1):
        if source.section_code != section:
            continue
        if doc and source.doc != doc:
            continue
        return index
    return None


def _metrics(ranks: list[int | None]) -> tuple[float, float]:
    hits = [1.0 if rank is not None and rank <= TOP_K else 0.0 for rank in ranks]
    rr = [1.0 / rank if rank is not None else 0.0 for rank in ranks]
    n = len(ranks) or 1
    return sum(hits) / n, sum(rr) / n


def run_one_cell() -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

    import yaml

    from config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, RAW_DIR
    from rag.chunking import chunk_document, validate_chunks
    from rag.documents import load_documents
    from rag.embeddings import count_tokens, current_model, encode_texts
    from rag.pipeline import RAGPipeline
    from rag.store import VectorStore

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

    cases = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))["cases"]
    answerable = [case for case in cases if not case.get("expect_refusal")]

    with tempfile.TemporaryDirectory(prefix="bakeoff-") as tmp:
        store = VectorStore(Path(tmp), "bakeoff")
        embeddings = encode_texts([chunk.embed_text() for chunk in chunks], model=model)
        store.upsert(chunks, embeddings)
        pipeline = RAGPipeline(store=store)
        row: dict[str, Any] = {
            "embed_model": os.environ.get("EMBED_MODEL", ""),
            "model_id": spec.model_id,
            "chunk_words": CHUNK_WORDS,
            "chunks": len(chunks),
            "max_tokens": spec.max_tokens,
        }
        for mode in ("dense", "hybrid"):
            ranks = [
                _rank(pipeline.retrieve(case["question"], k=TOP_K, mode=mode), case)
                for case in answerable
            ]
            recall, mrr = _metrics(ranks)
            row[f"{mode}_recall"] = round(recall, 4)
            row[f"{mode}_mrr"] = round(mrr, 4)
        print(json.dumps(row))
    return 0


def _run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["EMBED_MODEL"] = str(cell["embed_model"])
    env["CHUNK_WORDS"] = str(cell["chunk_words"])
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}" + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    print(
        f"running {cell['name']}  "
        f"EMBED_MODEL={cell['embed_model']}  CHUNK_WORDS={cell['chunk_words']}",
        file=sys.stderr,
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--one-cell"],
        cwd=str(ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    line = completed.stdout.strip().splitlines()[-1]
    payload: dict[str, Any] = json.loads(line)
    payload["name"] = cell["name"]
    return payload


def _markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "cell",
        "model",
        "words",
        "chunks",
        "dense R@5",
        "dense MRR",
        "hybrid R@5",
        "hybrid MRR",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row["embed_model"],
                    str(row["chunk_words"]),
                    str(row["chunks"]),
                    f"{row['dense_recall']:.2f}",
                    f"{row['dense_mrr']:.2f}",
                    f"{row['hybrid_recall']:.2f}",
                    f"{row['hybrid_mrr']:.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _pick_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the CI default among cells that clear the 6a gate.

    Hybrid recall is already 1.0 on this corpus for every cell, so the
    embedder's own ranking (dense MRR) is the discriminating axis. MiniLM
    is the tie-break because it is smaller and already cached in CI.
    """
    eligible = [
        row
        for row in rows
        if row["hybrid_recall"] >= 0.80 and row["hybrid_mrr"] >= 0.70
    ]
    pool = eligible or rows
    return max(
        pool,
        key=lambda row: (
            row["dense_mrr"],
            row["hybrid_mrr"],
            row["hybrid_recall"],
            0 if row["embed_model"] == "minilm" else -1,
            -int(row["chunk_words"]),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--one-cell",
        action="store_true",
        help="internal: run the cell described by EMBED_MODEL/CHUNK_WORDS",
    )
    args = parser.parse_args(argv)
    if args.one_cell:
        return run_one_cell()

    rows = [_run_cell(cell) for cell in CELLS]
    print()
    print(_markdown(rows))
    winner = _pick_winner(rows)
    print()
    print(
        f"winner: {winner['name']}  "
        f"hybrid R@5={winner['hybrid_recall']:.2f}  "
        f"MRR={winner['hybrid_mrr']:.2f}"
    )
    print("gate: hybrid recall@5 >= 0.80 and MRR >= 0.70")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
