"""Ask a policy question through the vector-only RAG pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import FINAL_K  # noqa: E402
from rag.generate import ollama_available  # noqa: E402
from rag.pipeline import RAGPipeline, format_answer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the policy RAG assistant")
    parser.add_argument("question", nargs="+", help="natural-language question")
    parser.add_argument("--k", type=int, default=FINAL_K, help="final top-k chunks")
    parser.add_argument(
        "--mode",
        choices=("dense", "hybrid", "rerank"),
        default="rerank",
        help="dense | hybrid (RRF) | rerank (hybrid + cross-encoder)",
    )
    parser.add_argument(
        "--detect-conflicts",
        action="store_true",
        help="annotate version conflicts in the retrieved set (does not drop sources)",
    )
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="always send archived chunks to the LLM (default: only for old/new questions)",
    )
    args = parser.parse_args(argv)
    question = " ".join(args.question)

    if not ollama_available():
        print(
            "Ollama is not reachable. Start it, or set OLLAMA_HOST.",
            file=sys.stderr,
        )
        return 2

    answer = RAGPipeline().answer(
        question,
        k=args.k,
        mode=args.mode,
        detect_conflicts=True if args.detect_conflicts else None,
        prefer_current=not args.include_archived,
    )
    print(format_answer(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
