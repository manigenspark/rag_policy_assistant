"""Record Ollama answers for the eval set so CI can check accuracy without a GPU."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from config import OLLAMA_MODEL  # noqa: E402
from rag.generate import ollama_available  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

EVAL_PATH = ROOT / "tests" / "eval_set.yaml"
OUT_PATH = ROOT / "tests" / "fixtures" / "generated_answers.json"


def main() -> int:
    if not ollama_available():
        print("Ollama is not reachable", file=sys.stderr)
        return 2
    cases = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))["cases"]
    pipeline = RAGPipeline()
    answers: dict[str, object] = {}
    for case in cases:
        mode = str(case.get("retrieval") or "hybrid")
        answer = pipeline.answer(case["question"], k=5, mode=mode)
        answers[case["id"]] = {
            "question": case["question"],
            "text": answer.text,
            "n_sources": len(answer.sources),
            "source_ids": [source.chunk_id for source in answer.sources],
            "mode": mode,
        }
        print(f"recorded {case['id']}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": OLLAMA_MODEL,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "answers": answers,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
