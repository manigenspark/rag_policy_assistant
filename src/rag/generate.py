"""Ollama chat adapter and the grounded RAG prompt."""
from __future__ import annotations

import requests

from config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT
from rag.types import Source

SYSTEM_PROMPT = """You are an internal policy assistant.
Answer the employee's question using ONLY the numbered sources in the user message.
Rules:
- Cite every factual claim with a source marker like [S1] or [S2].
- If the sources do not contain the answer, say you cannot find it in the
  provided policies. Do not invent policy.
- Quote figures (dollars, days, thresholds) exactly as they appear in the sources.
- If the same section appears as current and archived, report BOTH figures.
  Label the archived value as previous and the current value as current.
  Do not mention only the current number when an older number is in the sources.
- Do not add topics the question did not ask and the sources do not require
  (for example domestic travel or remote work on a per-diem question).
- Keep the answer concise: a few sentences, then stop.
"""


def ollama_available(host: str = OLLAMA_HOST, timeout: float = 3.0) -> bool:
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def build_user_prompt(question: str, sources: list[Source]) -> str:
    blocks = [source.header() + "\n" + source.text for source in sources]
    joined = "\n\n".join(blocks)
    return (
        f"Question:\n{question}\n\n"
        f"Sources:\n{joined}\n\n"
        "Answer:"
    )


def complete(
    question: str,
    sources: list[Source],
    *,
    host: str = OLLAMA_HOST,
    model: str = OLLAMA_MODEL,
    timeout: int = OLLAMA_TIMEOUT,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, sources)},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    response = requests.post(
        f"{host.rstrip('/')}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    message = body.get("message") or {}
    text = str(message.get("content") or "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned an empty completion: {body!r}")
    return text
