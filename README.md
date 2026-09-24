# RAG Policy Assistant

Internal Q&A over four generated company policy documents. Chunk, embed with
`sentence-transformers`, store in Chroma, retrieve with dense + keyword RRF,
optionally rerank with a cross-encoder, generate with Ollama `mistral:7b`, and
cite the source of every claim.

## What is this

A from-scratch RAG lab: no starter corpus, no hosted vector DB. The interesting
failure is planted on purpose — an archived expense policy that contradicts the
current one — so we can prove the retriever is healthy and the corpus is not.

## Architecture

```
INGEST (once)                         QUERY (per question)
──────────────                        ────────────────────
data/raw/*.md
  YAML lineage (version, status)
        │
        ▼
documents.py  split on ## POL-…
        │
        ▼
chunking.py   160-word windows,
              overlap only on overflow,
              validate_chunks()
        │
        ▼
embeddings.py MiniLM 384-d, cosine
        │
        ▼
store.py      Chroma PersistentClient
              hnsw:space=cosine
                                          question
                                              │
                                              ▼
                                     encode_query() ──► dense search k=20
                                              │
                                     keyword.py (codes + terms)
                                              │
                                              ▼
                                     hybrid.py RRF 1/(60+rank)
                                              │
                                              ▼
                                     rerank.py CrossEncoder
                                              │
                                              ▼
                                     lineage.py  flag version collisions
                                              │   (opt-in, never drops)
                                              ▼
                                     generate.py  Ollama mistral:7b
                                              │   temperature=0
                                              ▼
                                     answer + [S#] citations
```

## Engineering decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Embeddings | `all-MiniLM-L6-v2` (384-d, 256 tokens) | Local, free, no API key. Bake-off vs BGE is in the table below. |
| Vector store | Chroma persistent, cosine | Zero ops. pgvector is the production upgrade, not this lab. |
| Chunking | Split on `## POL-…` first, 160-word windows / 32-word overlap only when a section overflows | MiniLM truncates at 256 wordpieces. 160 words of policy prose is ~210–230 tokens. Overlap only on the one section that needs it (POL-RMT-004). |
| Hybrid | Substring + `POL-XXX-000` regex, fused with RRF `1/(60+rank)` | Lab allows grep-style keyword. Cosine distance and hit counts are not comparable, so we fuse ranks, not scores. BM25 is the upgrade once term frequency matters. |
| Rerank | `ms-marco-MiniLM-L-6-v2` on the top 20 | Joint query-chunk score. On this 27-chunk corpus it mostly reorders; it cannot see `status`. |
| Generation | Ollama `mistral:7b`, temperature 0 | No cloud key. CI records answers into a fixture so the runner does not need Ollama. |
| Planted defect | `expense_policy_2022.md` v1.4 archived, ingested unfiltered | The graded diagnosis has to show a real conflict. Filtering it out would hide the bug. |

Rejected: semantic chunking, 768/1024-d embeddings for their own sake, score blending, NDCG, LLM-as-judge.

## How to run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/minimal_loop.py          # two snippets, paraphrased query
python scripts/ingest.py                # parse → chunk → validate → embed
python scripts/ask.py "What is the international per diem?"
python scripts/compare_retrieval.py     # dense vs hybrid vs rerank
python scripts/diagnose.py              # planted-conflict evidence
```

Ollama must be reachable at `OLLAMA_HOST` (default
`http://host.docker.internal:11434`) with `mistral:7b` pulled. Set
`DETECT_CONFLICTS=1` or pass `--detect-conflicts` to annotate version
collisions without dropping sources.

## How to test it

```bash
pytest -v
flake8 .
mypy src/
```

All three must pass before a pipeline push. `pytest -m live_llm` hits Ollama
and is skipped in CI. Refresh recorded answers after prompt or retrieval
changes:

```bash
python scripts/record_answers.py
```

## Retrieval eval

`tests/eval_set.yaml` has 10 answerable questions plus one refusal. Gates,
measured with hybrid retrieval on MiniLM / 160 words:

| metric | gate | measured |
| --- | --- | --- |
| recall@5 | ≥ 0.80 | 1.00 |
| MRR | ≥ 0.70 | 0.93 |

`POL-EXP-004` as a code-only query is the locked hybrid-wins case: dense rank
11, hybrid rank 4.

## Embedding bake-off

Same eval set, throwaway indexes. Cell 1 vs 2 isolates the encoder; cell 2 vs 3
isolates chunk size.

| cell | model | words | chunks | dense R@5 | dense MRR | hybrid R@5 | hybrid MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| minilm-160 | minilm | 160 | 27 | 0.90 | **0.85** | 1.00 | 0.93 |
| bge-160 | bge-small | 160 | 27 | 0.90 | 0.80 | 1.00 | 1.00 |
| bge-300 | bge-small | 300 | 26 | 0.90 | 0.80 | 1.00 | 1.00 |

**CI default: minilm / 160.** Every cell clears the gate. BGE raises hybrid
MRR but drops dense MRR — the ranking the embedder owns. Reproduce:

```bash
python scripts/benchmark_embeddings.py
```

## Planted conflict

`expense_policy_2022.md` (v1.4, archived) shares every `POL-EXP-*` section with
the current policy and disagrees on the figures (95 vs 75 USD per diem, 500 vs
1,000 USD pre-approval). Retrieval returns both. The two-question analysis is
`DIAGNOSIS.md`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBED_MODEL` | `minilm` | `minilm` or `bge-small` |
| `CHUNK_WORDS` | `160` | Window size; validated against the model token ceiling |
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Chat endpoint |
| `OLLAMA_MODEL` | `mistral:7b` | Generator, temperature 0 |
| `DETECT_CONFLICTS` | `0` | `1` annotates version collisions; never drops chunks |

## Known limitations

- 27 chunks. recall@5 moves in steps of 0.1. The gates are regression guards.
- `CANDIDATE_K = 20` over 27 chunks is nearly exhaustive. The cross-encoder is
  doing most of the ranking at this scale.
- Rerank can prefer the archived passage because both copies are equally on
  topic and the model has no `status` field.
- Keyword matching is substring, not BM25.
- Refusal uses a coarse cross-encoder probability floor; it cannot separate a
  correct answer (0.63) from an off-topic best candidate (0.60).
- The whole corpus is ~12k tokens. Pasting it into the prompt would probably
  answer this set better. The pipeline exists for scale and attribution.

Screenshot order for the PDF is in `deliverable/CAPTURE.md`.
