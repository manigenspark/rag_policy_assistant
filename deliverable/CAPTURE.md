# Capture guide — 11 screenshots, rubric order

One PDF, this order, nothing extra between items. Run each command from
`rag_policy_assistant/` with the venv active. Crop to the terminal or editor
pane that the row names.

| # | Rubric item | Command / what to open | What the screenshot must show |
| --- | --- | --- | --- |
| 1 | Generated source documents, planted issue identified | `ls data/raw` then open `data/raw/expense_policy.md` and `data/raw/expense_policy_2022.md` (front matter + POL-EXP-006) | Four files. Current `version: "3.0"` / `status: current` vs archived `version: "1.4"` / `status: archived`. Per diem **95** vs **75**. |
| 2 | Minimal embed-store-retrieve loop | `python scripts/minimal_loop.py` | Query about overseas meal allowance. Rank 1 is `snippet::per_diem::0`. Line `PASS: paraphrased query retrieved the per-diem snippet first.` |
| 3 | Chunking, embedding, vector store (full pipeline) | `python scripts/ingest.py` | 4 documents, 26 sections, 27 chunks, token max under 256, `stored in chroma`. Also include `src/rag/chunking.py` + `src/rag/embeddings.py` + `src/rag/store.py` in the editor if the terminal alone is not enough. |
| 4 | Basic RAG end to end | `python scripts/ask.py --mode hybrid "What is the international per diem for meals and incidentals?"` | Answer cites 95 USD and lists sources with document, section, version, status, char offsets. |
| 5 | Hybrid outperforms vector-only | `python scripts/compare_retrieval.py --query POL-EXP-004` | `dense` rank of POL-EXP-004 is > 5 (measured 11). `hybrid` rank ≤ 5. `DEMO:` line present. |
| 6 | Reranking code | Open `src/rag/rerank.py` and run `python scripts/compare_retrieval.py --query POL-EXP-004` | CrossEncoder pair scoring. Output has a `rerank` column/rank. |
| 7 | Evaluation test set and harness code | Open `tests/eval_set.yaml` (11 cases) and `tests/test_retrieval_eval.py` + `tests/test_generation_eval.py` | At least 8 questions. Expected section/keywords. Recall and accuracy tests visible. |
| 8 | Harness running, recall/accuracy results | `pytest -s tests/test_retrieval_eval.py tests/test_generation_eval.py` | `recall@5=1.00  MRR=0.93` and generation fixture test `PASSED`. |
| 9 | Planted issue, flawed answer, written diagnosis | `python scripts/diagnose.py` (rerank generation) plus `DIAGNOSIS.md` | Question about per diem. Rerank answer mentions **75 and 95**. Diagnosis: retrieval healthy, corpus at fault, grep lines shown. |
| 10 | Source attribution | `python scripts/ask.py --mode hybrid "How many unused PTO days can I carry into the next year?"` | Answer with `[S1]` (or similar) and a `sources:` block naming `pto_policy.md` § `POL-PTO-003`. |
| 11 | Passing pipeline | GitLab CI for this project, all jobs green | `install_dependencies`, `run_tests`, `check_formatting`, `check_types` passed. |

## Dry-run checklist

These should already have been run once locally before you capture:

```bash
python scripts/minimal_loop.py
python scripts/ingest.py
python scripts/compare_retrieval.py --query POL-EXP-004
python scripts/diagnose.py
pytest -s tests/test_retrieval_eval.py tests/test_generation_eval.py
flake8 . --max-line-length=100
mypy src/
```
