# RAG Policy Assistant — Architecture Walkthrough

Target length: 15 minutes plus questions. Show
`rag_pipeline_architecture.png` from slide 4 onward and point at boxes as you
go; do not put the diagram up before you have framed the problem.

---

## 1. What this is

An internal Q&A assistant over company policy documents. Ask it a question in
plain English, it answers from the policy corpus and cites which document and
section the answer came from.

The interesting part is not that it answers questions. It is that the pipeline
is built to tell you *why* it got an answer wrong, which is the part that
usually gets skipped.

**Say:** "I'll spend most of this on three decisions — chunk size, two-stage
retrieval, and lineage metadata — because those are the ones that determine
whether the thing works."

---

## 2. Why this is harder than it looks

Two naive approaches and why each fails:

- **Keyword search only.** Someone asks about "daily meal allowance abroad";
  the policy says "international per-diem reimbursement". Zero words in common.
  `grep` returns nothing.
- **Embed each whole document.** An embedding is a single point in space.
  Average the per-diem rates, lodging caps, approval thresholds, and receipt
  rules of an entire expense policy and you get a point meaning "expenses,
  generally" — weakly near every expense question and strongly near none.

So: split documents into pieces small enough to be about one thing, and search
them with more than one method.

---

## 3. The corpus

Four policy documents, generated for this exercise, 500-800 words each:

| document | version | status |
|---|---|---|
| `remote_work_policy.md` | 2.1 | current |
| `expense_policy.md` | 3.0 | current |
| `pto_policy.md` | 2.0 | current |
| `expense_policy_2022.md` | 1.4 | **archived** |

26 sections, 27 chunks. Every section carries a code — `POL-EXP-004` — which
matters later.

The fourth document is a deliberate defect: a stale copy of the expense policy
that contradicts the current one on five sections. Per-diem $75 against $95.
Approval threshold $1,000 against $500. Its body reads like a perfectly normal
policy; only the front matter says `status: archived`. We ingest it without
filtering, which is exactly what a real company does by accident.

**Say:** "I planted that on purpose, and I'll come back to it, because it
produces the failure mode that actually bites you in production."

---

## 4. Architecture

Put the diagram up. Two phases: ingestion runs once offline, the query path
runs per question. Walk left column top to bottom, then right column.

Do not explain every box. Say: "Six stages in, six stages out. I'll go deep on
three of them."

---

## 5. Ingestion

Parse, chunk, validate, embed, store.

Documents are split on `##` headings first, because the author already marked
the semantic boundaries — a heading is a free, human-authored chunk edge, and it
hands us the section code and title as metadata for nothing. Only sections
longer than the chunk size get windowed; 25 of our 26 sections become exactly
one chunk.

The embedding model turns each chunk into 384 floating-point numbers. That is
all it does. It does not read, answer, or understand — it is a text-to-vector
function, trained so that similar meaning lands nearby. Think of it as the
filing system, not the librarian.

---

## 6. Chunk size, and the trap

**160 words, 32-word overlap.** The number is derived, not chosen.

`all-MiniLM-L6-v2` truncates its input at 256 wordpieces. Policy prose
tokenizes worse than ordinary English — `$95.00`, `POL-EXP-004`, `2026-01-01`
each explode into several wordpieces — so 160 words lands around 210-230,
leaving headroom.

**This is the slide to linger on.** If a chunk exceeds the cap, the tail is
never embedded and **no error is raised**. Retrieval silently degrades with no
traceback and no log line. You will spend a day blaming your retriever.

So the rule is: derive chunk size from your embedding model's
`max_seq_length`, and assert it in a test. We hard-fail at ingest if any chunk
exceeds 256 wordpieces.

Overlap is about boundaries. A window can land mid-sentence:

```
window 1: "... any booking exceeding $500 requires written manager"
window 2: "approval prior to purchase. Approval must be secured ..."
```

Neither chunk contains the complete rule. 32 words of overlap is longer than a
typical sentence, so the full rule survives intact in at least one chunk.

---

## 7. Hybrid retrieval — why two searches

Each method fails exactly where the other succeeds.

- **Dense (vector) search** handles paraphrase. "Daily meal allowance abroad"
  finds "international per-diem" with no shared words.
- **Keyword search** handles rare literals. Ask for `POL-EXP-004` and the
  embedding model has no useful representation for it — it never saw that code
  in training, so it returns something topically plausible and confidently
  wrong. A regex finds it instantly.

We run both, take 20 candidates from each.

**Say:** "This is the single clearest argument for hybrid search, and it's why
a policy corpus specifically needs it — policy documents are full of codes,
section numbers, and form names."

---

## 8. Merging two incompatible lists

Dense search returns distances (0 to 2, lower is better). Keyword search
returns hit counts (integers, higher is better). There is no principled
conversion between them, and min-max normalising shifts with every query.

Reciprocal Rank Fusion throws the scores away and keeps only position:

```
score(chunk) = Σ  1 / (60 + rank)
```

Worked example, and the point of the constant:

```
chunk_A  rank 1 dense, rank 3 keyword  →  1/61 + 1/63  = 0.0323
chunk_B  rank 2 dense, absent keyword  →  1/62         = 0.0161
```

Without the 60, rank 1 scores 1.0 and rank 2 scores 0.5 — a cliff where one
retriever's top hit dominates everything. With it, rank 1 and rank 3 differ by
3%, so position barely matters *within* a list and what matters is appearing in
*both*. RRF rewards agreement between methods that fail in uncorrelated ways.

**If asked about weighted fusion (0.7 / 0.3):** that is an alternative to RRF,
not a stage before it. Weighted blending needs comparable scales; RRF exists
precisely because they are not comparable. Pick one.

---

## 9. Reranking — what it is, and a measured result that surprised me

**The mechanism.** The bi-encoder embeds each chunk at ingest, *before any
question exists*, so at query time you compare two summaries that never saw
each other. The cross-encoder embeds nothing: question and chunk go through the
network **together in one forward pass**, so every token of the question can
attend to every token of the chunk. Output is a single relevance score per
pair. It is reading them side by side rather than comparing fingerprints.

Why not use it for everything: nothing can be precomputed, so scoring the whole
corpus costs one forward pass per chunk per query. Hence two stages —

```
27 chunks → [fast, precomputed]  → candidates → [slow, accurate] → 5 → LLM
             optimise RECALL                     optimise PRECISION
```

**Now the measured result.** Question: *"Do I need approval before booking
international flights?"*, scored against six sections.

```
bi-encoder cosine                          cross-encoder logits
  1. POL-EXP-004  Pre-Approval     0.42      1. POL-PTO-002  Requesting Time Off  +3.74
  2. POL-RMT-006  Working Abroad   0.35      2. POL-RMT-006  Working Abroad       +1.35
  3. POL-EXP-006  Intl Travel      0.27      3. POL-EXP-004  Pre-Approval         +0.51
  4. POL-PTO-002  Requesting TO    0.23      4. POL-EXP-006  Intl Travel          -4.93
```

The bi-encoder ranked the correct section first. **Reranking demoted it to
third.** The cross-encoder promoted the PTO section, because `POL-PTO-002`
contains "require manager approval before travel is booked" — a near-verbatim
paraphrase of the question.

**Say this out loud, it is the most useful point on the slide:** the
cross-encoder is a very strong matcher of question-answer *phrasing*, and that
is also its failure mode. It matched the shape of the question and had no idea
that leave-request approval and expense pre-approval are different policy
domains. It judges "does this look like an answer to this question", not "is
this the right policy".

**What that means for the design.** Reranking is not a free win. It is the
correct architecture at scale, where stage one must cheaply narrow thousands of
candidates, but on a 27-chunk corpus where the bi-encoder is already nearly
exhaustive it can actively hurt. So we measure it: `compare_retrieval.py`
produces a three-way table (vector-only, hybrid, hybrid+rerank) across the
eval set, and if reranking does not earn its place here, that is the finding.

**Honest note on refusal.** I expected cross-encoder scores to give a clean
"nothing relevant exists" signal, since cosine distances cannot. Measured, an
off-topic question still scored +0.397 (probability 0.60) against a correct
answer's 0.63 — overlapping. So the refusal threshold is a coarse guard that
excludes clearly irrelevant chunks, not a calibrated decision. A purpose-built
answerability model would be better founded.

---

## 10. Attribution and lineage — the chunk contract

One dataclass is the interface between every stage:

```python
chunk_id, text, doc, doc_version, status, effective_date,
section_code, section_title, char_start, char_end,
word_count, token_count
```

`char_start` / `char_end` fall out of the windowing loop for free, so an answer
can cite exact source text rather than a whole section. It also gives a test
worth having: `raw[char_start:char_end] == chunk.text`, which proves the cited
offsets are real and nothing was mangled at ingest.

`doc_version` and `status` are the fields that make the next slide possible.

Answers come back as: `expense_policy.md § POL-EXP-004 (v3.0, current, chars
1204-2033)`.

---

## 11. The planted failure, and how you diagnose it

Ask: *"What's the per-diem for international travel?"*

The system answers **$75** — confidently, fluently, and wrong. The current rate
is $95.

This looks exactly like a hallucination. It is not. Two questions, each
answered with evidence:

**Was it retrieval?** Dump the ranked results. The expected chunk is at rank 1
with a strong score; the archived one is at rank 2. Both are in the index,
validation passed, spans round-trip. The retriever found precisely what it was
asked to find. **Retrieval is healthy.**

**Was it the source data?** `grep` the raw files, before anything is embedded:

```
expense_policy.md:      per diem of 95 USD per day
expense_policy_2022.md: per diem of 75 USD per day
```

Two documents, two numbers, both on disk. **The corpus is the fault.**

Why that is airtight rather than merely plausible: for *generation* to be at
fault, the context would have to contain one consistent answer that the model
contradicted. It contains two mutually exclusive facts, so any answer is either
wrong or hedged — the model was handed an impossible task. For *retrieval* to
be at fault, the gold chunk would have to be missing or buried. It is rank 1.

And the fix follows from the diagnosis: no amount of retrieval tuning helps,
because retrieval works. The remedies are all upstream — filter on `status`,
make ranking version-aware, or remove the stale document from the source
system. The real fix is data governance.

**Say:** "The symptom pointed at the last stage. The cause was at the first.
That is the whole reason lineage metadata is on every chunk."

---

## 12. What we measure

Ten fixed questions with known gold chunks, run in CI.

- **recall@5** — did the gold chunk reach the final five? This is the
  unrecoverable failure: if the answer is not in the prompt, no model and no
  prompt engineering recovers it. It is the ceiling on everything downstream.
- **MRR** — `1/rank` of the gold chunk. Measures ordering, which matters
  because LLMs weight earlier context more heavily.

Both, because they isolate different faults. Recall 1.0 with MRR 0.45 means you
always find it and always bury it — a reranking problem. Recall 0.6 with MRR
0.95 means you nail it when you find it but miss it 40% of the time — a
retrieval problem.

Plus chunk-level invariants: token budget, gold answer fits inside one chunk,
spans round-trip. And answer-accuracy checks against recorded answers, so they
still run on a CI runner that has no LLM.

**Deliberately not measured:** NDCG (with one gold chunk and binary relevance
it is a monotone transform of MRR — reporting both looks rigorous and adds
nothing) and LLM-as-judge (nondeterministic, and unavailable in CI).

---

## 13. Honest limitations

Lead with these rather than waiting to be caught.

- **27 chunks is small.** With 10 eval cases, recall@5 moves in steps of 0.1.
  The thresholds are regression guards, not performance claims.
- **At this size you arguably don't need RAG at all.** The whole corpus is
  ~12k tokens and the model has a 32k window. Pasting everything into the
  prompt would probably answer *better*. The reasons to build the pipeline are
  scale and attribution, not this corpus.
- **`CANDIDATE_K = 20` over 27 chunks** means the first stage filters very
  little and the cross-encoder is doing most of the ranking. Honest fix: lower
  it, or state plainly that this is an exhaustive rerank at this scale.
- **ChromaDB is demo-to-mid scale.** Pinecone or pgvector in production.
- **Substring matching, not BM25.** Fine for 27 chunks and exact codes;
  BM25 is the upgrade once term frequencies matter.
- **Refusal uses a hand-tuned cross-encoder threshold**, and measurement shows
  it cannot separate a correct answer (0.63) from an off-topic question's best
  candidate (0.60). A calibrated answerability model would be better founded.
- **Reranking is unproven on this corpus** and on one measured query it made
  ranking worse. It stays in the design because it is correct at scale, and
  the three-way comparison will report whether it helps here.

---

## 14. Three things worth taking away

1. **Derive chunk size from your tokenizer, then assert it in a test.**
   Silent truncation has no error message.
2. **Retrieval is two jobs, not one.** Optimise recall cheaply, then precision
   expensively. Stage one must not lose the answer; stage two puts it on top.
3. **Put lineage on every chunk.** `version` and `status` are what turn "the
   model hallucinated" into "the corpus contains two contradictory truths, here
   is the grep." Most RAG debugging pain is data-quality pain wearing a
   retrieval costume.
