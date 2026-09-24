# Diagnosis: the per-diem answer is a corpus defect

Reproduce the evidence dump:

```bash
.venv/bin/python scripts/diagnose.py
```

The planted file is `data/raw/expense_policy_2022.md`: `version: 1.4`,
`status: archived`, ingested with no filter. It shares every `POL-EXP-*`
section code with `expense_policy.md` v3.0 and disagrees on the figures.

Ask: *What is the international per diem for meals and incidentals?*

`ask.py` defaults to rerank. The model answers:

> The international per diem … is either **75 USD** per day [S1] or **95 USD**
> per day [S2], depending on the version of the policy.

That looks like a confused model. It is not. Two questions, each answered with
evidence.

## 1. Was it retrieval?

No. Hybrid retrieval puts the current section at rank 1 and the archived copy
at rank 2. Rerank still has both in the top 2; it just swaps them.

| mode | rank | section | version | status | file |
| --- | --- | --- | --- | --- | --- |
| hybrid | 1 | POL-EXP-006 | 3.0 | current | expense_policy.md |
| hybrid | 2 | POL-EXP-006 | 1.4 | archived | expense_policy_2022.md |
| rerank | 1 | POL-EXP-006 | 1.4 | archived | expense_policy_2022.md |
| rerank | 2 | POL-EXP-006 | 3.0 | current | expense_policy.md |

Hybrid distances are 0.967 vs 0.968. Rerank probabilities (stored as
`1 - probability`) are 0.998 vs 0.997. Neither stage can tell the passages
apart, because both are genuinely about international per diem. The
cross-encoder has no `status` field; it only sees text.

The same pattern holds for the second planted conflict (pre-approval
threshold): hybrid ranks v3.0 then v1.4; rerank flips them. Gold POL-EXP-004
is in the top 2 either way.

For *retrieval* to be at fault, the gold chunk would have to be missing or
buried. It is rank 1 on the eval path (hybrid) and rank 2 on the interactive
path (rerank). **Retrieval is healthy.**

## 2. Was it the source data?

Yes. `grep` on the raw files, before anything is embedded:

```
expense_policy.md:66:      per diem of 95 USD per day
expense_policy_2022.md:64: per diem of 75 USD per day

expense_policy.md:43:      exceeding 500 USD requires
expense_policy_2022.md:46: exceeding 1,000 USD requires
```

Across the whole index, all seven `POL-EXP-*` sections exist in two versions.
That scan does not use a question; it is a property of the corpus.

For *generation* to be at fault, the prompt would have to contain one
consistent figure that the model contradicted. Hybrid's prompt contains 95 and
75; rerank's prompt contains 75 then 95. The model is handed an impossible
task: quote figures exactly as they appear, from sources that disagree. Hybrid
happens to follow [S1] (current, 95). Rerank follows both, because [S1] is the
archive. Either output is a faithful reading of a contradictory context.

**Default `ask.py` now prefers current.** Archived chunks are dropped before
generation, so the employee-facing answer is 95 USD. Reproduce the raw
failure with `python scripts/ask.py --include-archived "..."` or
`scripts/diagnose.py` (`prefer_current=False`). Filtering is the remedy;
the planted file stays in the index so the diagnosis is still greppable.

## What the detector does — and does not do

`src/rag/lineage.py` flags retrieved sources that share a `section_code` and
differ in `doc_version`. It does **not** drop the archived chunk. That is
deliberate: silently filtering would hide the defect that this write-up has to
show.

Default path (`DETECT_CONFLICTS=0`, `ask.py` without the flag): the answer
text is unchanged, so the graded run still produces the genuine conflict.

Opt-in (`DETECT_CONFLICTS=1` or `ask.py --detect-conflicts`): the same five
sources, plus:

```
LINEAGE WARNING: 2 section(s) retrieved in more than one document version.
Archived copies were not dropped.
  POL-EXP-006  versions 1.4, 3.0
    [S1]  expense_policy.md  v3.0  current
    [S2]  expense_policy_2022.md  v1.4  archived
```

Measured: source ids with the flag off and on are identical.

## Remedy

No retrieval constant fixes this. Hybrid already has the current chunk at
rank 1; rerank cannot prefer "current" because that fact is not in the
passage. The remedies are upstream:

1. Stop ingesting `status: archived` documents, or
2. Make ranking version-aware (prefer `current` when section codes collide), or
3. Delete `expense_policy_2022.md` from the source system.

(1) and (3) are data governance. (2) is a product rule, not a better embedder.
The opt-in warning is the smallest change that makes the failure visible
without pretending the pipeline resolved it.
