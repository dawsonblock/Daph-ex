# Erratum: Gate A's B2 arm understated BM25 retrieval

**Gate A's qualified claim is unaffected and still stands.**

## What changed

`hrm_adaptive_memory/retrieval/lexical.py::tokenize` originally used the
pattern `[A-Za-z0-9_./-]+`, which treats `.` and `-` as ordinary token
characters. A sentence-final entity therefore tokenized as `plan-000-965.`,
which never matches the same entity written mid-sentence (`plan-000-965`).
Every evidence record whose entity ended a sentence was invisible to lexical
queries.

The pattern is now `[A-Za-z0-9_]+(?:[./-][A-Za-z0-9_]+)*`: internal separators
are kept (`v1.2`, `src/main.py`, `Plan-000-965`) but trailing punctuation is
not part of the token. Regression tests live in
`tests/unit/test_gate_b_components.py`.

## Scope of impact on frozen Gate A evidence

The Gate A promotable result is `Q(B3) − Q(B0)`. Neither arm uses retrieval:
B0 has no evidence and B3 is fed oracle-labelled evidence directly. **The
qualified PASS (mean 0.998, LCB95 0.994) is therefore unaffected**, as are the
B1 and B1b control arms, which are drawn from the corpus by deterministic
hashing rather than by lexical scoring.

The affected number is the **descriptive** `B2−B0 = 0.60` reported in
`evidence/gate_a_report_v2r1.json` and the smoke/pilot runs. That figure
understates BM25: with the corrected tokenizer, BM25's complete-evidence-set
success on the same corpus rises from 0.618 to 0.818, and the
`numeric_derivation` family rises from 0.000 to 1.000 — that family's apparent
retrieval failure was entirely the tokenizer defect.

## Handling

Frozen Gate A evidence is **not** rewritten. The corrected retrieval
measurement is `evidence/gate_b_v2/`; the pre-fix Gate B pass is retained and
marked void at `evidence/gate_b_retrieval_v1/VOIDED.md`. Any citation of the
Gate A B2 number must reference this erratum.
