# Gate C1 on V4 — the retrieval path is near-inert, and coverage is why

**Verdict: `FAIL_MEASURED_ON_V4`.** With the mechanism pinned at `3260ce0` and
unchanged, bounded two-pass retrieval reaches **0.062** on qualification and
**0.016** OOD against an oracle-evidence ceiling of **0.880 / 0.908**.

Measured on a Colab T4 at pinned commit `3b1fbd5`, sequential generation
(`batch_equivalence_passed=false`, `batch_size=1`), so the result is clean.

> **Provenance limit.** These values are transcribed from the run output. The
> receipt files were not pushed back to this repository, so unlike every other
> gate they are **not independently re-verifiable here**. The Gate C2 coverage
> numbers below *are* re-verifiable — they were produced locally and committed.

## The ladder

| arm | qual quality | qual complete-set | OOD quality |
|---|---|---|---|
| `R0` one-pass | 0.074 | 0.162 | 0.004 |
| `R1` current two-pass | 0.062 | 0.134 | 0.016 |
| `R2` oracle bridge identity | 0.054 | 0.134 | 0.008 |
| `R3` oracle bridge + relation | 0.054 | 0.134 | 0.008 |
| `R4` oracle query + oracle selection | 0.462 | 0.168 | 0.404 |
| `R5` oracle evidence | **0.880** | 1.000 | **0.908** |

Iteration is now *negative* on qualification (`R1−R0 = −0.012`), and handing
the mechanism a perfect bridge makes it slightly worse still (`R2−R1 = −0.008`).
A retrieval step that cannot find evidence only adds noise.

## The R3 → R4 jump is not a ranking result

`R4` gains **+0.408**, and the obvious reading is "selection is the bottleneck".
The complete-set column contradicts that:

- `R4` complete-set success is **0.168**, but its quality is **0.462**.

`R4` keeps only the required records that retrieval actually found. It therefore
answers many tasks from an **incomplete but perfectly clean** packet. So the
gain is not better ranking — it is *the removal of every non-gold record*. Two
things follow, and they point in different directions:

1. Any distractor in the packet is close to catastrophic.
2. The reader can often succeed on partial gold evidence.

`R3→R4` must not be cited as a pure selection-ranking term.

## Coverage measured directly, without the reader

Answer accuracy cannot separate "retrieval never found it" from "the reader
could not use it", so coverage was measured against evaluator-only ground truth
read from each task's proof graph. Receipts: `evidence/gate_c2/`.

**Complete-evidence-set success @50:**

| retriever | qualification | OOD |
|---|---|---|
| BM25 | 0.350 | 0.048 |
| dense (pinned MiniLM) | 0.364 | 0.072 |
| union (interleaved) | 0.422 | 0.128 |
| RRF | **0.426** | **0.128** |

At @10, BM25 on OOD is **0.000** — it finds the complete evidence set for zero
of 250 tasks.

### This reverses Gate B

Gate B concluded that BM25 dominates the tested dense stack. **That does not
survive V4**: dense beats BM25 on both splits, and fusion beats either alone.
The Gate B ordering was a property of the identifier-heavy v2 corpus, where
exact-token joins were unusually easy — exactly the shortcut V4 removes. The
Gate B claim is now scoped accordingly.

## What this means for the sprint

Even the best retriever leaves complete-set success at 0.426 / 0.128. Coverage,
not ranking, is the dominant constraint, and it is upstream of everything else:
a selector cannot recover a record that was never retrieved, and `R4` shows the
reader is not the limit.

Sequence follows from that: raise coverage first (fusion is already worth
+0.076 qualification and +0.080 OOD over BM25 at @50), then attack selection on
frozen candidate pools, and only then ask whether semantic gap reasoning is
justified — the `Q0–Q3` query-only ladder answers that without spending GPU.

## Still blocked

Executive, adaptive retrieval, adaptive recurrence, Graphiti, RuVector,
TurboVec, procedural memory, latent memory. Gate D remains unmeasurable.
`CALCULATE` remains unqualified.
