# Gate B report — can practical retrieval recover the evidence HRM can use?

**Verdict: `PASS_RETRIEVAL_EXPANSION`**

bm25 recovers complete evidence sets on 81.8% of tasks and lifts answer quality +0.798 over B0.

Gate A proved HRM converts correct evidence into correct answers
(B3−B0 = +0.998). Gate B asks whether a
real retriever can find that evidence. Model, prompt condition
(`direct`), packing, decoding, verifier, and corpus are pinned
identical to Gate A, so every difference below is attributable to retrieval alone.

- Tasks: 500 · evidence records: 1200 · k = 10
- Corpus digest: `4ee67dcad8d153b2…` / `f8ea20353bed006f…`
- Anchors from `evidence/gate_a/qualified_run_002/gate_a_report_v2r1.json`: B0 = 0.002, B3 = 1.000

## Arms

| arm | CompleteSet | ReqEvRecall | R@10 | MRR | nDCG | irrelevant tok | latency ms | index s | downstream Q | Δ vs B0 | oracle gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bm25` | **0.818** | 0.909 | 0.909 | 1.000 | 0.927 | 0.875 | 1.3 | 0.0 | 0.800 | +0.798 | 0.200 |
| `hybrid_score` | **0.606** | 0.740 | 0.740 | 0.838 | 0.750 | 0.911 | 26.7 | 1.1 | 0.598 | +0.596 | 0.402 |
| `hybrid_rerank` | **0.600** | 0.711 | 0.711 | 0.788 | 0.716 | 0.919 | 27.0 | 1.2 | 0.576 | +0.574 | 0.424 |
| `hash` | **0.566** | 0.671 | 0.671 | 0.556 | 0.536 | 0.925 | 11.5 | 0.0 | 0.538 | +0.536 | 0.462 |
| `hybrid_rrf` | **0.546** | 0.691 | 0.691 | 0.712 | 0.644 | 0.914 | 26.8 | 1.2 | 0.502 | +0.500 | 0.498 |
| `dense` | **0.360** | 0.474 | 0.474 | 0.332 | 0.329 | 0.940 | 25.5 | 1.4 | 0.326 | +0.324 | 0.674 |

## Complete evidence-set success by family

A retriever that finds one of two required records has not made a two-hop task
solvable, so this — not Recall@k — is the decisive multi-hop measure.

| arm | `single_hop` | `temporal_update` | `distractor_heavy` | `two_hop` | `numeric_derivation` |
|---|---|---|---|---|---|
| `bm25` | 1.000 | 1.000 | 1.000 | 0.090 | 1.000 |
| `hybrid_score` | 1.000 | 0.990 | 1.000 | 0.040 | 0.000 |
| `hybrid_rerank` | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| `hash` | 0.930 | 0.980 | 0.920 | 0.000 | 0.000 |
| `hybrid_rrf` | 1.000 | 0.820 | 0.870 | 0.040 | 0.000 |
| `dense` | 0.720 | 0.560 | 0.500 | 0.020 | 0.000 |

## Failure attribution

Best arm (`bm25`) — 100 failures of 500 tasks; 91.0% of failures are retrieval-bound.

| family | `CALCULATION_FAILURE` | `REASONING_FAILURE` | `RETRIEVAL_FAILURE` |
|---|---|---|---|
| `single_hop` | 0 | 0 | 0 |
| `temporal_update` | 0 | 0 | 0 |
| `distractor_heavy` | 0 | 0 | 0 |
| `two_hop` | 0 | 8 | 91 |
| `numeric_derivation` | 1 | 0 | 0 |

## Why complete evidence was not always enough

Gate B's best arm answered only 1 of 9 `two_hop` tasks whose evidence was
*fully* retrieved, while Gate A's oracle arm scored 100/100 on that family.
The difference is not the amount of evidence. Holding the required evidence
present in every condition and varying one factor at a time
(`scripts/diagnose_gate_b_packing.py`, N = 100 per cell):

**Packet size** — no effect:

| packet | quality | slot-label echoes |
|---|---|---|
| 2 records | 1.000 | 0 |
| 3 records | 1.000 | 0 |
| 5 records | 1.000 | 0 |
| 10 records | 1.000 | 0 |

**Position of the required evidence** — no effect:

| oracle position | quality | slot-label echoes |
|---|---|---|
| first | 1.000 | 0 |
| last | 1.000 | 0 |
| middle | 1.000 | 0 |

**Distractor similarity** — this is the mechanism:

| distractor kind | quality | slot-label echoes |
|---|---|---|
| `bm25_top_k` | 0.390 | 61 |
| `random_corpus` | 1.000 | 0 |
| `same_template` | 0.670 | 33 |

With unrelated padding the model is perfect; with near-duplicate records that
differ only in their entity identifiers it collapses, and its characteristic
failure is emitting an evidence slot label (`[E4]`) instead of a value.

**Retrieval precision, not just recall, is a binding constraint.** A retriever
that returns more lexically similar material can lower answer quality even
when it raises recall — which is also why the hybrid arms, which surface more
look-alike records, underperform BM25 here.

## The seven Gate B questions

1. **Which backend has the highest complete evidence-set recall?**
   `bm25` at 0.818.
2. **Does dense beat BM25?**
   No — dense 0.360 vs BM25 0.818.
3. **Does hybrid beat each individual backend?**
   No — best hybrid `hybrid_score` 0.606 vs BM25 0.818.
4. **Are two-hop failures still retrieval-bound?**
   Complete-set success on `two_hop` is 0.090 for the best arm.
5. **Are numeric failures retrieval-bound or reasoning-bound?**
   Complete-set success on `numeric_derivation` is 1.000 for the best arm.
6. **What does retrieval cost?**
   See the latency, index-time, and irrelevant-token columns above.
7. **Is iterative retrieval justified?**
   Yes — see the two-hop gap.

## What this authorizes

`BOUNDED_ITERATIVE_RETRIEVAL`. Still blocked pending their own gates:
`macro_executive_training`, `micro_compute_controller`, `adaptive_recurrence`, `graphiti_temporal_memory`, `external_vector_engines`, `transactional_persistent_memory`.

Two constraints must be addressed together in the next stage, because
optimizing either alone is measurably counterproductive here:

1. **Bridge-entity recovery.** A single-pass retriever cannot know the entity
   that links hop one to hop two, so `RETRIEVE_FOLLOWUP` has a concrete,
   measured opportunity on `two_hop`.
2. **Evidence selection.** Simply retrieving more raises recall while lowering
   answer quality through distractor confusion. Redundancy control and
   near-duplicate suppression belong in the same stage as iterative retrieval,
   not deferred to a later packing stage.
