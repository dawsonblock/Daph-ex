# Gate C2 — development sweep (Phases 3–5)

**Development split only.** Qualification and OOD were not touched. Everything
chosen here is frozen before it is applied elsewhere.

Receipts: `evidence/gate_c2/dev_sweep/dev_sweep.json`

## Phase 3 — Q3 formulation sweep

| formulation | complete_set@50 | complete_set@10 | partial_proof@50 |
|---|---|---|---|
| Q3a bridge alone | 0.500 | 0.358 | 0.688 |
| Q3b bridge + relation | 0.625 | 0.450 | 0.898 |
| **Q3c subject + bridge + relation** | **0.900** | **0.592** | **0.949** |
| Q3d relation + bridge | 0.617 | 0.442 | 0.893 |
| Q3e two-query union | 0.617 | 0.408 | 0.865 |

**Frozen: `Q3c_subject_bridge_relation`.** Retaining the subject is worth +0.275
over bridge+relation and +0.400 over the bridge alone. This corroborates the
earlier Q0–Q3 ladder from the opposite direction, where querying the bridge
alone scored *worse* than the original question and collapsed to 0.000 on OOD.
The current mechanism's follow-up discards the subject; that is the largest
retrieval-side defect measured so far.

Q3 remains oracle-informed and is a diagnostic, not a production mechanism.

## Phase 4 — fusion arms and unique contribution

| arm | complete_set@50 | partial_proof@50 |
|---|---|---|
| P0 bm25 | 0.758 | 0.914 |
| P1 minilm | 0.533 | 0.731 |
| P2 bge | 0.592 | 0.782 |
| P3 rrf bm25+minilm | 0.783 | 0.919 |
| P4 rrf bm25+bge | 0.767 | 0.911 |
| P5 rrf three-way | 0.758 | 0.904 |

Unique gold contribution: `{'bge': 2, 'bm25': 25, 'minilm': 2}`.
Jaccard overlap: `{'bm25_vs_bge': 0.3492, 'bm25_vs_minilm': 0.3299, 'minilm_vs_bge': 0.3873}`.

## Phase 5 — RRF k sweep

| k | complete_set@50 |
|---|---|
| 10 | 0.767 |
| 30 | 0.767 |
| 60 | 0.767 |
| 100 | 0.767 |

**Flat.** k was never worth tuning on this corpus; k=10 wins only on a
complete_set@10 tiebreak. Recorded as a negative result.

## Why the retriever is NOT frozen here

The development split contains only the **canonical** and **abbreviation**
regimes — description and alias are held out for OOD by design. BGE's entire
measured value was the description regime (0.083 → 0.242 OOD). Development
therefore **cannot** select for the capability BGE provides, and freezing on
development complete-set alone would discard it for the wrong reason.

Gate C2-R stays `IN_PROGRESS_RETRIEVER_FREEZE_BLOCKED_BY_SPLIT_CONFOUND`.
Resolving it needs either a development split carrying all four regimes, or a
freeze rule that evaluates per regime rather than on aggregate coverage.
