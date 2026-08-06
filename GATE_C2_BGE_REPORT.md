# Gate C2 — BGE dense arm

**Verdict: `PROMOTED_NARROWLY`.** `bge-small-en-v1.5` is promoted as the
preferred dense representation for **semantic-description OOD retrieval on V4**,
and as a candidate dense arm for Gate C2 fusion. It is **not** claimed to be
"the best retriever".

Receipts (each with `manifest.json`, `metrics.json`, `per_task.jsonl`,
`rankings.jsonl`, `RESULTS.sha256`):

- `evidence/gate_c2/retrieval/dense_bge_qualification/`
- `evidence/gate_c2/retrieval/dense_bge_ood/`
- `evidence/gate_c2/retrieval/dense_minilm_{qualification,ood}/`
- `evidence/gate_c2/retrieval/bm25_{qualification,ood}/`

Model pinned: `BAAI/bge-small-en-v1.5` @ `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`,
CLS pooling, L2-normalized, dim 384, max_seq 512, query prefix only.

## complete_set@50

| backend | qualification | OOD |
|---|---|---|
| bm25 | 0.350 | 0.048 |
| dense_minilm | 0.364 | 0.072 |
| dense_bge | 0.370 | 0.152 |

## Per entity regime — why the promotion is narrow

| regime | minilm | bge | delta |
|---|---|---|---|
| canonical | 0.404 | 0.400 | -0.004 |
| abbreviation | 0.324 | 0.340 | +0.016 |
| alias | 0.061 | 0.069 | +0.008 |
| description | 0.083 | 0.242 | +0.158 |

Essentially the whole OOD gain is the **description** regime. Alias barely moves.

## SEMANTIC_OOD vs REFERENTIAL_OOD

These are now treated as separate problems:

- **SEMANTIC_OOD** (`description`) — a natural-language phrase. A stronger
  encoder is the right instrument, and BGE delivers.
- **REFERENTIAL_OOD** (`alias`) — an arbitrary name substitution. No encoder can
  resolve it without retrieving the identity record, so this is
  **identity resolution plus multi-hop retrieval**, not an embedding problem.
  It requires the separate Gate C2-I ladder.

Promoting on aggregate OOD would have masked that distinction.

## Not claimed

- Qualification improvement (+0.006, within noise).
- Alias improvement (+0.008).
- Any downstream HRM effect — not measured here.
- Fusion performance — `rrf_bm25_bge` is not yet run.
