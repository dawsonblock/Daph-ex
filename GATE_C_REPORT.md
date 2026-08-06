# Gate C report — bounded iterative retrieval

**Verdict: `FAIL_ITERATIVE_RETRIEVAL` — on one statistical criterion, not on mechanism.**

Five of six pre-declared checks pass. The mechanism reaches the oracle ceiling
with zero failures across all 500 tasks. It fails the sixth check because the
benchmark cannot certify a family-concentrated effect at family granularity —
a property of the corpus, not of the system. The bar was frozen before the
numbers were read and has not been moved.

## Arms

Model, revision, prompt condition (`direct`), composer, decoding, verifier, and
corpus are pinned identical to Gate A and Gate B, so every delta is
attributable to the retrieval/selection change alone. Receipts:
`evidence/gate_c/sprint2/`, SHA256 in `RECEIPTS.sha256`.

| arm | quality | complete-set | retrieval calls | slot-label echoes |
|---|---|---|---|---|
| `one_pass` (Gate B best) | 0.800 | 0.818 | 1.00 | 99 |
| `one_pass_selected` | 0.818 | 0.818 | 1.00 | **0** |
| `two_pass_selected` | **1.000** | **1.000** | 1.18 | 0 |
| `two_pass_calculate` | 1.000 | 1.000 | 1.18 | 0 |

Anchors: B0 (no evidence) = 0.002, B3 (oracle evidence) = 1.000.
**`two_pass_selected` equals the oracle ceiling.**

## Marginal utility of each capability

| capability | Δ quality |
|---|---|
| precision packing (`one_pass_selected − one_pass`) | **+0.018** |
| bounded follow-up (`two_pass_selected − one_pass_selected`) | **+0.182** |
| deterministic calculator (`two_pass_calculate − two_pass_selected`) | **+0.000** |
| combined | **+0.200** |

Per family, versus `one_pass`: `two_hop` +0.99, `numeric_derivation` +0.01,
and exactly 0.00 for `single_hop`, `temporal_update`, `distractor_heavy` — no
family regressed.

## Why the verdict is FAIL

| check | result |
|---|---|
| overall quality ≥ 0.90 | PASS (1.000) |
| `two_hop` complete-set ≥ 0.70 | PASS (1.000) |
| no family regression beyond 0.03 | PASS (worst 0.00) |
| near-duplicate confusion not increased | PASS (99 → 0 echoes) |
| retrieval cost bounded ≤ 2.0 calls | PASS (1.18) |
| paired LCB95 > 0 under **every** grouping | **FAIL** |

Applying Gate A's own rule — LCB95 > 0 under every declared grouping key, most
conservative view decides:

| grouping | groups | groups with effect | LCB95 |
|---|---|---|---|
| `family` | 5 | 2 | **+0.0000** |
| `template_id` | 15 | 4 | +0.0659 |
| `source_cluster_id` | 50 | 11 | +0.1080 |

The effect is real and large, but bridge structure exists in only one family of
five. A family-clustered bootstrap treats "which families exist" as the
sampling uncertainty, so roughly 8% of resamples contain no bridge-bearing
family at all and the 5th percentile pins to zero. This is not weak evidence of
a small effect; it is a benchmark that cannot express the effect's generality.

**The fix is a better corpus, not a lower bar.** `controlled_gate_a_v3` with
≥40 genuinely distinct templates, ≥20 real source styles, and bridge structure
spread across multiple families would make Gate C decidable — and is the same
corpus Gate A1 (structural generalisation) requires.

## Failure taxonomy

All 500 tasks classify as `NONE`. There is no remaining bridge-detection,
query-formulation, retrieval, packing, reader, or tooling failure to attribute
on this corpus.

## Three findings that constrain what comes next

**1. The slot-label echo was a precision artefact, not a reasoning limit.**
Gate B found HRM emitting `[E4]` instead of an answer on 99 tasks. Precision
packing alone drives that to **0** while raising quality only 0.018 — the
echoes were concentrated in tasks that were failing for retrieval reasons
anyway. This answers the F-arm question (packet label format) without needing
the ablation: the interface was not the problem, the confusable evidence was.

**2. CALCULATE has no measured value here.** The calculator produced answers on
all 100 `numeric_derivation` tasks and changed quality by **+0.000** — HRM was
already doing that arithmetic correctly. It must not be promoted to an action
on this evidence. Its safe-evaluator implementation is retained, unpromoted.

**3. Adaptive retrieval is not justified — a fixed policy suffices.** Of 91
follow-ups fired, **91 were positive, 0 neutral, 0 negative**. Under the Gate D
partition rule, near-universal benefit means a fixed two-pass policy is
correct and a learned trigger would have nothing to learn. Gate D should be
expected to FAIL on this corpus, and no controller training is authorised.

## The oracle decompositions were not run, and why

The I-arms (bridge-selection vs retrieval headroom) and P-arms (selector
ablation) were built and are committed, but running them on this corpus would
be uninformative: `two_pass_selected` already equals the oracle ceiling of
1.000, so I1 = I2 = I3 and every headroom term is exactly zero by
construction. They should run against `controlled_gate_a_v3`, where headroom
will exist.

## What this authorises

Nothing new. Gate C is not passed, so adaptive retrieval, executive training,
adaptive recurrence, Graphiti, RuVector, TurboVec, AgentDB, Infini
consolidation, and PixelRAG all remain blocked.

The next work is benchmark construction — `controlled_gate_a_v3` for Gate A1
and a re-runnable Gate C — not new mechanism.
