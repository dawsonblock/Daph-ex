# Gate C1 report — structural generalization of bounded iterative retrieval

> **AMENDED after publication.** Four corpus defects were found after this
> evaluation ran: evidence-level source-style contamination in both directions,
> an alias regime that is really prefix truncation, description-regime tasks
> that are **unanswerable from evidence** (0 of 120 have their subject anywhere
> in their own evidence, so 48% of the OOD split is impossible for any
> retriever), and a non-oracle `oracle_bridge` arm.
>
> **All OOD numbers below are voided as quantitative measurements.** The
> qualification decomposition stands (qualification has no description regime).
> The qualitative conclusion also stands, because it is provable from code and
> corpus text alone: the entity extractor matches 0 of 250 OOD questions and 0
> of 160 `natural_name` qualification questions, so zero follow-ups fired.
>
> **Also amended:** the `bridge` (+0.146) and `retrieval` (+0.288) terms below
> are *raw arm differences, not validated causal attributions*. Both depend on
> an `oracle_bridge` arm now known not to be independent of the extractor under
> test. The arm scores stand as historical measurements; the causal labels do
> not. Only V4's independent R0–R5 ladder can establish the real decomposition.
>
> What remains strong from this experiment is simpler: the mechanism materially
> improves some qualification tasks, `natural_name` tasks expose its syntactic
> brittleness, `ENTITY_PATTERN` is not semantic bridge inference, and HRM
> performs substantially better given correct evidence.
>
> Details: `data/hrm/controlled_gate_a_v3/V3_KNOWN_LIMITATIONS.md`.
> Superseded by `controlled_gate_a_v4`.

**Verdict: `FAIL_STRUCTURAL_GENERALIZATION`.** The mechanism that reached the
oracle ceiling on v2 does not survive v3, and on the out-of-distribution split
it contributes **nothing at all**.

Mechanism pinned at `3260ce0` and unchanged. Corpus frozen at `45b3c02` before
evaluation. Receipts: `evidence/gate_c/v3_{qualification,ood}/`, SHA256 in
`RECEIPTS.sha256`.

## Results

| arm | qualification (500) | OOD (250) |
|---|---|---|
| `one_pass` | 0.246 (css 0.464) | 0.080 (css 0.188) |
| `one_pass_selected` | 0.294 (css 0.450) | 0.080 (css 0.148) |
| `two_pass_selected` | 0.394 (css 0.586) | 0.080 (css 0.148) |
| `two_pass_calculate` | 0.386 (css 0.586) | 0.080 (css 0.148) |
| `oracle_bridge` | 0.540 (css 0.800) | 0.080 (css 0.148) |
| `oracle_evidence` | **0.828** (css 1.000) | **0.764** (css 1.000) |

For scale: the same mechanism scored **1.000** on v2.

## The four questions

**1. How much is query/bridge inference?**
Qualification: **+0.146** (`oracle_bridge − two_pass_selected`). A perfect
bridge choice is worth more than the entire iteration mechanism delivers.
OOD: unresolved — see the instrument limitation below.

**2. How much is retrieval once the query is known?**
Qualification: **+0.288** (`oracle_evidence − oracle_bridge`). This is now the
single largest recoverable term.
OOD: **+0.684**, but this bundles bridge, retrieval, and selection together
because all three arms are inert there.

**3. How much is evidence selection / context presentation?**
Qualification: **+0.048** on quality, while *lowering* complete-set recovery
0.464 → 0.450. Selection improves answers by discarding confusable records even
as it discards some required ones.
OOD: **0.000** on quality and **−0.040** on complete-set recovery — pure loss.

**4. How much remains when HRM receives perfect evidence?**
Qualification reader error **0.172**; OOD **0.236**. This is error relative to
perfect task accuracy under the current model and prompt. It is *not* headroom
belonging to any retrieval component and must not be added to the terms above.

## Root cause of the OOD collapse

The entity extractor finds **nothing** on OOD:

| | questions yielding an entity | evidence records yielding an entity |
|---|---|---|
| OOD | **0 / 250** | **0 / 983** |

`ENTITY_PATTERN` requires a hyphen-digit suffix (`Atlas-317`). OOD uses aliases
(`Nimbus As`) and descriptions (`the auxiliary unit listed in the intake
record`). With no entities there are no bridges, so **zero follow-ups fired on
all 250 OOD tasks**, and anchoring had no anchors. The retrieval stack is not
degraded on OOD; it is **inert**.

This is already visible inside qualification: the `natural_name` regime yields
**0 / 160** entities. A third of the qualification split is running with the
entity machinery switched off, which is part of why qualification is weak too.

The mechanism was never doing bridge inference. It was doing lexical
identifier chaining, and v3 removed stable lexical identifiers.

## Instrument limitation — the OOD bridge/retrieval split is not yet measurable

`oracle_bridge` (I2) derives the "true" bridge by re-running the same
`extract_entities` over the required records. Where extraction fails, the
oracle arm fails identically to the deterministic one — which is exactly what
happened on OOD, where I2 fired zero follow-ups and scored identically to I1.

So the OOD `bridge = 0.000` figure means **"both arms are inert"**, not
"bridge inference is fine". A genuine oracle must read the latent bridge
identity recorded by the generator rather than re-deriving it from surface
text. Until that is fixed, only the qualification decomposition is
interpretable, and the OOD `retrieval = 0.684` term is a combined
bridge+retrieval+selection quantity.

## Gate D — still not measurable, for a new reason

| | always one-pass | always two-pass | oracle trigger | opportunity |
|---|---|---|---|---|
| qualification | 0.294 | 0.394 | 0.394 | **+0.000** |
| OOD | 0.080 | 0.080 | 0.080 | **+0.000** |

Follow-up delta partition on qualification: **50 positive, 450 neutral, 0
negative** — including 0 negative within `C_SECOND_PASS_CONFUSING`, the group
built specifically to make a second pass harmful.

On v2, Gate D was unmeasurable because the action always helped (91/91
positive). On v3 it is unmeasurable because the action *rarely fires* (211 of
500) and still never hurts. v3 successfully created heterogeneous task
structure but did not yet produce heterogeneous *outcomes*, because a follow-up
that never fires cannot do harm.

**No controller is justified.** A learned trigger has nothing to learn: the
best fixed policy already equals the oracle policy on both splits.

## Findings that carry forward

- **CALCULATE remains unqualified and is now mildly harmful** (−0.008
  qualification). It stays unpromoted.
- **Slot-label echoes track evidence quality, not packet format.** Qualification
  echoes fall 85 → 28 under precision packing and to **3** under oracle
  evidence; OOD falls 53 → 0 under oracle evidence. Echoes are a symptom of
  reader uncertainty given bad evidence. The F-arms would be measuring a
  symptom, so they remain implemented and unrun.
- **The reader is not the bottleneck.** With perfect evidence HRM scores 0.828
  and 0.764 on a corpus with aliases, descriptions, four unseen source styles,
  and non-numeric answers. The gap to that ceiling is retrieval's.

## What this authorizes

Nothing downstream. Gate C1 fails, so adaptive retrieval, executive training,
adaptive recurrence, Graphiti, RuVector, TurboVec, AgentDB, consolidation, and
PixelRAG all remain blocked.

The next mechanism is an **information-gap / query-formulation layer** that
does not depend on surface identifier shape: construct an explicit reasoning
state (known, target, missing relation, candidate bridges) and issue a query
for the missing *relation on the bridge*, not for the bridge's name.

It has a measured target: **0.394 → 0.540** on qualification (`+0.146`, the
bridge term), and on OOD the whole **0.080 → 0.764** span is currently
unclaimed. Two prerequisites before that work starts:

1. Fix the `oracle_bridge` instrument to use latent bridge identity, so the
   OOD bridge/retrieval split becomes measurable.
2. Replace entity anchoring with relation-aware connectivity, since anchoring
   is a pure loss wherever identifiers are not lexically stable.
