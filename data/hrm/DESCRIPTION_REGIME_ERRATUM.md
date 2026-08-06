# Erratum — the description regime is ambiguous by construction

Applies to `controlled_gate_c2_calibration_v1`, `controlled_gate_c2_chain_validation_v2`,
and `controlled_gate_c2_chain_validation_v3`. Corpora and receipts are retained
unmodified; nothing historical is rewritten.

## What the diagnostic found

A staged funnel over all 150 description tasks across the three corpora
(`evidence/gate_c2/description_diagnostic/`):

| stage | cal_v1 | chain_v2 | chain_v3 |
|---|---|---|---|
| identity record retrieved | 1.000 | 1.000 | 1.000 |
| description mention recognised | 1.000 | 1.000 | 1.000 |
| resolver returned an entity | 1.000 | 1.000 | 1.000 |
| **canonical entity correct** | **0.120** | **0.000** | **0.040** |

`ResolutionRate` is 1.000 and `ResolutionPrecision` is 0.00–0.12. The dominant
failure class is `AMBIGUOUS_MATCH` (36, 40, 39 of 50).

## Root cause: the reference does not identify a unique entity

`_DESCRIPTORS` holds only four phrases, reused across every task in a corpus.
So a single description string refers to many different entities:

| corpus | distinct description strings | entities per string |
|---|---|---|
| cal_v1 | 4 (for 50 tasks) | up to 13 |
| chain_v3 | 4 (for 50 tasks) | up to 13 |

The question *"Which band is held by the backup unit entered in the transfer
ledger?"* has **13 valid referents** in its own corpus. The identity record is
retrieved and states the correct mapping, but the descriptive phrase cannot
select among the thirteen records that all make the same claim about different
entities. No runtime resolver can succeed, because the information required to
disambiguate is absent from the question.

`alias` and `abbreviation` are not affected in the same way: their surfaces are
derived per entity, giving 40–42 distinct surfaces for 50 tasks.

## Consequence

The v3 description verdict (`OGC = 0.025` against an oracle gap of 0.800)
measured an **ill-posed regime**, not a mechanism limitation. The oracle arm
reached 1.000 only because it reads the true canonical entity from
`_oracle_metadata`, bypassing an ambiguity the runtime cannot resolve.

**Voided as a mechanism measurement:** every description-regime figure in
C2-C-v1, v2, and v3, and the description components of the earlier BGE policy
comparison.

**Unaffected:** canonical, abbreviation, and alias results, including the
triple-replicated chain-completion gains (abbreviation OGC 0.960, alias 0.815).
Those regimes have per-entity distinct surfaces.

## Why the earlier protocols missed it

v1 and v2 judged description with absolute deltas, where a near-zero gain read
as a near-miss (+0.08 against +0.10). Oracle-gap normalisation in v3 exposed the
magnitude (0.025 of 0.800), and only the staged funnel identified the cause.
Detecting it required separating *resolution rate* from *resolution precision* —
the resolver was always answering, and always wrong.

## Required fix before any description claim

Descriptions must be uniquely referring, e.g. by including a discriminating
attribute (`"the backup unit entered in the transfer ledger for Calcite"`), or by
drawing descriptors from a pool large enough that each is used once. A corpus
audit must then assert that every entity-regime surface resolves to exactly one
canonical entity.
