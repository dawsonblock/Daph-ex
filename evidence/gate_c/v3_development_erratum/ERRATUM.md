# Development-split erratum — v3 revision e648569 measured JSON formatting, not evidence use

## What was measured

The first v3 build (corpus frozen at commit `e648569`) was evaluated on the
**development split only**, with the mechanism pinned at `3260ce0`:

| arm | quality | complete-set |
|---|---|---|
| `one_pass` | 0.225 | 0.525 |
| `one_pass_selected` | 0.217 | 0.467 |
| `two_pass_selected` | 0.283 | 0.592 |
| `two_pass_calculate` | 0.267 | 0.592 |
| `oracle_bridge` | 0.508 | 0.825 |
| `oracle_evidence` | 0.692 | 1.000 |

Receipts in this directory are the run as it happened, retained unmodified.

## The defect

Under `oracle_evidence` — a perfect evidence set, perfectly selected — accuracy
by answer kind was:

| answer kind | correct |
|---|---|
| numeric | 30/30 |
| symbolic | 28/30 |
| enum | 22/28 |
| boolean | 3/4 |
| **json** | **0/28** |

Inspection showed the model extracting the correct value and failing only to
reproduce a JSON wrapper the question never requested: gold
`{"state": "provisional"}` against output `provisional`. That measures
formatting compliance, not evidence use, and it is not the research question.

## The correction

A `json_field` answer is now the bare value, while the evidence renders that
value inside a JSON object. The task therefore tests reading a value out of a
JSON-structured record. The corpus was rebuilt and re-frozen at `45b3c02`.

## Why this was legitimate

The change was made from **development-split evidence only**. The qualification
and OOD splits had not been evaluated when the defect was found, and had still
not been evaluated when the corrected corpus was frozen. Development exists for
exactly this purpose.

**Standing rule from here:** once qualification or OOD has been evaluated, they
are not modified. If a defect is later found in either, that evaluation is
voided in full rather than adjusted.

## Findings that survive the correction

These do not depend on the json defect and were reproduced by design:

- Bridge/query inference is the dominant bottleneck: `oracle_bridge` − `two_pass_selected` = **+0.225**.
- Retrieval after a correct query still has headroom: `oracle_evidence` − `oracle_bridge` = **+0.184**.
- Entity-anchored precision packing **regressed** on v3 — complete-set recovery
  0.525 → 0.467 — because v3 deliberately breaks explicit entity identity.
- The deterministic calculator was actively harmful (−0.017); it stays unqualified.
- Evidence slot-label echoes returned, meeting the F-arm trigger condition.
