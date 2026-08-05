# controlled_gate_a_v2

Immutable controlled capability-use corpus for HRM Gate A (500 tasks, 100 per
family; 1200 evidence records). Rebuilt from `controlled-gate-a-v2` generator,
seed 3601.

Change from v1: the generator now rejects any question whose entity name
accidentally contains the gold answer token (v1 task `temporal_update-043`
carried its answer `587` inside `Service-043-587`). Leakage audit lives in
`evidence/controlled_corpus_audit_v2.json`.

Do not edit these files in place; build a new versioned directory instead.
