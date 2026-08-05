# Evidence index

Every scientific run is preserved, including voided and failed ones. Nothing
in this tree is rewritten after the fact; corrections are recorded as new runs
plus an erratum. Files *inside* a run directory keep the paths recorded at run
time, so a manifest may reference a pre-reorganisation path — the mapping is
below.

## Layout

```
evidence/
├── gate_a/                          controlled evidence use (Gate A0)
│   ├── voided_run_001/              first 500-task qualification — VOIDED
│   ├── qualified_run_002/           the qualified run + gate_a_report_v2r1.json
│   ├── ERRATA.md                    tokenizer erratum affecting the B2 figure
│   ├── protocol_manifest_v2.json    the frozen protocol actually used
│   ├── hrm_smoke_v2_{direct,cot,synthcot}/   prompt-condition selection
│   ├── hrm_pilot_v2_direct/         100-task pilot
│   └── controlled_corpus_audit_*.json        leakage audits
├── gate_b/                          single-pass retrieval
│   ├── voided_pretokenizer_fix_run_001/   measured with the defective tokenizer
│   ├── retrieval_only_run_002/      corrected retrieval-only pass
│   ├── qualification/               six-arm run with downstream answers
│   ├── packing_diagnostic/          why complete evidence was not always enough
│   └── gate_b_verdict.json          machine-readable verdict
└── gate_c/                          bounded iterative retrieval (in progress)
    └── sprint2/                     four-arm marginal-utility measurement
```

## Voided runs and why

| Run | Status | Reason |
|---|---|---|
| `gate_a/voided_run_001` | VOIDED | HF-subword truncation synthesised the gold answer token at a B1 chunk boundary (`Station-058-741` → `Station-058-74`), violating the frozen leak rule. Model behaviour on that task was unaffected, but the run cannot be promoted. |
| `gate_b/voided_pretokenizer_fix_run_001` | VOIDED | Measured before the lexical tokenizer fix; `[A-Za-z0-9_./-]+` glued sentence-final punctuation onto tokens, hiding evidence from every lexical query. Understates bm25/hash/hybrid arms. |
| `data/hrm/controlled_gate_a_v1` | SUPERSEDED | Task `temporal_update-043` carried its gold answer inside the entity name `Service-043-587`. |

## Path mapping after the gate-scoped reorganisation

| Former path | Current path |
|---|---|
| `evidence/gate_a_qualification_v2/` | `evidence/gate_a/voided_run_001/` |
| `evidence/gate_a_qualification_v2r1/` | `evidence/gate_a/qualified_run_002/` |
| `evidence/gate_a_report_v2r1.json` | `evidence/gate_a/qualified_run_002/gate_a_report_v2r1.json` |
| `evidence/gate_a_protocol_manifest.json` | `evidence/gate_a/protocol_manifest_v1_superseded.json` |
| `evidence/gate_a_protocol_manifest_v2.json` | `evidence/gate_a/protocol_manifest_v2.json` |
| `evidence/gate_a_b2_tokenizer_erratum.md` | `evidence/gate_a/ERRATA.md` |
| `evidence/gate_b_retrieval_v1/` | `evidence/gate_b/voided_pretokenizer_fix_run_001/` |
| `evidence/gate_b_retrieval_v2/` | `evidence/gate_b/retrieval_only_run_002/` |
| `evidence/gate_b_v2/` | `evidence/gate_b/qualification/` |
| `evidence/gate_b_verdict.json` | `evidence/gate_b/gate_b_verdict.json` |

## Pre-HRM history

`research_build_v320/`, `research_build_v340/`, `e3_*`, and `voc_stage1_smoke_v1/`
are frozen artifacts from the legacy Qwen/ExFusion and metareasoner lines. They
are retained unchanged and are not part of the HRM gate ladder.
