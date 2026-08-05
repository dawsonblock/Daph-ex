# Current architecture audit (DAPH-HRM-B0)

Baseline freeze: see `artifacts/baseline/` (commit in `git_commit.txt`,
216-test receipt, Gate A PASS metrics). This audit classifies every subsystem
before Sprint 1 (Gate B) work begins. Classifications:

- **canonical** — active runtime the unified roadmap builds on
- **implemented-unqualified** — code exists; no gate has qualified it
- **scaffold** — interface/contract only, no qualified behavior
- **legacy** — frozen experimental library; not merged into active runtime
- **frozen-evidence** — immutable scientific artifacts; never rewritten
- **external-integration** — adapter to an external engine; gate-blocked
- **prohibited** — must not enter current qualification

| Subsystem | Path | Classification | Notes |
|---|---|---|---|
| HRM adapter (pinned native model) | `hrm_adaptive_memory/hrm/model.py` | canonical | revision-pinned, fail-closed config validation |
| Recurrent hooks / variable recurrence | `hrm_adaptive_memory/hrm/{recurrent_hooks,variable_recurrence}.py` | implemented-unqualified | recurrence gates (Stage 10+) have not run |
| Context study (B0/B1/B1b/B2/B3) | `hrm_adaptive_memory/experiments/context_study.py` | canonical | Gate A qualified; leak filters incl. truncation re-check |
| Controlled dataset generator v2 | `hrm_adaptive_memory/experiments/controlled_dataset.py` | canonical | answer-in-question guard; audited leak-free |
| Gate A qualification | `hrm_adaptive_memory/evaluation/context_gate.py`, `evaluation/bootstrap.py` | canonical | grouped bootstrap, no IID fallback |
| Retrieval controls (BM25/hash/RRF-hybrid) | `hrm_adaptive_memory/retrieval/`, `backends/local.py` | canonical (control arms) | primitive by design; Gate B compares against pinned dense |
| Canonical Gate B arms (6) | `hrm_adaptive_memory/backends/canonical.py` | canonical | bm25/hash/dense/hybrid_score/hybrid_rrf/hybrid_rerank behind one contract |
| Pinned embedding backend | `hrm_adaptive_memory/retrieval/embedding.py` | canonical | model+revision+pooling+norm+dim+dtype hashed into a config digest |
| Retrieval metrics + failure attribution | `hrm_adaptive_memory/evaluation/{retrieval_metrics,failure_analysis}.py` | canonical | complete-evidence-set success is the decisive multi-hop measure |
| Resource accounting | `hrm_adaptive_memory/evaluation/resources.py` | canonical | phase-attributed latency; unknown metrics are `None`, never guessed |
| HRM state contract + commit ledger | `hrm_adaptive_memory/hrm/state.py` | scaffold (behavior-neutral) | selected-state-equals-committed-state invariant; unused by Gate A |
| RuVector sidecar bridge | `hrm_adaptive_memory/backends/ruvector.py` | external-integration | source-lock + Gate A gated; Stage 8 only |
| Context packer / budgets | `hrm_adaptive_memory/context/packer.py` | canonical | 4K workspace budgets |
| Memory lifecycle/stores/contradiction | `hrm_adaptive_memory/memory/` | implemented-unqualified | transactional-memory gates (Stage 14+) have not run |
| Controller actions/policy | `hrm_adaptive_memory/controller/` | scaffold | blocked until Gate C oracle opportunity |
| Counterfactual/oracle execution | `hrm_adaptive_memory/execution/` | scaffold | Gate C machinery; unqualified |
| Source lock | `hrm_adaptive_memory/source_lock.py` | canonical | fail-closed third-party runtime gating |
| Derivation cache | `hrm_adaptive_memory/derivation.py` | implemented-unqualified | integrity-checked cache; unused by Gate A |
| Qwen/ExFusion research system | `daph/` | legacy | LEGACY_QWEN_EXFUSION; tests kept passing; do not merge into runtime |
| Metareasoner stage-1 | `daph_metareasoner/` | legacy | LEGACY_METAREASONING; experimental library only |
| Legacy namespace shim | `hrm_memory/` | legacy | one-release deprecation alias |
| Gate A evidence + protocol manifests | `evidence/gate_a_*`, `evidence/hrm_{smoke,pilot}_*`, `evidence/controlled_corpus_audit_*` | frozen-evidence | includes the VOIDED first qualification attempt |
| Legacy E3/VoC evidence | `evidence/{research_build,e3_*,voc_*}` | frozen-evidence | pre-HRM history |
| Controlled corpora | `data/hrm/controlled_gate_a_*` | frozen-evidence | immutable versioned datasets |
| Hierarchos import | (external zip) | prohibited | concepts only (state contract, parity, fail-closed checkpoints, transactional memory, ACT); its RWKV core, manager/worker, fixed-slot LTM, ROSA, GUI, quantization experiments, and response-derived self-learning are excluded |

## Gate ladder position

Gate A **PASSED** (`evidence/gate_a/qualified_run_002/gate_a_report_v2r1.json`): mean B3−B0 = 0.998,
LCB95 = 0.994 (all groupings). Enabled: retrieval expansion only. Blocked:
iterative retrieval promotion (needs Gate B), controllers (need Gate C),
adaptive recurrence (Stage 10 gate), Graphiti (Stage 9), transactional
persistent memory (Stage 14), external vector engines (Stage 8).

Known causal finding for Gate B: BM25 B2−B0 = +0.60 overall — perfect on
single-evidence families, ≈0 on `two_hop`/`numeric_derivation` because the
second required record shares no lexical terms with the question
(complete-evidence-set recovery failure, not an HRM failure).

**Correction (Sprint 1).** Part of that finding was an instrumentation defect,
not a retrieval property. `tokenize` used `[A-Za-z0-9_./-]+`, so a
sentence-final entity produced the token `plan-000-965.` and never matched the
same entity written mid-sentence. With the corrected pattern, BM25's
complete-evidence-set success rises 0.618 → 0.818 and `numeric_derivation`
rises 0.000 → 1.000; only `two_hop` remains genuinely retrieval-bound, and
structurally so (the bridge entity is unknowable from the question alone).
Gate A's qualified B3−B0 claim is unaffected — neither arm uses retrieval. See
`evidence/gate_a/ERRATA.md`.
