# Changelog

## 3.7.1 — Gate C measured; mechanism saturates, benchmark does not certify

- Bounded two-pass retrieval + entity-anchored precision packing reaches the
  oracle ceiling: 1.000 quality, 1.000 complete-evidence-set recovery, zero
  failures across all 500 tasks and all twelve taxonomy classes.
- Verdict is nonetheless FAIL_ITERATIVE_RETRIEVAL: under Gate A's own rule
  (LCB95 > 0 for every grouping key) the family view yields +0.0000 because
  bridge structure exists in only one family of five. The threshold was frozen
  before results were read and was not moved.
- Marginal utility: precision packing +0.018, bounded follow-up +0.182,
  deterministic calculator +0.000.
- Negative results recorded: CALCULATE is not promoted; adaptive retrieval is
  not justified (91/91 follow-ups positive, so Gate D should be expected to
  fail here); the slot-label echo was an evidence-confusability artefact.
- Fixed an evidence-selection defect that discarded an already-retrieved second
  hop because anchoring used question entities only; the affected run is
  retained and voided rather than overwritten.

## 3.7.0 — Gate A0 and Gate B qualified; lineage repaired

Release integrity: 3.6.1 shipped newer science under stale metadata (pyproject
3.6.1, `daph.__version__` 3.4.1, README asserting Gate A had not been run).
This release makes version, README, changelog, and gate state agree, enforced
by tests, and adds machine-readable `RESEARCH_STATUS.json`.

- **Gate A0 PASS** — HRM-Text-1B uses correctly supplied evidence on the
  controlled synthetic benchmark: mean B3−B0 = 0.998, grouped-bootstrap LCB95
  = 0.994 for every grouping key. Scoped to the controlled synthetic corpus;
  not a claim about general, natural-document, or persistent memory.
- **Gate B PASS** — BM25 recovers complete evidence sets on 81.8% of tasks and
  lifts downstream quality to 0.800 vs a 0.002 baseline. Scoped to *the tested
  dense stack*: MiniLM-L6-v2, single-vector, mean-pooled, cosine.
- Retrieval precision established as a binding constraint: with required
  evidence held present, quality falls 1.00 (random distractors) → 0.67
  (same-template) → 0.39 (retriever top-k), with the model emitting evidence
  slot labels instead of answers.
- Corrected a lexical tokenizer defect that glued sentence-final punctuation
  onto tokens, hiding evidence from every lexical query; BM25 complete-set
  success rose 0.618 → 0.818 and `numeric_derivation` 0.000 → 1.000. Gate A0's
  qualified claim is unaffected (neither arm retrieves); see the erratum.
- Corrected a generator defect that embedded a gold answer in its own question
  (`controlled_gate_a_v2`), and a B1 control leak where subword truncation
  could synthesise the answer token at a chunk boundary.
- Added: canonical six-arm retrieval backends, revision-pinned embedding
  backend with hashed config, complete-evidence-set metrics, failure
  attribution, phase-attributed resource accounting, the HRM state contract
  and commit ledger, bounded two-pass retrieval with bridge detection, and an
  AST-restricted calculator.
- Voided and failed runs are retained, never overwritten.

## 3.6.1 — Gate A confound controls

- Remove model-visible B0/B1/B2/B3 labels from study prompts and retain condition identity only in immutable receipts.
- Split `CAPABILITY_USE` from non-promotable `EVIDENCE_GROUNDED` studies so abstention framing cannot be reported as oracle-evidence capability.
- Add an optional answer-free, token-matched B1b hard-distractor control and require it uniformly when selected.
- Require explicit source-cluster labels and report grouped bootstrap results for template, family, and source cluster; Gate A uses the most conservative result and requires every declared cluster view to pass.
- Record gross quality plus retrieval, compute, latency, token, and verification costs in counterfactual receipts.
- Lock the supplied TurboVec snapshot (`3eba4445…ee341`, Python 0.8.0 / Rust 0.9.0) as a disabled compressed-dense experimental backend; no TurboVec runtime or adapter is enabled.

No real Gate A experiment is included. Retrieval expansion remains blocked.

## 3.6.0 — HRM adaptive-memory control plane

- Rename the canonical research namespace to `hrm_adaptive_memory` while retaining one-release `hrm_memory` compatibility aliases.
- Add asynchronous retrieval, graph, memory, consolidation, HRM-runtime, and action-executor contracts with immutable receipts and explicit capability negotiation.
- Add five logical memory kinds, fail-closed lifecycle transitions, immutable provider-neutral derivation caching, and an audited external-source lock.
- Add loopback-only sidecar configuration and keep every external runtime disabled until its scientific prerequisite passes.
- Add a canonical paired B0/B1/B2/B3 runner that constructs each context, independently consumes oracle labels, token-matches irrelevant context, records prompt/evidence/model receipts, and prevents fake results from qualifying.
- Add deterministic grouped Gate A bootstrap with 24/100/500 task tiers; only a 500-task qualification can unlock retrieval expansion.
- Preserve primitive BM25/hash/hybrid controls and add 13 adversarial control-plane tests, including a deterministic loopback RuVector bridge contract; complete suite: 204 passing tests.

No real Gate A experiment is included. RuVector, Graphiti, iterative retrieval, adaptive recurrence, and controller training remain blocked.

## 3.5.0 — HRM external memory + adaptive compute foundation

- Add a revision-pinned native HRM-Text-1B adapter with correct PrefixLM masking.
- Add append-only source, semantic, and episodic memory with provenance and lineage.
- Add structural chunking, BM25+dense RRF retrieval, reranking interfaces, and evidence metrics.
- Add redundancy-aware 4096-token evidence packing and the mandatory oracle-context gate.
- Add recurrent H/L state tracing, recurrence ablation declarations, isolated counterfactual execution, and a fail-closed utility controller.
- Add executable Stage A/B/C commands, immutable configuration, protocol documentation, and tests.

This release establishes the experimental substrate. It does not claim HRM memory, retrieval, recurrence, or controller gains.

## 3.4.1 — qualification-tier enforcement

- Bound `SMOKE`, `PILOT`, `QUALIFICATION`, and `FINAL` sample/group/seed minimums to the executable E3 qualification paths; a two-task result can no longer promote an arm.
- Added a separate placement-promotion decision requiring tier validation, natural-test success, at least two-of-three seed replication, and stable `PROFILE_PILOT`/`PROFILE_FULL` evidence for profiled placements.
- Made calibrated sensitivity sampling family-stratified and recorded per-family success/failure availability and realized balance.
- Expanded every verified task family to three distinct prompt templates and labeled generator-scale difficulty separately from empirical/model difficulty.
- Corrected profile stability to rank only layers shared by every seed and added a multi-seed profile aggregation command.
- Added final-tier predeclared sample-size enforcement and fail-fast CLI validation before model loading or GPU training.
- Made the multi-seed location study resumable and removed duplicated receipt records from summary payloads for long qualification runs.
- Cached expensive E2 calibration outcomes and added a declared largest-feasible-family rule (minimum five families) when an arm cannot supply a mixed-success sensitivity band; the natural test still retains all nine families.
- Made multi-seed profile aggregation emit the canonical mean-contribution ranking/region and a digest-bound `AGGREGATED_PROFILE`, avoiding placement from an arbitrary seed.

This release changes qualification enforcement, not the scientific result. The historical one-rescue result remains `MECHANISM_SIGNAL`; router training remains blocked.

## 3.4.0 — receipt-backed E3 scientific accounting

- Replaced the correctness-as-utility fallback with mandatory per-task quality and actual E2/E3 execution compute.
- Split qualification into capability gate E3-Q and cost-aware gate E3-U, with explicit `FAIL_QUALITY`, `PASS_QUALITY_FAIL_UTILITY`, `PASS_QUALITY_AND_UTILITY`, and `INSUFFICIENT_POWER` states.
- Added configurable lambda sweeps, aggregate/per-example break-even compute prices, grouped template bootstrap, seed/family/difficulty breakdowns, and immutable paired records.
- Added distinct calibrated-sensitivity and untouched natural-test contracts plus nine deterministic verified task families.
- Added profile tiers and stability metrics, data-driven placement promotion, effort-frontier/Pareto reporting, and an actual-compute oracle gate.
- Added explicit answer-only, external verified-reward, and unimplemented-GRPO objective contracts; supervised CE is never labeled RLVR.
- Added immutable artifact commit/version/test/source-tree metadata and a postprocessing CLI that emits separate quality and utility evidence.
- Added a batch-size-one research step override that records the E3 refinement dose actually executed.

The historical one-rescue result remains a mechanism signal, not statistical or cost-aware qualification. Router training remains blocked.

## 3.3.0 — standalone marginal-utility controller

- Added the independent `daph_metareasoner` Stage 1 package around one frozen model and four actions: STOP, THINK, VERIFY, and DECOMPOSE.
- Added isolated counterfactual state/action collection with explicit gross quality change, action cost, net VOC, hidden-state features, immutable digests, and execution receipts.
- Added a mandatory oracle opportunity gate, cheap and hidden binary probes, hidden and sham action-value ensembles, ensemble uncertainty, paired confidence gates, and oracle-capture reporting.
- Added fixed, confidence, entropy, stability, length, family, and action-frequency-matched random controls.
- Added verified-only on-path execution with hard budget and loop guards; unchosen actions are never executed.
- Added leakage-resistant experience/validation/test/OOD task generation and reproducible CLI workflows.
- Preserved the first pinned real-model smoke as a negative result: oracle opportunity did not clear the predeclared threshold, so controller training was correctly blocked.

No learned-controller or value-of-computation hypothesis is claimed as validated by this release.

## 3.2.0 — middle-layer E3 research build

- Made bounded middle-layer recurrent refinement the canonical E3 experiment while retaining final-state refinement as a control.
- Added deterministic heuristic, manual, and profile-guided region selection plus zero-gated pretrained-layer reuse.
- Added complete middle/refinement metadata to compute receipts while preserving exact E2 and physical E0 < E1 < E2 < E3.
- Added a full/partial layer-contribution profiler with supervised CE, verified-reward, and external-callback objective contracts.
- Fixed scalar state-dict hashing and verified QwenExFusion counterfactual collection end to end.
- Added a reusable internal Qwen effort probe and real adaptive dispatch; unverified policy fallback now errors.
- Added E2-first hard-case mining, rescue/regression metrics, bootstrap E3 qualification, variant/dose/location experiment contracts, and task-first staged E3 training.
- Blocked policy fitting until effort-arm and oracle-opportunity qualification pass.
- Added 14 new v3.2 research gates; full suite: 116 passing tests.

No layer-concentration, E3-quality, or routing hypothesis is claimed as validated by this release.
