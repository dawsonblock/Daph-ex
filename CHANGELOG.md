# Changelog

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
