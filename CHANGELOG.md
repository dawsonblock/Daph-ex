# Changelog

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
