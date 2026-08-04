# Changelog

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
