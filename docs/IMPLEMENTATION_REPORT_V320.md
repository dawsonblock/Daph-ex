# v3.2 implementation report

## Outcome

The build now implements the requested research plumbing without changing the E2 graph. Exact QwenCompat-to-E2 parity remains a hard test. Fixed receipts and adaptive receipts include the common probe and E3-specific middle work. Counterfactual collection now hashes scalar parameters, records an internal probe source, and captures E2/E3 outcomes and profile provenance.

Layer profiling, hard-case mining, task-level rescue statistics, dose/location/variant contracts, staged E3 objectives, and policy stop gates are implemented. The legacy hybrid model and old final E3 checkpoint semantics remain loadable.

## Acceptance evidence

- The v3.2.0 release snapshot recorded 116 passing tests; the current answer-only qualification build passes 124 tests in the default test environment.
- Exact Gate 0B: covered by existing and new tests.
- Physical effort ordering and E3 extra compute: covered by receipt tests.
- Scalar digest and one-record QwenExFusion collection: covered end to end.
- Adaptive execution: verified-controller E3 dispatch and no-controller error covered.
- Profiler: signed contribution, full/partial labels, deterministic persistence covered.
- Scientific metrics: paired rescues/regressions and bootstrap qualification covered.
- Policy fit: blocked until both prerequisite reports pass.

## Remaining scientific uncertainties

1. Whether the target Qwen checkpoint reproduces middle-layer concentration under verified-reward adaptation.
2. Whether middle recurrent refinement beats the final-state control at matched capacity, data, and training budget.
3. Whether repeated pretrained layers improve verified outcomes rather than perturbing representations.
4. The best refinement dose and whether any response is monotonic.
5. Whether E0/E1/E2/E3 form a non-dominated quality/compute frontier.
6. Whether an early hidden probe predicts utility better than prompt-only and compute-matched controls.
7. Whether any result replicates at the Tier-C 1.5B–1.7B scale and leave-family-out OOD.

No positive answer is claimed by this engineering release.
