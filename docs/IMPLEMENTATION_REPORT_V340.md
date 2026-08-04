# v3.4.0 implementation report

## Outcome

The scientific-accounting defect is fixed without changing QwenCompat, Gate 0A, Gate 0B, the physical E0/E1 depth paths, the internal effort probe, or verified-controller provenance.

Every E2/E3 task pair now supplies measured quality and normalized physical compute from the actual generation receipts. Qualification materializes:

`delta_quality = Q3 - Q2`

`delta_compute = C3 - C2`

`delta_utility = delta_quality - lambda_compute * delta_compute`

There is no correctness fallback and no effort-ID cost lookup. Missing compute is a validation error.

## Implemented systems

- `daph/e3_metrics.py`: receipt validation, explicit utility records, grouped bootstrap, E3-Q/E3-U gates, lambda sweep, and break-even lambda.
- `daph/effort_frontier.py`: E0-E3 quality/compute/utility frontier, Pareto dominance, receipt-backed oracle opportunity, and router block.
- `daph/e3_protocol.py`: smoke/pilot/qualification/final tiers, profile tiers, cross-seed stability, placement promotion, claim levels, and immutable artifact metadata.
- `daph/verified_tasks.py`: nine deterministic exact-answer families and separate calibrated/natural split builders.
- `daph/e3_training.py`: explicit answer-only CE, external verified-reward, and unimplemented GRPO contracts.
- `scripts/run_e3_hardcase_ablation.py`: per-task generation receipts, Q/C/U fields, natural test, lambda sweep, and separate qualification.
- `scripts/qualify_e3_results.py`: self-contained post-run scientific evidence generation.

E3 supports a fixed configured dose and a batch-size-one per-example research override. The receipt records the dose actually executed. Learned halting is intentionally not implemented.

## Promotion discipline

An E3 placement is promoted only when quality and utility lower bounds are positive, rescues exceed regressions, the result replicates across seeds, profile-based placement is stable, and the untouched natural test passes. A passing E3 arm still does not authorize a router: a receipt-backed oracle must also beat the best fixed arm with a positive grouped-bootstrap lower bound.

## Current evidence interpretation

The historical `1 rescue / 24` result is retained unchanged. It is `MECHANISM_SIGNAL`: the quality lower bound is not positive, and cost-aware utility was not recorded per task in that historical artifact. It is not retroactively promoted.

## Remaining scientific uncertainty

- Whether middle E3 produces a positive quality lower bound on hundreds of tasks.
- Whether any quality gain survives realistic compute prices on the natural distribution.
- Whether gains replicate across training seeds and task families.
- Whether the simple middle heuristic continues to beat profile-guided placement after a stable profile.
- Whether teacher-forced answer-only gains transfer to sequence-level verified generation.
- Whether E0/E1/E3 form a useful Pareto frontier or are dominated by E2.
- Whether the oracle gap is large enough to repay controller overhead.
- Whether a 1.5B-1.7B model changes the frontier after the small-model pilot passes.

No E3 capability or routing hypothesis is claimed as proven.
