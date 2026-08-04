# E3 qualification protocol v3.4.1

Use [`configs/e3_qualification_v341.json`](../configs/e3_qualification_v341.json) as the pre-registration. The target is one receipt-backed, multi-seed comparison on pinned `Qwen/Qwen2.5-0.5B`, not an architecture revision.

## Required arms and controls

Run `final_refine`, `middle_recurrent`, and `profiled_middle_recurrent` with refinement doses 1, 2, and 4 during selection. Evaluate the predeclared common dose of 4 on held-out data. Freeze E2 and train only E3 refiner/scale parameters with answer-token-only CE and the weak regression guard.

Use three independent training seeds: `20260803`, `20260817`, and `20260831`. A placement replicates only if at least two seeds independently pass E3-Q and E3-U on both evaluation splits.

## Required evaluation splits

- `CALIBRATED_SENSITIVITY`: 500 tasks, family-stratified to the declared E2 mixed-success band. E3 outcomes are never inspected during selection.
- `NATURAL_HELDOUT`: 500 tasks selected before either E2 or E3 evaluation. No outcome-based filtering is permitted.

Use at least five exact-verifier families; all nine generated families are preferred. Bootstrap whole `template_id` clusters with 10,000 resamples. Report quality, utility, rescue/regression, CE, hidden-state delta, deterministic compute, and wall-clock latency per family, template, generator bucket, and seed.

## Utility and gates

For every task and arm, compute:

`U = Q - lambda_compute * C`

from verified quality and the actual execution receipt. Evaluate lambda values `0`, `0.1`, `0.25`, `0.5`, `1`, and `2` while preserving `lambda=1` as the primary predeclared result.

E3-Q requires `LCB95(delta_quality) > 0`. E3-U requires `LCB95(delta_utility) > 0`. A placement is promotable only when both tests pass on calibrated and natural splits, rescues exceed regressions, at least two seeds replicate, and no severe family regression is present.

## Profile prerequisite

The profiled arm requires three independent `PROFILE_PILOT` runs with at least 200 train examples, 200 validation examples, and 20 updates per candidate. Aggregate them with `scripts/analyze_profile_stability.py`. If the tier or stability check fails, retain profiled placement as a research observation but exclude it from promotion.

## Downstream decision

If no placement promotes, E3 remains unqualified. If a placement promotes, freeze all arms, collect receipt-backed E0-E3 counterfactuals, and run the oracle-minus-best-fixed gate. Router training remains forbidden unless that oracle lower bound is also positive.

Answer-only CE is not RLVR. Persistent workspaces, progressive latent supervision, middle-layer adaptation, GRPO, and dynamic halting are follow-up hypotheses, gated on the result of this experiment.
