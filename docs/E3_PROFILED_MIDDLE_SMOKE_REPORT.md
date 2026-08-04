# Profiled-middle E3 smoke report

## Result

The paper-guided middle-layer hypothesis produced a stronger held-out completion-CE improvement than a matched final-layer refiner, but it did not produce verified task rescues. This is encouraging mechanism evidence, not scientific qualification.

The pinned source was `Qwen/Qwen2.5-0.5B` at revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. Phase 0A imported 100% of source tensors and passed exact parity. Gate 0B also passed exactly with zero CE difference, zero logit MAE, and unchanged E2 behavior.

## Harness corrections

The run uncovered and fixed three experiment-harness defects before measurement:

- the ablation script always snapshotted `layers[-1]`, even for middle E3;
- changing `default_e3_steps` did not change the canonical `e3_config.e3_refine_steps`, so nominal dose variants executed the same configured dose;
- the documented profile CLI could not import `daph` when executed directly.

The corrected harness explicitly records the active refinement layer, changes the canonical serialized step count, gives every location the same seeded refiner initialization, and supports `final_refine`, `middle_recurrent`, and `profiled_middle_recurrent`.

## Sparse checkpoint-specific profile

This was a deliberately low-budget supervised-CE profile: 8 training examples, 4 validation examples, 2 adaptation steps per candidate, and 11 sampled layers. It is labeled `PARTIAL_PROFILE`; it is not an RLVR reproduction or a global layer ranking.

| Measurement | Result |
|---|---:|
| Base validation CE | 3.35218 |
| Full-reference validation CE | 1.43802 |
| Highest sampled contribution | layer 14, 0.23189 |
| Best contiguous sampled region | layers 12–14 |
| Selected insertion layer | 13 |
| Mean 40%–60% contribution | 0.21042 |
| Middle concentration observed | yes |

The profile digest is `62edb20bf45e4919964df218bf9d917b2ed6ed596237707fbd9bb54da137a232`.

## Matched location comparison

Both variants used the same pinned backbone, 64 training tasks, 8 selection tasks, 8 untouched held-out tasks, seed, optimizer budget, initial refiner state, residual-scale initialization, and E2-frozen parameter set. Each dose received 20 optimizer steps. Four refinement steps won selection for both locations.

| Held-out measurement | Profiled middle, layer 13 | Final control, layer 23 |
|---|---:|---:|
| E2 completion CE | 2.80439 | 2.80439 |
| E3 completion CE | 2.78369 | 2.79896 |
| E3 CE improvement | 0.02070 | 0.00543 |
| Extra deterministic compute | 3.5748% | 3.5748% |
| Mean hidden-state delta L2 | 1.65507 | 0.15428 |
| E2 exact accuracy | 0/8 | 0/8 |
| E3 exact accuracy | 0/8 | 0/8 |
| Rescues / regressions | 0 / 0 | 0 / 0 |

At equal measured compute, the middle location's CE gain was approximately 3.81 times the final location's gain. Its held-out E3 CE was 0.01527 lower than the final control.

The selection split also showed a monotonic middle-layer dose response:

| Middle steps | E3–E2 completion CE | Extra compute |
|---:|---:|---:|
| 1 | -0.00664 | 0.8937% |
| 2 | -0.01122 | 1.7874% |
| 4 | -0.01911 | 3.5749% |

## Decision

The paper prior helped choose a more effective location for this small training objective. It has not yet demonstrated useful reasoning or task success: neither location rescued a single exact answer, the profile was sparse and low-budget, the held-out set had only 8 examples, and causal-LM training still supervised prompt tokens as well as answer tokens.

Therefore:

- middle-location mechanism signal: **positive smoke result**;
- verified E3 task utility: **not demonstrated**;
- E3 qualification: **fail**;
- router training: **remains blocked**.

The next run should use answer-only loss, tasks on which E2 has a non-degenerate mix of successes and failures, at least one independent hard-task family, a larger held-out split, and replicated seeds.

Raw compact evidence is stored under `evidence/e3_middle_profiled_v1/`. The 2.5 GB Gate 0B checkpoint is intentionally excluded; its identity and parity receipts are included.
