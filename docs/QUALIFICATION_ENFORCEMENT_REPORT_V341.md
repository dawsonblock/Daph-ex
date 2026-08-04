# Qualification enforcement report v3.4.1

## Closed loophole

`qualify_e3_pairs()` no longer permits a two-row result to look powered. Its local smoke minimum is 24 tasks and executable experiment paths bind an `ExperimentScale` to every report. The tier requirements are:

| Tier | Held-out tasks per split | Bootstrap groups | Independent training seeds |
|---|---:|---:|---:|
| `SMOKE` | 24 | 2 | 1 |
| `PILOT` | 200 | 5 | 3 |
| `QUALIFICATION` | 500 | 5 | 3 |
| `FINAL` | 500 minimum and exact predeclared size | 5 | 3 |

The hard-case, location-study, and paired-result qualification commands call `ExperimentScale.validate()` before model loading or training. Observed task, group, and seed counts are checked again when the paired statistics are materialized.

## Local statistics versus promotion

`PASS_QUALITY_AND_UTILITY` is the paired statistical result. Canonical placement promotion is a separate decision and additionally requires:

- the declared experiment tier to pass;
- both calibrated-sensitivity and untouched natural tests to pass;
- at least two of three training seeds to pass both tests;
- rescues greater than regressions;
- matched held-out refinement dose;
- for profiled placement, a stable multi-seed `PROFILE_PILOT` or `PROFILE_FULL` artifact.

The effort router remains blocked even after E3 promotion until the actual-compute oracle opportunity gate passes.

## Calibration and grouping

Mixed-success calibration is balanced within task family. The manifest records available and selected E2 successes/failures for every family and rejects an infeasible or severely imbalanced request. Each of the nine verified families now has three prompt templates, yielding up to 27 genuine template clusters for grouped bootstrap.

Generator-scale difficulty is stored as `GENERATOR_EASY`, `GENERATOR_MEDIUM`, or `GENERATOR_HARD` with `difficulty_source=generator_numeric_scale_v1`. It is not described as model-defined difficulty.

## Profile enforcement

Single-seed profiling writes a non-promotable validation record. `scripts/analyze_profile_stability.py` combines independent profile runs, validates the declared profile tier, measures shared-layer Spearman correlation, top-k overlap, and region stability, and emits an `AGGREGATED_PROFILE` whose placement is selected from the mean contribution across seeds. Shared-layer ranks are computed only on the intersection evaluated by every seed.

## Scientific status

This patch changes enforcement only. It contains no new real E3 qualification result. The historical one-rescue result remains a `MECHANISM_SIGNAL`, and E3 remains scientifically unqualified until the predeclared multi-seed experiment passes both quality and cost-aware utility on the natural held-out split.
