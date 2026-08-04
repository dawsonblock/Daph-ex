# Answer-only mixed-E2 E3 location study

> v3.4.0 accounting correction: this historical bootstrap measured binary quality change, despite an older field naming it verified utility. It did not subtract per-task E3 compute. The result is retained as `MECHANISM_SIGNAL`, not cost-aware qualification. New runs must pass separate E3-Q and E3-U gates using actual execution receipts; see `UTILITY_ACCOUNTING_REPORT.md`.

## Outcome

This experiment fixes the two largest weaknesses of the earlier hard-case smoke run:

- E3 is trained only on answer tokens; prompt and padding tokens are masked with `-100`.
- Every train, selection, and held-out split is calibrated to 50% E2 exact accuracy, rather than using a degenerate 0%-accuracy set.

The matched held-out comparison used four recurrent refinement steps for every location. The Qwen E2 backbone remained frozen and each arm had the same deterministic compute overhead.

| E3 location | Zero-based layer | E2 accuracy | E3 accuracy | Rescues | Regressions | E3 CE delta | Compute overhead | 95% quality LCB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| final state | 23 | 50.00% | 50.00% | 0 | 0 | -0.06385 | 3.5848% | 0.0000 |
| heuristic middle | 12 | 50.00% | 54.17% | 1 | 0 | -0.11999 | 3.5848% | 0.0000 |
| profiled middle | 13 | 50.00% | 50.00% | 0 | 0 | -0.10503 | 3.5848% | 0.0000 |

The heuristic-middle arm produced the first held-out verified E2-to-E3 rescue in this research line and had no regression. It also achieved the largest teacher-forced CE improvement at the matched dose. This is a positive mechanism signal, but it is not a scientific qualification: with only one changed outcome among 24 tasks, the paired bootstrap lower confidence bound remains zero.

Therefore:

`E3 qualification = FAIL`

`policy training allowed = false`

The stop gate is operating as intended.

## Dose-selection result

On the independent selection set, heuristic-middle improved from no verified change at one step to one rescue and no regressions at two and four steps. Its CE improvement also grew from `0.04034` to `0.06912` to `0.12657`. Profiled-middle showed a similar CE dose response and one selection-set rescue at four steps. Final-state refinement regressed one selection example at two and four steps.

The held-out study forced all three locations to four steps. This prevents the earlier selection winners (final=1, middle=4, profiled=4) from confounding location with compute dose.

## Interpretation

The old sparse supervised profile chose layer 13, while the default heuristic middle at layer 12 performed better on this held-out set. That profile used only eight training examples, four validation examples, two updates, and 11 sampled layers. This result reinforces the existing warning that the profile validates plumbing but is too weak to establish the best reasoning location.

The next decisive experiment is replication on substantially larger held-out sets and independent verified task families. A stronger checkpoint-specific profile should use hundreds or thousands of verified examples, multiple seeds, and a verified task metric. Router training must remain blocked until the paired lower confidence bound is strictly positive and the fixed effort arms pass the quality/compute frontier gate.

## Reproducibility

The experiment used `Qwen/Qwen2.5-0.5B` at immutable revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, seed `20260803`, 40 training steps per dose, dose candidates `{1,2,4}`, and 2,000 paired bootstrap samples at 95% confidence.

Raw calibrated splits, calibration outcomes, per-location reports, hashes, and the consolidated result are under [`evidence/e3_answer_only_mixed_v1`](../evidence/e3_answer_only_mixed_v1/manifest.json).
