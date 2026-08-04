# E3 utility-accounting report

## Corrected definition

The canonical cost-aware score is:

`U_e = Q_e - lambda_compute * C_e`

`Q_e` is the task verifier's numeric quality. `C_e` is normalized physical compute derived from the task's actual execution receipt. The experiment stores both receipts, their raw deterministic compute units, and the paired normalization rule.

The previous field `mean_verified_utility_delta` was removed. It could contain only a binary correctness difference when explicit utility fields were absent. No compatibility alias remains.

## Two gates

E3-Q answers whether E3 improves verified capability. It bootstraps `delta_quality` by template/family group and requires a positive lower confidence bound plus more rescues than regressions.

E3-U answers whether the improvement is worth its cost at the declared compute price. It independently bootstraps `delta_utility` and requires a positive lower confidence bound.

The states are:

- `INSUFFICIENT_POWER`
- `FAIL_QUALITY`
- `PASS_QUALITY_FAIL_UTILITY`
- `PASS_QUALITY_AND_UTILITY`

Only the final state qualifies an E3 arm. Router work additionally requires a positive oracle-opportunity gate.

## Lambda sweep

The default declared values are `0, 0.1, 0.25, 0.5, 1, 2`. Each row reports mean utility delta, grouped-bootstrap interval/lower bound, and the fraction of tasks where E3 has higher utility.

When `delta_compute > 0`, break-even compute price is:

`lambda_star = delta_quality / delta_compute`

The aggregate break-even value uses mean deltas. Per-example summaries are reported separately and do not replace the aggregate gate.

## Historical pilot

Using only the published aggregate values for intuition, not as new receipt-backed evidence:

- mean binary quality change: about `0.04167`
- mean compute change: about `0.03585`
- lambda=1 point estimate: about `0.00582`

The quality lower bound was zero; after pricing compute, the utility lower bound cannot be positive. The historical pilot remains unqualified.
