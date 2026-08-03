# Technical implementation report

## Changed architecture

- `daph/qwen_exfusion.py`: partial-depth E0/E1, exact E2, delta-semantics E3, branch gating, training initialization, parameter provenance, and structured receipts.
- `daph/compute.py`: deterministic operation-family compute estimator.
- `scripts/run_phase0_retention.py`: canonical QwenCompat → QwenExFusion Gate 0B and `phase0b_gate_report.json`.
- `daph/train_real.py`: QwenExFusion support, E2 distillation, explicit stages, exact provenance groups, exposure counters, unambiguous resume counters, and final accumulation flush.
- `daph/counterfactual.py`: measured receipt-backed utility, E2-normalized canonical cost, and physical-order rejection.
- `daph/pretrained.py`: canonical provenance in adapted checkpoints.
- `tests/test_effort_compute_ordering.py`: physical graph, parity, branch, gradient, distillation, residual, and provenance gates.

## Training design

`TrainingStageConfig` explicitly describes train/freeze groups, distinct learning rates, effort exposure, teacher mode, E0/E1 distillation, and E3 training. Shallow losses are padding-aware causal CE plus temperature-scaled E2 KL. E3 uses task CE by default. Training receipts record micro/optimizer steps, examples, tokens, next microstep, stage state, and effort exposure.

## Limitations

- E0/E1 default to the scientifically clean shallow-exit baseline. The optional zero-residual `CheapContinuation` bottleneck can be enabled for frozen-backbone Stage 1 and ablated explicitly.
- AttnRes remains disabled in the canonical experiment.
- Deterministic compute units are calibrated proxies, not device-specific FLOP profiler output. Optional latency/memory fields require a benchmark harness.
- Meaningful model-quality qualification still requires real adaptation data and an immutable evaluation corpus; synthetic tests prove plumbing and invariants only.
