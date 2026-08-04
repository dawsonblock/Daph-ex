# Technical implementation report

> Historical v3.1 report. For the current middle-layer research build, see
> `docs/IMPLEMENTATION_REPORT_V320.md`, `docs/LAYER_CONTRIBUTION_DESIGN.md`, and
> `docs/E3_MIDDLE_ARCHITECTURE.md`.

## Changed architecture

- `daph/qwen_exfusion.py`: partial-depth E0/E1, exact E2, delta-semantics E3, branch gating, training initialization, parameter provenance, and structured receipts.
- `daph/compute.py`: deterministic operation-family compute estimator.
- `scripts/run_phase0_retention.py`: canonical QwenCompat → QwenExFusion Gate 0B and `phase0b_gate_report.json`.
- `daph/train_real.py`: QwenExFusion support, E2 distillation, explicit stages, exact provenance groups, exposure counters, unambiguous resume counters, and final accumulation flush.
- `daph/counterfactual.py`: measured receipt-backed utility, E2-normalized canonical cost, and physical-order rejection.
- `daph/pretrained.py`: canonical provenance in adapted checkpoints.
- `tests/test_effort_compute_ordering.py`: physical graph, parity, branch, gradient, distillation, residual, and provenance gates.
- `scripts/run_small_real_adaptation.py`: pinned, memory-bounded HF Qwen/WikiText adaptation and qualification harness.

## Training design

`TrainingStageConfig` explicitly describes train/freeze groups, distinct learning rates, effort exposure, teacher mode, E0/E1 distillation, and E3 training. Shallow losses are padding-aware causal CE plus temperature-scaled E2 KL. E3 uses task CE by default. Training receipts record micro/optimizer steps, examples, tokens, next microstep, stage state, and effort exposure.

## Limitations

- E0/E1 default to the scientifically clean shallow-exit baseline. The optional zero-residual `CheapContinuation` bottleneck can be enabled for frozen-backbone Stage 1 and ablated explicitly.
- AttnRes remains disabled in the canonical experiment.
- Deterministic compute units are calibrated proxies, not device-specific FLOP profiler output. Optional latency/memory fields require a benchmark harness.
- Meaningful model-quality qualification still requires real adaptation data and an immutable evaluation corpus; synthetic tests prove plumbing and invariants only.

## Real checkpoint verification

`Qwen/Qwen2.5-0.5B` at immutable revision `060db6499f32faf8b98477b0a26969ef7d8b9987` was imported and evaluated on a deterministic WikiText-2 slice. Gate 0A achieved 100% source coverage and near-numerical logit parity. Gate 0B achieved exact E2 logits and exact backbone identity. Deterministic normalized compute was E0 `0.5070`, E1 `0.7570`, E2 `1.0000`, and E3 `2.2567`.

The stable ten-step adaptation improved E0 CE by `0.0491` and E1 CE by `0.0189`, preserved E2 exactly, and worsened E3 CE by `0.8085`. Therefore this is an engineering pass but a quality-qualification failure. See `docs/REAL_MODEL_SMOKE_REPORT.md` for the complete result and next workflow.

## Quality correction

The initial failure was traced to recurrent/MoE/latent augmentation in every E3 layer, an unbounded residual scale, severe shallow-exit representation mismatch, and insufficient training exposure. Canonical E3 now performs one bounded final-layer latent refinement; E0/E1 add optional hidden-state distillation; and the real-model harness trains and validation-selects each candidate separately.

On all 16 smoke-validation records, E0/E1 CE improved by `1.4681`/`0.6682`, E2 remained unchanged, and E3 improved by `0.000097` from its own initialized value. Deterministic compute is `0.7570`, `0.9237`, `1.0000`, and `1.00894`. See `docs/QUALITY_CORRECTION_REPORT.md` for root causes and limitations.
