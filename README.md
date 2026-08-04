# DAPH / ExFusion v3.4.1

Pretrained-compatible adaptive computation with a physically ordered four-level effort hierarchy.

## Canonical architecture

- `QwenCompatModel` is the exact source-compatible checkpoint representation used by Gate 0A.
- `QwenExFusionModel` is the canonical pretrained adaptive-compute architecture and Gate 0B target.
- `DAPHHybridModelV3` remains available as a legacy experimental SSM/attention/MoE research path.

`QwenExFusionModel` reuses one imported backbone and executes distinct graphs:

| Mode | Execution | Intent |
|---|---|---|
| E0 | first `ceil(0.50 × layers)` blocks → final RMSNorm/head | cheapest approximation |
| E1 | first `ceil(0.75 × layers)` blocks → final RMSNorm/head | intermediate approximation |
| E2 | every imported block → unchanged final RMSNorm/head | full pretrained anchor |
| E3 | full backbone plus bounded recurrent refinement around a configured/profiled middle region | additional difficult-input compute |

E0/E1 optionally enable a small zero-residual bottleneck continuation for frozen-backbone distillation; it is off by default so the direct shallow-exit baseline remains measurable.

Deterministic `EffortComputeReceipt` accounting proves, for supported backbones with at least three resolvable depths:

`C(E0) < C(E1) < C(E2) < C(E3)` and `C_norm(E2) = 1.0`.

`effort_mode="adaptive"` runs a configurable shallow imported-Qwen prefix as a shared probe, pools its internal hidden state, and dispatches each sample to E0–E3 without re-running the prefix. Adaptive execution refuses to run without an installed `VERIFIED_FIT` controller. Policy training itself is blocked until both effort-arm and oracle-opportunity gates pass.

The former final-state E3 remains available as `final_refine`. Canonical `middle_recurrent`, experimental zero-gated `middle_repeat`, and profile-guided selection are research variants. The attached single-layer-RL study motivates the middle-depth prior; it does not establish that recurrence or layer reuse will improve this model. The repository therefore measures contributions on the exact checkpoint and preserves negative findings.

At conversion time all augmentation scales are exactly zero, preserving:

`QwenExFusion(E2) == QwenCompat` to numerical tolerance.

Inspired by architectural principles from Kimi K3, adapted for smaller experimental systems:

- **Sequence mixing**: SelectiveSSM (default continuous state) + periodic global attention
- **Width mixing**: LatentMoE (latent experts + RMSNorm + optional SiTU-GLU + Quantile Balancing)
- **Depth mixing**: BlockAttnRes / AttnResBank
- **Compute budget**: EffortController + cost-aware aux loss + early-exit
- **Merging**: architecture-aware DARE → TIES (pure sign-majority) → Fisher

## Install

```bash
pip install -e .
# or just PYTHONPATH=.
```

## Legacy quick start

```python
from daph import DAPHConfigV3, DAPHHybridModelV3

cfg = DAPHConfigV3(
    hidden_size=256,
    latent_size=128,
    num_layers=6,
    num_recurrent_per_block=3,
    moe_activation="situ",
    use_quantile_balancing=True,
    use_attn_res=True,
)
model = DAPHHybridModelV3(cfg)
out = model(input_ids)  # dict with logits, effort_scores, ...
```

## Tests

```bash
python -m pytest -q
```

## Canonical experiment commands

```bash
# Gate 0A and exact Gate 0B
python scripts/run_phase0_retention.py \
  --hf-model Qwen/Qwen2.5-0.5B-Instruct --hf-revision <commit-sha> \
  --data val.jsonl --output runs/phase0 --phase both

# Synthetic plumbing check (not a source-model qualification)
python scripts/run_phase0_retention.py --synthetic --output runs/phase0_synthetic

# Staged adaptation, per-effort evaluation, counterfactual collection,
# oracle qualification, and policy training use the public Python APIs:
# TrainingStageConfig/train_adapt, eval_per_effort,
# CounterfactualCollector/oracle_analysis, and EffortPolicyTrainer.
```

The immutable experiment sequence is:

HF checkpoint → Gate 0A → QwenCompat → exact Gate 0B → layer profile → E3 hard-case training/ablations → effort qualification → freeze → counterfactual collection → oracle gate → hidden policy → sham/random controls → IID test → leave-family-out OOD test.

See [`docs/PIPELINE_COMMANDS.md`](docs/PIPELINE_COMMANDS.md) for executable examples for every stage.

E3 scientific accounting is specified in [`docs/UTILITY_ACCOUNTING_REPORT.md`](docs/UTILITY_ACCOUNTING_REPORT.md), with the enforced replicated protocol in [`docs/E3_EXPERIMENT_PROTOCOL_V341.md`](docs/E3_EXPERIMENT_PROTOCOL_V341.md), enforcement audit in [`docs/QUALIFICATION_ENFORCEMENT_REPORT_V341.md`](docs/QUALIFICATION_ENFORCEMENT_REPORT_V341.md), and immutable artifact contract in [`docs/EVIDENCE_SCHEMA_V340.md`](docs/EVIDENCE_SCHEMA_V340.md).

## Standalone marginal-utility controller

`daph_metareasoner` is the smaller controller-first research path. It wraps one frozen model with `STOP`, `THINK`, `VERIFY`, and `DECOMPOSE`, collects isolated state/action outcomes, proves oracle conditional value before training, compares hidden-state probes with cheap shams, and permits on-path execution only after paired IID/OOD utility gates pass. It intentionally excludes latent workspaces, specialists, retrieval, and vector-speaking agents.

The first pinned 0.5B-Instruct engineering smoke failed the oracle opportunity gate, so no value controller was trained. See [`docs/MARGINAL_UTILITY_CONTROLLER.md`](docs/MARGINAL_UTILITY_CONTROLLER.md) for the architecture, commands, stop criteria, and negative evidence.

GitHub Actions runs the complete Python compatibility matrix, exact architecture gates, synthetic Phase 0 evidence generation, and package build. See [`docs/CI_WORKFLOW.md`](docs/CI_WORKFLOW.md) for the job graph, artifact contract, and recommended branch protection.

## Real-model smoke result

The initial pinned `Qwen/Qwen2.5-0.5B` + WikiText-2 smoke exposed weak exits and an unstable E3 graph. The corrected run keeps exact E2, improves E0/E1 CE by `1.468`/`0.668`, and changes E3 from a `2.257×` degrading path into a stable `1.009×` final refinement with a small positive CE delta. This is an engineering pass, not yet a router-quality claim.

See [`docs/QUALITY_CORRECTION_REPORT.md`](docs/QUALITY_CORRECTION_REPORT.md) for the root-cause analysis, corrected measurements, limitations, and next workflow. The original failure is retained in [`docs/REAL_MODEL_SMOKE_REPORT.md`](docs/REAL_MODEL_SMOKE_REPORT.md).

The subsequent frozen-E2 hard-case ablation found a teacher-forced E3 CE dose response but no verified E2→E3 rescues on its held-out arithmetic tasks, so E3 remains unqualified. See [`docs/E3_HARDCASE_ABLATION_REPORT.md`](docs/E3_HARDCASE_ABLATION_REPORT.md).

A corrected checkpoint-specific sparse profile then selected layers 12–14. In a small matched-budget smoke comparison, refinement at the profiled middle layer produced a `0.02070` held-out CE gain versus `0.00543` at the final layer (approximately `3.81×` larger at equal compute), but both variants still produced zero exact-answer rescues. See [`docs/E3_PROFILED_MIDDLE_SMOKE_REPORT.md`](docs/E3_PROFILED_MIDDLE_SMOKE_REPORT.md).

The answer-only follow-up calibrated every split to 50% E2 accuracy and compared final, heuristic-middle, and profiled-middle refinement at a matched four-step dose. Heuristic-middle produced the first held-out verified rescue (`1` rescue, `0` regressions) and the largest CE improvement, but its paired 95% quality lower bound remained zero. The historical report did not price per-task compute and is now explicitly a mechanism signal, not utility qualification. E3 and policy training remain unqualified. See [`docs/E3_ANSWER_ONLY_MIXED_RESULT.md`](docs/E3_ANSWER_ONLY_MIXED_RESULT.md).

## Status

The canonical Qwen path, legacy hybrid path, and standalone marginal-utility research package coexist. Engineering acceptance is covered by 167 tests. Experiment tiers, unique-task counts, seed replication, natural-test promotion, and stable-profile promotion are enforced in executable paths. A real but low-budget sparse profile, matched answer-only location study, and negative controller oracle smoke are bundled. Neither E3 task utility nor a learned controller is scientifically qualified.
