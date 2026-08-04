# DAPH / ExFusion v3

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
| E3 | E2 plus final-layer bounded latent delta refinement | additional difficult-input compute |

E0/E1 optionally enable a small zero-residual bottleneck continuation for frozen-backbone distillation; it is off by default so the direct shallow-exit baseline remains measurable.

Deterministic `EffortComputeReceipt` accounting proves, for supported backbones with at least three resolvable depths:

`C(E0) < C(E1) < C(E2) < C(E3)` and `C_norm(E2) = 1.0`.

`effort_mode="adaptive"` runs the first imported Qwen block as a shared probe, pools its post-block hidden state, and dispatches each sample to E0–E3 without re-running that block. The probe is common work already included in every fixed-arm receipt. A controller must be trained and installed before adaptive results are scientifically interpreted; fixed-arm qualification remains the prerequisite for policy training.

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

HF checkpoint → Gate 0A → QwenCompat → QwenExFusion conversion → exact Gate 0B → staged multi-effort adaptation → effort qualification → freeze → counterfactual collection → oracle gate → hidden policy → sham/random controls → IID test → leave-family-out OOD test.

See [`docs/PIPELINE_COMMANDS.md`](docs/PIPELINE_COMMANDS.md) for executable examples for every stage.

GitHub Actions runs the complete Python compatibility matrix, exact architecture gates, synthetic Phase 0 evidence generation, and package build. See [`docs/CI_WORKFLOW.md`](docs/CI_WORKFLOW.md) for the job graph, artifact contract, and recommended branch protection.

## Real-model smoke result

The initial pinned `Qwen/Qwen2.5-0.5B` + WikiText-2 smoke exposed weak exits and an unstable E3 graph. The corrected run keeps exact E2, improves E0/E1 CE by `1.468`/`0.668`, and changes E3 from a `2.257×` degrading path into a stable `1.009×` final refinement with a small positive CE delta. This is an engineering pass, not yet a router-quality claim.

See [`docs/QUALITY_CORRECTION_REPORT.md`](docs/QUALITY_CORRECTION_REPORT.md) for the root-cause analysis, corrected measurements, limitations, and next workflow. The original failure is retained in [`docs/REAL_MODEL_SMOKE_REPORT.md`](docs/REAL_MODEL_SMOKE_REPORT.md).

The subsequent frozen-E2 hard-case ablation found a teacher-forced E3 CE dose response but no verified E2→E3 rescues on its held-out arithmetic tasks, so E3 remains unqualified. See [`docs/E3_HARDCASE_ABLATION_REPORT.md`](docs/E3_HARDCASE_ABLATION_REPORT.md).

## Status

The canonical Qwen path and legacy hybrid path coexist. AttnRes is deliberately disabled in the first canonical pretrained experiment until model-level cross-layer history is implemented. The full local suite currently passes 102 tests.
