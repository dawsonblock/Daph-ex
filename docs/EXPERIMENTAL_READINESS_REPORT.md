# DAPH ExFusion v3.1.4 — Experimental Freeze Report

## Status: ENGINEERING FREEZE for first adaptive-compute experiment

Full adaptive loop implemented and tested:

1. Multi-effort execution (E0–E3)
2. Generate + decode + verify
3. Counterfactual collection with full provenance
4. Dataset qualification (uniform model/config/λ/projection; utility integrity)
5. Oracle analysis with bootstrap LCB
6. Experiment roles: Q / train / val / test / OOD (+ leave-family-out)
7. Official `trainer.fit()` → `TrainingReceipt` → `VERIFIED_FIT` artifact
8. Install path with base-model + state + optional source-tree checks
9. Controls: prompt sham, effort-frequency random, compute-matched random ensemble

## Official policy path

```
PolicyTrainingConfig → trainer.fit() → TrainingReceipt
  → build_artifact() [requires receipt] → VERIFIED_FIT
  → install_effort_policy(require_verified_fit=True)
```

Manual `train_epoch` artifacts are labeled `MANUAL_UNVERIFIED` and rejected by default install.

## Next experiment (do not change architecture)

1. Multi-effort base training (router off)
2. Freeze checkpoint + digests
3. Collect verified E0–E3 on D_Q
4. Oracle gate: LCB95(U_oracle − U_best_fixed) > 0
5. If pass: train hidden policy on D_train via fit()
6. Final evaluation vs fixed / sham / compute-matched random on D_test
7. Leave-family-out on D_OOD

## Primary metrics

- ΔU_hidden-fixed
- ΔU_hidden-sham
- ΔU_hidden-matched-random
- GapCapture = (U_hidden − U_best_fixed) / (U_oracle − U_best_fixed)
- Report Q and C separately

## Not established

Scientific evidence that the effort hierarchy is useful. That is the experiment.

## Base-model path (added)

- `daph/pretrained.py` — Qwen/LLaMA-style state_dict map → embed, lm_head, attn, norms, MLP→shared expert
- `research_config()` — ~23M params (not the 466M default)
- `daph/train_real.py` — JSONL text, multi-effort sampling, freeze + differential LR, per-effort val
- `TrainConfig.effort_mode="sample"` with `effort_probs`

### Recommended sequence

1. `research_config()` or dimension-matched config for a chosen Qwen checkpoint
2. `load_pretrained_into_exfusion(...)` / `import_state_dict`
3. Stage 1: freeze imported keys, train new modules
4. Stage 2: `effort_mode="sample"` multi-effort adaptation
5. Freeze base → counterfactual → oracle → policy
