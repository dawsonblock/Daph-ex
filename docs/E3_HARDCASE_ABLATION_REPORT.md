# Frozen-E2 hard-case E3 ablation

## Result: not qualified

This targeted experiment tested whether final-layer latent refinement can rescue a frozen `Qwen/Qwen2.5-0.5B` E2 anchor on disjoint composed-arithmetic tasks. It does **not** establish a useful E3 arm yet.

- Source revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Training / selection / held-out tasks: 128 / 32 / 32
- E2 was frozen and its probe logits were exactly unchanged after every variant.
- Post-Gate-0B activation: raw latent scale `1e-3`.
- Trained parameters: final `latent_refine.fc1`, `latent_refine.fc2`, and `latent_scale`; the identity-initialized refinement LayerNorm weight was intentionally frozen because its MPS backward gradient was non-finite.

| Latent steps | Selection E3–E2 completion CE | Rescues | Regressions | Net rescue rate | Mean `||Δh||` |
|---:|---:|---:|---:|---:|---:|
| 1 | -0.01976 | 0 | 0 | 0.000 | 1.327 |
| 2 | -0.03384 | 0 | 0 | 0.000 | 2.887 |
| 4 | -0.05636 | 0 | 0 | 0.000 | 5.777 |

Four steps won selection on completion CE. On the held-out split, it reduced completion CE from `2.68890` to `2.61412` (`Δ=-0.07477`) at `+0.03575` normalized deterministic compute. However, both E2 and E3 had 0/32 exact-answer accuracy, yielding zero rescues and zero regressions.

The experiment therefore shows a controlled dose response in teacher-forced completion CE and hidden-state movement, but **no verified task benefit**. The hard-case E3 qualification stays failed and router training remains blocked.

Raw local artifact: `runs/e3-hardcase-v1/ablation/e3_hardcase_ablation_report.json` (SHA-256 `1f493f08dec05e87a8dbceeaa435c86d57920a224384c0e243d1ed6fbb0e17c8`). The task manifests and their hashes are recorded in [`../evidence/e3_hardcase_v1/manifest.json`](../evidence/e3_hardcase_v1/manifest.json).

## Interpretation

The final-state refinement can improve likelihood of the supplied numeric targets without changing greedy exact answers. That means the next experiment should improve task supervision—not loosen the qualification gate. In particular, use answer-token-only loss, longer hard-output curricula, and at least one independent non-arithmetic verifier-backed family before considering an earlier insertion-point ablation.
