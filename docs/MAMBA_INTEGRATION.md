# Mamba integration for ExFusion

## Placement rule

Mamba / SelectiveSSM is a **zero-scaled residual augmentation**, not a replacement for the Qwen E2 backbone.

```
QwenCompat (E2 anchor)
  + rec_scale × SSM/Mamba   (init 0)
  + moe_scale × routed MoE  (init 0)
  + latent_scale × refine  (init 0)
```

Gate 0B requires: with all scales = 0, `QwenExFusion(E2) ≡ QwenCompat`.

## Options

| Option | What | When |
|--------|------|------|
| **A. Scan backend** | `mamba_ssm.selective_scan_fn` under `SelectiveSSM` | After Gate 0B; GPU training |
| **B. Full Mamba/Mamba2 block** | Replace `self.recurrent` module | If E0 needs more capacity |
| **C. HF MambaMixer** | transformers path | Experiments only |
| **D. Eager/compile** | Default in-repo scan | CI, Phase 0, CPU |

## Enable Option A

```bash
pip install mamba-ssm causal-conv1d --no-build-isolation
export DAPH_SCAN_BACKEND=mamba_ssm
```

Or in Python:

```python
from daph.mamba_backend import try_enable_mamba_backend
try_enable_mamba_backend()
```

## Validation

1. Eager vs mamba_ssm scan max error on fixed tensors (should be ~1e-4 FP32).
2. Gate 0B still passes with backend enabled (scales zero → no SSM path effect on E2).
3. AR: `(state, x_t) → (y_t, state')` matches full-sequence slice.

## Do not

- Put Mamba on the E2 Qwen path before retention passes
- Require CUDA kernels for Phase 0A/0B unit tests
- Train from scratch instead of zero-scaled augmentation
