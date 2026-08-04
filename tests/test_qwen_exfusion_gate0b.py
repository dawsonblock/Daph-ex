#!/usr/bin/env python3
"""Gate 0B: QwenExFusion E2 ≡ QwenCompat when augmentation scales are zero."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.qwen_compat import QwenCompatModel
from daph.qwen_exfusion import (
    QwenExFusionModel,
    augment_qwen_compat_model,
    gate0b_exact_parity,
)


def test_gate0b_exact_after_augment():
    torch.manual_seed(0)
    H, L, V, I, nq, nkv = 32, 2, 64, 64, 4, 2
    compat = QwenCompatModel(V, H, L, nq, nkv, I)
    exf = augment_qwen_compat_model(compat, num_routed_experts=2, top_k=1)
    ids = torch.randint(0, V, (3, 12))
    mask = torch.ones(3, 12, dtype=torch.long)
    m = gate0b_exact_parity(compat, exf, ids, mask)
    assert m["passed"], m
    assert m["logit_mae"] == 0.0
    print(f"Gate 0B exact OK mae={m['logit_mae']}")


def test_nonzero_scale_breaks_parity():
    torch.manual_seed(1)
    compat = QwenCompatModel(64, 32, 1, 4, 2, 64)
    exf = augment_qwen_compat_model(compat, num_routed_experts=2, top_k=1)
    with torch.no_grad():
        exf.layers[-1].latent_scale.fill_(1.0)
    ids = torch.randint(0, 64, (2, 6))
    m = gate0b_exact_parity(compat, exf, ids)  # zeros scales again
    # gate0b zeros scales — so should still pass
    assert m["passed"]
    # without zeroing, parity should fail
    with torch.no_grad():
        exf.layers[-1].latent_scale.fill_(0.5)
    logits_c = compat(ids)
    logits_e = exf(ids, effort_mode="fixed_3")
    assert not torch.allclose(logits_c, logits_e, atol=1e-5)
    print("nonzero scale changes logits OK")


def test_scales_init_zero():
    m = QwenExFusionModel(32, 32, 1, 4, 2, 64)
    for layer in m.layers:
        assert float(layer.rec_scale.detach()) == 0.0
        assert float(layer.moe_scale.detach()) == 0.0
        assert float(layer.latent_scale.detach()) == 0.0
    print("scales init zero OK")


if __name__ == "__main__":
    test_scales_init_zero()
    test_gate0b_exact_after_augment()
    test_nonzero_scale_breaks_parity()
    print("\nAll Gate 0B tests passed.")
