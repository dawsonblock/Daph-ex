#!/usr/bin/env python3
"""v3.1.1 blocker fixes: KDA conv cache, adaptive dispatch, TIES→Fisher masks."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.config import DAPHConfigV3
from daph.model import DAPHHybridModelV3
from daph.kda import KimiDeltaAttention
from daph.merge import (
    difficulty_weighted_ties_merge,
    merge_task_vectors_dare_ties_fisher,
)


def _cfg(**kw):
    base = dict(
        hidden_size=64, latent_size=32, num_attention_heads=4, state_size=8,
        num_recurrent_per_block=1, num_routed_experts=4, top_k_experts=2,
        num_layers=2, vocab_size=100, enable_channel_gates=True,
        use_attn_res=False, dropout=0.0, use_quantile_balancing=False,
        default_e3_steps=2,
    )
    base.update(kw)
    return DAPHConfigV3(**base)


def _cache_diff(model, ids):
    with torch.no_grad():
        full = model(ids)["logits"]
        cache = None
        parts = []
        for t in range(ids.shape[1]):
            out = model(ids[:, t : t + 1], cache=cache, use_cache=True)
            parts.append(out["logits"])
            cache = out["cache"]
        incr = torch.cat(parts, dim=1)
    return (full - incr).abs().max().item()


def test_kda_cache_matches_full():
    torch.manual_seed(0)
    cfg = _cfg(recurrent_type="kda", kda_num_heads=4, use_global_attention=True)
    model = DAPHHybridModelV3(cfg).eval()
    ids = torch.randint(0, 100, (1, 8))
    d = _cache_diff(model, ids)
    print(f"  KDA+attn cache max_abs_diff={d:.6e}")
    assert d < 1e-4, f"KDA cache broken: {d}"

    cfg2 = _cfg(recurrent_type="kda", kda_num_heads=4, use_global_attention=False)
    model2 = DAPHHybridModelV3(cfg2).eval()
    d2 = _cache_diff(model2, ids)
    print(f"  KDA-only cache max_abs_diff={d2:.6e}")
    assert d2 < 1e-4, f"KDA-only cache broken: {d2}"
    print("GATE KDA conv-state cache OK")


def test_adaptive_not_identical_to_disabled():
    torch.manual_seed(1)
    cfg = _cfg(use_global_attention=True)
    model = DAPHHybridModelV3(cfg).eval()
    # Bias first-layer effort controller toward level 0
    with torch.no_grad():
        # force high logit on level 0
        for p in model.layers[0].effort.net[-1].parameters():
            p.zero_()
        model.layers[0].effort.net[-1].bias[0] = 10.0
        model.layers[0].effort.net[-1].bias[1:] = -10.0

    ids = torch.randint(0, 100, (2, 5))
    with torch.no_grad():
        out_ad = model(ids, effort_mode="adaptive")
        out_dis = model(ids, effort_mode="disabled")
        out_e0 = model(ids, effort_mode="fixed_0")

    # adaptive should have chosen level 0
    assert out_ad["compute_stats"]["chosen_effort_level"] == 0
    assert out_ad["compute_stats"]["attention_executed"] == [False, False]
    # adaptive ≠ disabled (disabled is full)
    assert not torch.allclose(out_ad["logits"], out_dis["logits"], atol=1e-5)
    # adaptive ≈ fixed_0
    assert torch.allclose(out_ad["logits"], out_e0["logits"], atol=1e-5)
    print("GATE adaptive dispatches to E0–E3 OK")


def test_ties_masks_block_minority_fisher():
    """Majority sign +1; minority -1 with huge Fisher must NOT flip after TIES masks."""
    tv = [
        {"w": torch.tensor([1.0])},
        {"w": torch.tensor([1.0])},
        {"w": torch.tensor([-1.0])},
    ]
    w = torch.tensor([1.0, 1.0, 1.0])
    # TIES alone elects +1
    ties = difficulty_weighted_ties_merge(tv, w, trim_ratio=0.0, ssm_soft_merge=False)
    assert torch.allclose(ties["w"], torch.tensor([1.0])), ties["w"]

    # Huge Fisher on minority expert
    fish = [
        {"w": torch.tensor([1.0])},
        {"w": torch.tensor([1.0])},
        {"w": torch.tensor([1e9])},
    ]
    merged = merge_task_vectors_dare_ties_fisher(
        tv, w, dare_base_p=0.0, fisher_diagonals=fish, use_fisher=True, trim_ratio=0.0
    )
    # With TIES masks, minority cannot contribute → still positive
    assert merged["w"].item() > 0, f"minority Fisher overrode TIES: {merged['w']}"
    print("GATE TIES masks constrain Fisher OK")


if __name__ == "__main__":
    test_kda_cache_matches_full()
    test_adaptive_not_identical_to_disabled()
    test_ties_masks_block_minority_fisher()
    print("\nAll v3.1.1 blocker gates passed.")
