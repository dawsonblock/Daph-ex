#!/usr/bin/env python3
"""v3.1 correctness gates: config effect, KDA init, sparse MoE, effort, cache."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import torch
import torch.nn as nn
from daph.config import DAPHConfigV3
from daph.model import DAPHHybridModelV3, ModelCache
from daph.hybrid_block import HybridBlock
from daph.latent_moe import LatentMoE
from daph.kda import KimiDeltaAttention
from daph.merge import difficulty_weighted_fisher_merge, difficulty_weighted_ties_merge


def _tiny_cfg(**kw):
    base = dict(
        hidden_size=64, latent_size=32, num_attention_heads=4, state_size=8,
        num_recurrent_per_block=1, num_routed_experts=4, top_k_experts=2,
        num_layers=2, vocab_size=100, enable_channel_gates=True,
        use_attn_res=False, dropout=0.0, use_quantile_balancing=False,
    )
    base.update(kw)
    return DAPHConfigV3(**base)


def test_use_global_attention_disables_module():
    cfg_on = _tiny_cfg(use_global_attention=True)
    cfg_off = _tiny_cfg(use_global_attention=False)
    b_on = HybridBlock(cfg_on)
    b_off = HybridBlock(cfg_off)
    assert b_on.attn is not None
    assert b_off.attn is None

    calls = {"n": 0}
    def hook(mod, inp, out):
        calls["n"] += 1
    # only on-model has attn
    b_on.attn.register_forward_hook(hook)
    x = torch.randn(2, 5, 64)
    b_on(x)
    assert calls["n"] == 1
    # off path must not have module
    y, meta = b_off(x)
    assert meta.get("attention_executed") is False
    assert y.shape == x.shape
    print("GATE use_global_attention disables path OK")


def test_kda_init_survives_model_construction():
    cfg = _tiny_cfg(recurrent_type="kda", kda_num_heads=4, num_layers=2)
    model = DAPHHybridModelV3(cfg)
    found = 0
    for m in model.modules():
        if isinstance(m, KimiDeltaAttention):
            found += 1
            assert torch.allclose(m.g_proj.weight, torch.zeros_like(m.g_proj.weight)), "g_proj.weight should be 0"
            assert torch.allclose(m.g_proj.bias, torch.ones_like(m.g_proj.bias)), "g_proj.bias should be 1"
            assert torch.allclose(m.o_proj_gate.weight, torch.zeros_like(m.o_proj_gate.weight))
            assert torch.allclose(m.o_proj_gate.bias, torch.zeros_like(m.o_proj_gate.bias))
    assert found >= 2
    print("GATE KDA init survives full model construction OK")


def test_sparse_moe_unused_expert_not_called():
    class Boom(nn.Module):
        def forward(self, x):
            raise RuntimeError("unused expert was executed")

    moe = LatentMoE(hidden_size=32, latent_size=16, num_routed_experts=4, top_k=1,
                    use_quantile_balancing=True)  # enable bias path
    # Force router to always pick expert 0
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.expert_bias.fill_(-100.0)
        moe.expert_bias[0] = 100.0
    for e in range(1, 4):
        moe.routed[e] = Boom()
    moe.eval()  # no load-balance bias mutation during forward
    x = torch.randn(2, 6, 32)
    y, _ = moe(x)
    assert y.shape == x.shape
    tel = moe._last_telemetry
    assert tel["expert_call_count"] == 1
    assert tel["tokens_per_expert"][0] == 2 * 6
    assert tel["tokens_per_expert"][1] == 0
    print("GATE sparse MoE unused expert not called OK")


def test_effort_fixed_0_skips_attention_and_moe():
    cfg = _tiny_cfg(use_global_attention=True, num_layers=1)
    model = DAPHHybridModelV3(cfg).eval()
    # Patch attention and moe to explode
    for layer in model.layers:
        if layer.attn is not None:
            def boom_attn(*a, **k):
                raise RuntimeError("attention should not run in E0")
            layer.attn.forward = boom_attn
        def boom_moe(*a, **k):
            raise RuntimeError("moe should not run in E0")
        layer.moe.forward = boom_moe

    ids = torch.randint(0, 100, (2, 5))
    out = model(ids, effort_mode="fixed_0")
    assert out["logits"].shape == (2, 5, 100)
    assert out["compute_stats"]["attention_executed"] == [False]
    print("GATE effort fixed_0 skips attn+moe OK")


def test_cache_incremental_matches_full():
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_global_attention=True, num_layers=2, recurrent_type="ssm")
    model = DAPHHybridModelV3(cfg).eval()
    ids = torch.randint(0, 100, (1, 6))

    with torch.no_grad():
        full = model(ids)["logits"]

        cache = None
        pieces = []
        for t in range(ids.shape[1]):
            out = model(ids[:, t:t+1], cache=cache, use_cache=True)
            pieces.append(out["logits"])
            cache = out["cache"]
        incr = torch.cat(pieces, dim=1)

    # Allow modest tolerance for recurrent numeric drift; target is << 1e-2
    max_diff = (full - incr).abs().max().item()
    print(f"  cache max_abs_diff={max_diff:.6e}")
    assert max_diff < 5e-3, f"cache mismatch too large: {max_diff}"
    print("GATE cache incremental vs full OK")


def test_fisher_sensitivity():
    # Two experts, opposite deltas; Fisher must change the merge
    tv = [
        {"w": torch.tensor([1.0, 1.0])},
        {"w": torch.tensor([-1.0, -1.0])},
    ]
    # equal weights
    w = torch.tensor([1.0, 1.0])
    # Expert 1 dominates Fisher
    f1 = [{"w": torch.tensor([1e6, 1e6])}, {"w": torch.tensor([1.0, 1.0])}]
    m1 = difficulty_weighted_fisher_merge(tv, f1, w)
    # Expert 2 dominates Fisher
    f2 = [{"w": torch.tensor([1.0, 1.0])}, {"w": torch.tensor([1e6, 1e6])}]
    m2 = difficulty_weighted_fisher_merge(tv, f2, w)
    diff = (m1["w"] - m2["w"]).abs().max().item()
    assert diff > 0.1, f"Fisher must affect merge; diff={diff}"
    # m1 should lean positive, m2 negative
    assert m1["w"].mean() > 0
    assert m2["w"].mean() < 0
    print("GATE Fisher sensitivity OK")


if __name__ == "__main__":
    test_use_global_attention_disables_module()
    test_kda_init_survives_model_construction()
    test_sparse_moe_unused_expert_not_called()
    test_effort_fixed_0_skips_attention_and_moe()
    test_cache_incremental_matches_full()
    test_fisher_sensitivity()
    print("\nAll v3.1 correctness gates passed.")
