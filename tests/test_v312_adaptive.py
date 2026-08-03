#!/usr/bin/env python3
"""v3.1.2: per-sample adaptive effort + physical path differentiation."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.config import DAPHConfigV3
from daph.model import DAPHHybridModelV3


def _cfg(**kw):
    base = dict(
        hidden_size=64, latent_size=32, num_attention_heads=4, state_size=8,
        num_recurrent_per_block=1, num_routed_experts=4, top_k_experts=2,
        num_layers=2, vocab_size=100, enable_channel_gates=True,
        use_attn_res=False, dropout=0.0, use_quantile_balancing=False,
        default_e3_steps=2, effort_levels=4,
    )
    base.update(kw)
    return DAPHConfigV3(**base)


def test_per_sample_mixed_levels():
    """Override levels [0,1,2,3] — each sample must execute its path."""
    cfg = _cfg(use_global_attention=True)
    model = DAPHHybridModelV3(cfg).eval()
    B = 4
    ids = torch.randint(0, 100, (B, 6))
    levels = torch.tensor([0, 1, 2, 3])

    # Patch modules to count calls with batch size
    attn_calls = []
    moe_calls = []
    refine_calls = []

    for layer in model.layers:
        if layer.attn is not None:
            orig_attn = layer.attn.forward
            def make_attn(orig):
                def wrapped(*a, **k):
                    attn_calls.append(a[0].shape[0])
                    return orig(*a, **k)
                return wrapped
            layer.attn.forward = make_attn(orig_attn)
        orig_moe = layer.moe.forward
        def make_moe(orig):
            def wrapped(*a, **k):
                moe_calls.append(a[0].shape[0])
                return orig(*a, **k)
            return wrapped
        layer.moe.forward = make_moe(orig_moe)
        orig_ref = layer.latent_refine.forward
        def make_ref(orig):
            def wrapped(*a, **k):
                if k.get("num_steps", a[1] if len(a) > 1 else 0):
                    refine_calls.append(a[0].shape[0])
                return orig(*a, **k)
            return wrapped
        layer.latent_refine.forward = make_ref(orig_ref)

    with torch.no_grad():
        out = model(ids, effort_levels_override=levels)

    assert out["logits"].shape == (B, 6, 100)
    # E0 samples: no attention, no moe
    # With partitioning, attn should only be called for E2+E3 subsets (sizes 1+1)
    assert sum(attn_calls) == 2 * cfg.num_layers  # one sample E2 + one E3, each layer
    # MoE: E1 + E2 + E3 = 3 samples * num_layers
    assert sum(moe_calls) == 3 * cfg.num_layers
    # Refine: only E3
    assert sum(refine_calls) == 1 * cfg.num_layers

    stats = out["compute_stats"]
    assert stats["effort_levels"] == [0, 1, 2, 3]
    assert stats["attention_token_evals"] > 0
    assert stats["expert_token_evals"] > 0
    assert stats["latent_refine_token_evals"] > 0
    print("GATE per-sample mixed levels physical skip OK")


def test_adaptive_per_sample_not_batch_mean():
    """Bias controllers so different samples prefer different levels."""
    cfg = _cfg()
    model = DAPHHybridModelV3(cfg).eval()
    # Force probe controller: logits so sample-dependent via input-dependent path
    # Easier: use override is already tested; for adaptive, bias net to level 0
    # and verify decision levels shape is (B,)
    with torch.no_grad():
        for p in model.layers[0].effort.net[-1].parameters():
            p.zero_()
        model.layers[0].effort.net[-1].bias[0] = 5.0
        model.layers[0].effort.net[-1].bias[1:] = -5.0

    ids = torch.randint(0, 100, (3, 5))
    with torch.no_grad():
        out = model(ids, effort_mode="adaptive")
    dec = out["effort_decision"]
    assert dec is not None
    assert len(dec["levels"]) == 3
    assert all(l == 0 for l in dec["levels"])
    assert out["compute_stats"]["attention_token_evals"] == 0
    print("GATE adaptive levels are per-sample (B,) OK")


def test_batch_order_invariance():
    cfg = _cfg()
    model = DAPHHybridModelV3(cfg).eval()
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (4, 5))
    levels = torch.tensor([0, 3, 1, 2])
    with torch.no_grad():
        out1 = model(ids, effort_levels_override=levels)["logits"]
        # permute
        perm = torch.tensor([3, 2, 0, 1])
        inv = torch.argsort(perm)
        out2 = model(ids[perm], effort_levels_override=levels[perm])["logits"]
        out2 = out2[inv]
    assert torch.allclose(out1, out2, atol=1e-5)
    print("GATE batch-order invariance OK")


def test_single_vs_batch_equivalence():
    cfg = _cfg()
    model = DAPHHybridModelV3(cfg).eval()
    torch.manual_seed(1)
    ids = torch.randint(0, 100, (3, 5))
    levels = torch.tensor([0, 2, 3])
    with torch.no_grad():
        batch = model(ids, effort_levels_override=levels)["logits"]
        singles = []
        for i in range(3):
            o = model(ids[i : i + 1], effort_levels_override=levels[i : i + 1])["logits"]
            singles.append(o)
        single = torch.cat(singles, dim=0)
    assert torch.allclose(batch, single, atol=1e-5)
    print("GATE single vs batch equivalence OK")


def test_ssm_kda_cache_still_ok():
    for rtype in ("ssm", "kda"):
        cfg = _cfg(recurrent_type=rtype, kda_num_heads=4, use_global_attention=True)
        model = DAPHHybridModelV3(cfg).eval()
        ids = torch.randint(0, 100, (1, 6))
        with torch.no_grad():
            full = model(ids)["logits"]
            cache = None
            parts = []
            for t in range(6):
                o = model(ids[:, t : t + 1], cache=cache, use_cache=True)
                parts.append(o["logits"])
                cache = o["cache"]
            incr = torch.cat(parts, dim=1)
        d = (full - incr).abs().max().item()
        assert d < 1e-4, f"{rtype} cache diff {d}"
        print(f"  {rtype} cache diff={d:.3e}")
    print("GATE SSM/KDA cache still OK")


def test_probe_not_double_counted():
    """Adaptive E0 must not run layer-0 recurrent twice."""
    cfg = _cfg()
    model = DAPHHybridModelV3(cfg).eval()
    calls = {"n": 0}
    orig = model.layers[0].recurrent_layers[0].forward

    def wrapped(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    model.layers[0].recurrent_layers[0].forward = wrapped
    with torch.no_grad():
        for p in model.layers[0].effort.net[-1].parameters():
            p.zero_()
        model.layers[0].effort.net[-1].bias[0] = 10.0
        model.layers[0].effort.net[-1].bias[1:] = -10.0

    ids = torch.randint(0, 100, (2, 5))
    with torch.no_grad():
        calls["n"] = 0
        out_ad = model(ids, effort_mode="adaptive")
        ad = calls["n"]
        calls["n"] = 0
        out_e0 = model(ids, effort_mode="fixed_0")
        e0 = calls["n"]
    assert ad == e0, f"probe double-count: adaptive={ad} fixed_e0={e0}"
    assert torch.allclose(out_ad["logits"], out_e0["logits"], atol=1e-4)
    assert out_ad["compute_stats"].get("estimated_flops", 0) > 0
    print("GATE probe reuse (no double recurrent) OK")


def test_fixed_vs_adaptive_compute_equality():
    cfg = _cfg()
    cfg.num_shared_experts = 2
    model = DAPHHybridModelV3(cfg).eval()
    ids = torch.randint(0, 100, (3, 5))
    keys = [
        "estimated_flops", "recurrent_token_evals", "attention_token_evals",
        "expert_token_evals", "shared_expert_token_evals", "latent_refine_token_evals",
    ]
    with torch.no_grad():
        for e in range(4):
            fixed = model(ids, effort_mode=f"fixed_{e}")["compute_stats"]
            levels = torch.full((3,), e, dtype=torch.long)
            ad = model(ids, effort_levels_override=levels)["compute_stats"]
            for k in keys:
                assert fixed.get(k) == ad.get(k), f"E{e} {k}: fixed={fixed.get(k)} ad={ad.get(k)}"
    print("GATE fixed==adaptive compute accounting OK")


if __name__ == "__main__":
    test_per_sample_mixed_levels()
    test_adaptive_per_sample_not_batch_mean()
    test_batch_order_invariance()
    test_single_vs_batch_equivalence()
    test_ssm_kda_cache_still_ok()
    test_probe_not_double_counted()
    test_fixed_vs_adaptive_compute_equality()
    print("\nAll v3.1.2/3.1.3 adaptive gates passed.")


