#!/usr/bin/env python3
"""Smoke tests for DAPH / ExFusion v3 core components."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import torch
import torch.nn as nn
from daph.config import DAPHConfigV3
from daph.latent_moe import LatentMoE, situ_glu
from daph.gates import ChannelGate
from daph.effort import EffortController, effort_cost_loss, early_exit_mask_from_effort
from daph.hybrid_block import HybridBlock
from daph.ssm import SelectiveSSM
from daph.kda import KimiDeltaAttention, lower_bound_decay
from daph.attn_res import BlockAttnRes, AttnResBank
from daph.model import DAPHHybridModelV3
from daph.rfsn_hooks import ExFusionEmitter, InMemoryRFSNSink, ImmutableVaultSink, attach_emitter_to_model
from daph.benchmarks import benchmark_hybrid_block
from daph.merge import (
    extract_task_vectors,
    apply_dare_preprocessing,
    difficulty_weighted_ties_merge,
    merge_expert_modules,
    is_ssm_core_param,
)


def test_latent_moe():
    moe = LatentMoE(hidden_size=64, latent_size=32, num_routed_experts=8, top_k=2)
    x = torch.randn(2, 5, 64)
    y, logits = moe(x, return_router_logits=True)
    assert y.shape == x.shape
    assert logits is not None and logits.shape == (2, 5, 8)
    print("LatentMoE OK")


def test_situ_glu_and_qb():
    # SiTU-GLU activation path
    moe = LatentMoE(
        hidden_size=64, latent_size=32, num_routed_experts=6, top_k=2,
        activation="situ", beta_gate=1.2, beta_up=1.8,
        use_quantile_balancing=True, qb_bias_lr=0.05,
    )
    x = torch.randn(4, 7, 64)
    moe.train()
    bias_before = moe.expert_bias.detach().clone()
    y, logits = moe(x, return_router_logits=True)
    assert y.shape == x.shape
    # bias should have moved during training forward
    assert not torch.allclose(moe.expert_bias, bias_before)
    # situ_glu unit
    g = torch.randn(3, 8)
    u = torch.randn(3, 8)
    out = situ_glu(g, u, 1.5, 1.5)
    assert out.shape == g.shape
    assert torch.isfinite(out).all()
    print("SiTU-GLU + Quantile Balancing OK")


def test_channel_gate():
    gate = ChannelGate(64)
    x = torch.randn(2, 5, 64)
    branch = torch.randn(2, 5, 64)
    y = gate(x, branch)
    assert y.shape == x.shape
    print("ChannelGate OK")


def test_effort():
    ctrl = EffortController(64, num_levels=4)
    x = torch.randn(2, 5, 64)
    info = ctrl(x)
    assert info["effort_probs"].shape == (2, 4)
    assert info["effort_score"].shape == (2,)
    assert info["expected_cost"].shape == (2,)
    print("EffortController OK")


def test_effort_cost_loss():
    ctrl = EffortController(64, num_levels=4)
    x = torch.randn(4, 8, 64)
    info = ctrl(x)
    quality = torch.randn(4)
    components = effort_cost_loss(info, quality, lambda_cost=0.15)
    assert "effort_aux_loss" in components
    assert components["effort_aux_loss"].ndim == 0
    mask = early_exit_mask_from_effort(info, threshold=0.3)
    assert mask.shape == (4,)
    print("effort_cost_loss + early_exit_mask OK")


def test_selective_ssm():
    ssm = SelectiveSSM(32, 8).eval()
    x = torch.randn(2, 6, 32)
    y, state = ssm(x)
    assert y.shape == x.shape
    assert state.shape == (2, 32, 8)
    y2, _ = ssm(x[:, :1], state=state)
    y3, _ = ssm(x[:, :1])
    assert not torch.allclose(y2, y3)
    st0 = torch.randn(2, 32, 8)
    _, st1 = ssm(x, state=st0, mask=torch.zeros(2, 6), bypass_decay=0.0)
    assert torch.allclose(st1, st0.float())
    print("SelectiveSSM OK")


def test_hybrid_block():
    cfg = DAPHConfigV3(
        hidden_size=64,
        latent_size=32,
        num_attention_heads=4,
        state_size=8,
        num_recurrent_per_block=2,
        num_routed_experts=8,
        top_k_experts=2,
        enable_channel_gates=True,
        moe_activation="situ",
        use_quantile_balancing=True,
    )
    block = HybridBlock(cfg).eval()
    x = torch.randn(2, 6, 64)
    y, meta = block(x, use_cache=True)
    assert y.shape == x.shape
    assert "effort_score" in meta
    assert "recurrent_states" in meta
    assert len(meta["recurrent_states"]) == 2
    print("HybridBlock (with real SelectiveSSM) OK")


def test_hybrid_block_early_exit():
    cfg = DAPHConfigV3(
        hidden_size=64,
        latent_size=32,
        num_attention_heads=4,
        state_size=8,
        num_recurrent_per_block=1,
        num_routed_experts=4,
        top_k_experts=1,
        enable_channel_gates=True,
    )
    block = HybridBlock(cfg).eval()
    x = torch.randn(3, 5, 64)
    y, meta = block(x, enable_early_exit=True, early_exit_threshold=0.0)
    assert y.shape == x.shape
    assert "early_exit_mask" in meta or "early_exited" in meta
    print("HybridBlock early-exit path OK")


def test_block_attn_res():
    res = BlockAttnRes(hidden_size=64, max_blocks=4, num_heads=4)
    current = torch.randn(2, 5, 64)
    out0, _ = res(current, [])
    assert torch.allclose(out0, torch.zeros_like(current))
    hist = [torch.randn(2, 5, 64) for _ in range(3)]
    out, attn = res(current, hist)
    assert out.shape == current.shape
    assert attn.shape == (2, 4, 5, 3)
    print("BlockAttnRes OK")


def test_attn_res_bank():
    bank = AttnResBank(hidden_size=64, max_blocks=4, num_heads=4, gate=True)
    x = torch.randn(2, 5, 64)
    y0 = bank(x)
    assert y0.shape == x.shape
    bank.push(x)
    bank.push(torch.randn_like(x))
    y1 = bank(x)
    assert y1.shape == x.shape
    bank.reset()
    assert len(bank._history) == 0
    print("AttnResBank OK")


def test_full_model():
    cfg = DAPHConfigV3(
        hidden_size=64,
        latent_size=32,
        num_attention_heads=4,
        state_size=8,
        num_recurrent_per_block=2,
        num_routed_experts=4,
        top_k_experts=2,
        num_layers=3,
        vocab_size=100,
        enable_channel_gates=True,
        use_attn_res=True,
        num_attn_res_blocks=4,
        moe_activation="situ",
    )
    model = DAPHHybridModelV3(cfg).eval()
    ids = torch.randint(0, 100, (2, 7))
    out = model(ids, use_cache=True)
    assert out["logits"].shape == (2, 7, 100)
    assert len(out["past_recurrent_states"]) == 3
    assert len(out["effort_scores"]) == 3
    out2 = model(ids[:, -1:], past_recurrent_states=out["past_recurrent_states"], use_cache=True)
    assert out2["logits"].shape == (2, 1, 100)
    print("DAPHHybridModelV3 (with AttnRes) OK")


def test_ties_sign_majority():
    majority_vectors = [
        {"weight": torch.tensor([10.0])},
        {"weight": torch.tensor([-1.0])},
        {"weight": torch.tensor([-1.0])},
    ]
    result = difficulty_weighted_ties_merge(
        majority_vectors,
        torch.tensor([1.0, 1.0, 1.0]),
        trim_ratio=0.0,
        ssm_soft_merge=False,
    )
    assert torch.allclose(result["weight"], torch.tensor([-1.0])), result["weight"]
    print("TIES pure sign-majority (outlier cannot dominate) OK")


def test_dare_and_merge_modules():
    base = nn.Linear(8, 8, bias=False)
    experts = [copy.deepcopy(base) for _ in range(3)]
    for e in experts:
        e.weight.data.add_(torch.randn_like(e.weight) * 0.1)
    tvs = extract_task_vectors(experts, base)
    assert len(tvs) == 3
    processed, masks = apply_dare_preprocessing(tvs, dare_base_p=0.3)
    assert len(processed) == 3
    merged_model = merge_expert_modules(experts, base, dare_base_p=0.2, trim_ratio=0.1)
    assert isinstance(merged_model, nn.Linear)
    assert not torch.allclose(merged_model.weight, base.weight)
    print("DARE + merge_expert_modules OK")


def test_ssm_param_heuristic():
    assert is_ssm_core_param("layers.0.recurrent_layers.0.A_log")
    assert is_ssm_core_param("ssm.dt_proj.weight")
    assert not is_ssm_core_param("lm_head.weight")
    assert not is_ssm_core_param("layers.0.attn_norm.bias")
    print("is_ssm_core_param heuristic OK")



def test_kda():
    kda = KimiDeltaAttention(32, num_heads=4, g_min=-5.0).eval()
    x = torch.randn(2, 6, 32)
    y, state = kda(x)
    assert y.shape == x.shape
    assert isinstance(state, tuple) and state[0].dim() == 4  # (S, hist_k, hist_v)
    # state affects next step
    y2, _ = kda(x[:, :1], state=state)
    y3, _ = kda(x[:, :1])
    assert not torch.allclose(y2, y3, atol=1e-5)
    # lower_bound_decay range
    raw = torch.linspace(-10, 10, 20)
    g = lower_bound_decay(raw, g_min=-5.0)
    assert (g > 0).all() and (g <= 1).all()
    assert g.min() >= torch.sigmoid(torch.tensor(-5.0)) - 1e-5
    # masked bypass
    st0 = state
    _, st1 = kda(x, state=st0, mask=torch.zeros(2, 6), bypass_decay=0.0)
    assert torch.allclose(st1[0], st0[0], atol=1e-5)
    if st0[1] is not None:
        assert torch.allclose(st1[1], st0[1], atol=1e-5)
    print("KDA (KimiDeltaAttention) OK")


def test_hybrid_block_kda():
    cfg = DAPHConfigV3(
        hidden_size=64,
        latent_size=32,
        num_attention_heads=4,
        state_size=8,
        num_recurrent_per_block=2,
        num_routed_experts=4,
        top_k_experts=2,
        enable_channel_gates=True,
        recurrent_type="kda",
        kda_num_heads=4,
    )
    block = HybridBlock(cfg).eval()
    x = torch.randn(2, 5, 64)
    y, meta = block(x, use_cache=True)
    assert y.shape == x.shape
    assert "recurrent_states" in meta
    assert len(meta["recurrent_states"]) == 2
    print("HybridBlock with KDA recurrent OK")



def test_rfsn_hooks():
    sink = InMemoryRFSNSink()
    emitter = ExFusionEmitter(sink=sink, sequence_id="test-seq")
    cfg = DAPHConfigV3(
        hidden_size=64,
        latent_size=32,
        num_attention_heads=4,
        state_size=8,
        num_recurrent_per_block=1,
        num_routed_experts=4,
        top_k_experts=2,
        num_layers=2,
        vocab_size=50,
        enable_channel_gates=True,
        use_attn_res=False,
    )
    model = DAPHHybridModelV3(cfg).eval()
    ids = torch.randint(0, 50, (2, 5))
    out = model(ids, emitter=emitter)
    assert out["logits"].shape == (2, 5, 50)
    # Should have effort events per layer + forward summary
    assert len(sink) >= 3
    efforts = sink.query(event_type="effort")
    assert len(efforts) == 2
    forwards = sink.query(event_type="forward")
    assert len(forwards) == 1
    assert forwards[0].sequence_id == "test-seq"
    assert forwards[0].payload["num_layers"] == 2
    print("RFSN hooks (emitter + sink) OK")



def test_immutable_vault():
    vault = ImmutableVaultSink(salience_half_life_s=60.0)
    emitter = ExFusionEmitter(sink=vault, sequence_id="vault-test")
    cfg = DAPHConfigV3(
        hidden_size=64, latent_size=32, num_attention_heads=4, state_size=8,
        num_recurrent_per_block=1, num_routed_experts=4, top_k_experts=2,
        num_layers=2, vocab_size=50, use_attn_res=False,
    )
    model = DAPHHybridModelV3(cfg).eval()
    ids = torch.randint(0, 50, (1, 4))
    model(ids, emitter=emitter)
    assert len(vault) >= 3
    assert vault.verify_integrity()
    current = vault.as_of()
    assert len(current) >= 3
    # supersede one
    eid = current[0].event.event_id
    assert vault.supersede(eid)
    assert vault.get(eid).valid_to is not None
    # as-of now should exclude superseded
    still = [r for r in vault.as_of() if r.event.event_id == eid]
    assert len(still) == 0
    vault.decay_salience()
    print("ImmutableVaultSink (bi-temporal + integrity) OK")


def test_benchmark_smoke():
    r = benchmark_hybrid_block(
        recurrent_type="ssm", hidden_size=64, batch_size=2, seq_len=16,
        num_runs=3, warmup=1, device="cpu",
    )
    assert r.mean_ms > 0
    assert r.tokens_per_s > 0
    print(f"Benchmark smoke OK ({r.mean_ms:.2f} ms, {r.tokens_per_s:.0f} tok/s)")


if __name__ == "__main__":
    test_latent_moe()
    test_situ_glu_and_qb()
    test_channel_gate()
    test_effort()
    test_effort_cost_loss()
    test_selective_ssm()
    test_hybrid_block()
    test_hybrid_block_early_exit()
    test_block_attn_res()
    test_attn_res_bank()
    test_full_model()
    test_ties_sign_majority()
    test_dare_and_merge_modules()
    test_ssm_param_heuristic()
    test_kda()
    test_hybrid_block_kda()
    test_rfsn_hooks()
    test_immutable_vault()
    test_benchmark_smoke()
    print("\nAll v3 smoke tests passed.")
