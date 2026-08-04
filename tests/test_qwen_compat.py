#!/usr/bin/env python3
"""Qwen-compatibility primitives: RoPE, GQA, RMSNorm, SharedSwiGLU, gate identity."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.attention import CausalSelfAttention, RotaryEmbedding, apply_rotary_pos_emb
from daph.norms import RMSNorm
from daph.gates import ChannelGate
from daph.config import DAPHConfigV3
from daph.model import DAPHHybridModelV3
from daph.pretrained import import_state_dict, build_qwen_key_map
from daph.latent_moe import SharedSwiGLU


def test_rms_norm_no_mean_center():
    rms = RMSNorm(16)
    x = torch.randn(2, 4, 16)
    y = rms(x)
    # scale only — not zero-mean
    assert y.shape == x.shape
    assert not torch.allclose(y.mean(dim=-1), torch.zeros(2, 4), atol=1e-3)
    print("RMSNorm OK")


def test_rope_changes_by_position():
    rope = RotaryEmbedding(32, max_position=128, base=10000.0)
    q = torch.randn(1, 2, 4, 32)
    k = torch.randn(1, 2, 4, 32)
    cos0, sin0 = rope(4, position_offset=0)
    cos8, sin8 = rope(4, position_offset=8)
    q0, k0 = apply_rotary_pos_emb(q, k, cos0, sin0)
    q8, k8 = apply_rotary_pos_emb(q, k, cos8, sin8)
    assert not torch.allclose(q0, q8)
    print("RoPE position-dependent OK")


def test_gqa_shapes():
    attn = CausalSelfAttention(
        hidden_size=64, num_heads=8, num_key_value_heads=2,
        use_rope=True, rope_theta=10000.0,
    )
    assert attn.k_proj.weight.shape == (2 * 8, 64)  # n_kv * head_dim, H
    assert attn.q_proj.weight.shape == (8 * 8, 64)
    x = torch.randn(2, 5, 64)
    out, pk, pv = attn(x, use_cache=True)
    assert out.shape == (2, 5, 64)
    assert pk.shape == (2, 2, 5, 8)  # B, n_kv, L, head_dim
    # decode step
    x2 = torch.randn(2, 1, 64)
    out2, pk2, pv2 = attn(x2, past_k=pk, past_v=pv, use_cache=True, position_offset=5)
    assert out2.shape == (2, 1, 64)
    assert pk2.shape[2] == 6
    print("GQA + RoPE cache OK")


def test_qwen2_projection_bias_layout():
    attn = CausalSelfAttention(
        hidden_size=32, num_heads=4, num_key_value_heads=2,
        bias=True, out_bias=False,
    )
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.out_proj.bias is None


def test_shared_swiglu_map():
    cfg = DAPHConfigV3(
        hidden_size=64, latent_size=32, num_attention_heads=4, num_key_value_heads=2,
        state_size=8, num_recurrent_per_block=1, num_routed_experts=2, top_k_experts=1,
        num_shared_experts=1, num_layers=1, vocab_size=128, intermediate_size=128,
        shared_ffn="swiglu", use_rope=True, norm_type="rms", use_attn_res=False,
        dropout=0.0, use_quantile_balancing=False, use_load_balancing=False,
    )
    model = DAPHHybridModelV3(cfg)
    # Qwen-like tensors
    H, I, V = 64, 128, 128
    src = {
        "model.embed_tokens.weight": torch.randn(V, H),
        "lm_head.weight": torch.randn(V, H),
        "model.norm.weight": torch.ones(H),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(H, H),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(2 * (H // 4), H),  # 2 kv heads
        "model.layers.0.self_attn.v_proj.weight": torch.randn(2 * (H // 4), H),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(H, H),
        "model.layers.0.input_layernorm.weight": torch.ones(H),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(H),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(I, H),
        "model.layers.0.mlp.up_proj.weight": torch.randn(I, H),
        "model.layers.0.mlp.down_proj.weight": torch.randn(H, I),
    }
    report = import_state_dict(model, src, source_name="fake-qwen-gqa", zero_init_new=True)
    assert "layers.0.moe.shared.0.gate_proj.weight" in report.matched_keys
    assert "layers.0.moe.shared.0.up_proj.weight" in report.matched_keys
    assert "layers.0.moe.shared.0.down_proj.weight" in report.matched_keys
    assert "layers.0.attn.k_proj.weight" in report.matched_keys
    # identity gates on attn/moe
    g = model.layers[0].gate_attn
    # out_proj should be near identity after identity_init
    eye = torch.eye(H)
    assert torch.allclose(g.out_proj.weight.detach(), eye, atol=1e-5)
    print(f"SwiGLU+GQA map OK coverage={report.coverage_percent:.1f}% exact={report.exact_coverage_percent:.1f}%")


def test_gate_identity_vs_zero():
    g = ChannelGate(32)
    g.identity_init()
    x = torch.randn(1, 4, 32)
    branch = torch.randn(1, 4, 32)
    y = g(x, branch)
    # roughly preserves branch
    assert y.norm() > 0.5 * branch.norm()
    g2 = ChannelGate(32)
    g2.zero_out_init()
    y2 = g2(x, branch)
    assert y2.norm() < 1e-5
    print("gate identity/zero OK")


if __name__ == "__main__":
    test_rms_norm_no_mean_center()
    test_rope_changes_by_position()
    test_gqa_shapes()
    test_gate_identity_vs_zero()
    test_shared_swiglu_map()
    print("\nAll Qwen-compat tests passed.")
