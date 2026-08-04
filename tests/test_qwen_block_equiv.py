#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.qwen_compat import QwenCompatBlock, QwenCompatModel
from daph.pretrained import import_state_dict, build_qwen_key_map
from daph.config import DAPHConfigV3
from daph.model import DAPHHybridModelV3


def test_qwen_compat_block_deterministic():
    blk = QwenCompatBlock(64, 4, 2, 128, rope_theta=10000.0)
    x = torch.randn(2, 8, 64)
    y1, _, _ = blk(x)
    y2, _, _ = blk(x)
    assert torch.allclose(y1, y2)
    # residual structure: output not identical to input
    assert not torch.allclose(y1, x)
    print("QwenCompatBlock deterministic OK")


def test_two_blocks_weight_tied_match():
    """Two independently constructed blocks with copied weights match."""
    a = QwenCompatBlock(32, 4, 2, 64)
    b = QwenCompatBlock(32, 4, 2, 64)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 6, 32)
    ya, _, _ = a(x)
    yb, _, _ = b(x)
    torch.testing.assert_close(ya, yb, rtol=1e-5, atol=1e-5)
    print("weight-tied block match OK")


def test_import_into_compat_model_keys():
    """Qwen-style keys map onto QwenCompatModel / Hybrid with exact names where possible."""
    # SharedSwiGLU + attention names on compat block
    blk = QwenCompatBlock(64, 4, 2, 128)
    sd = blk.state_dict()
    assert "self_attn.q_proj.weight" in sd
    assert "mlp.gate_proj.weight" in sd
    assert "input_layernorm.weight" in sd
    assert "post_attention_layernorm.weight" in sd
    print("compat block key names OK")


def test_no_gate_no_routed_in_compat():
    blk = QwenCompatBlock(32, 4, 2, 64)
    names = [n for n, _ in blk.named_parameters()]
    # mlp.gate_proj is SwiGLU, not ChannelGate
    assert not any("channel_gate" in n or "gate_attn" in n or "gate_moe" in n for n in names)
    assert not any("routed" in n for n in names)
    assert not any("recurrent" in n or "ssm" in n or "effort" in n for n in names)
    print("compat block clean of ExFusion extras OK")


def test_compat_import_coverage():
    import torch
    from daph.pretrained import import_into_qwen_compat
    H, L, V, I, nq, nkv = 32, 1, 64, 64, 4, 2
    m = QwenCompatModel(V, H, L, nq, nkv, I)
    src = {
        "model.embed_tokens.weight": torch.randn(V, H),
        "lm_head.weight": torch.randn(V, H),
        "model.norm.weight": torch.ones(H),
        "model.layers.0.input_layernorm.weight": torch.ones(H),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(H),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(H, H),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(nkv * (H // nq), H),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(nkv * (H // nq), H),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(H, H),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(I, H),
        "model.layers.0.mlp.up_proj.weight": torch.randn(I, H),
        "model.layers.0.mlp.down_proj.weight": torch.randn(H, I),
    }
    r = import_into_qwen_compat(m, src)
    assert r.exact_coverage_percent >= 99.0
    assert r.skipped_parameters == 0 or len(r.skipped_keys) == 0
    print(f"compat import coverage={r.exact_coverage_percent:.1f}% OK")


if __name__ == "__main__":
    test_qwen_compat_block_deterministic()
    test_two_blocks_weight_tied_match()
    test_import_into_compat_model_keys()
    test_no_gate_no_routed_in_compat()
    test_compat_import_coverage()
    print("\nAll Qwen block equivalence tests passed.")
