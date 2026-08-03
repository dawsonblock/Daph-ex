#!/usr/bin/env python3
"""Additional v3.1 gates: E3 latent, JSONL vault, training smoke, effort differentiation."""

import sys, os, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.config import DAPHConfigV3
from daph.model import DAPHHybridModelV3
from daph.rfsn_hooks import AppendOnlyJSONLVault, ExFusionEmitter
from daph.train import TrainConfig, train_smoke
from daph.latent_refine import LatentRefineBlock
from daph.merge import merge_task_vectors_dare_ties_fisher


def _cfg(**kw):
    base = dict(
        hidden_size=64, latent_size=32, num_attention_heads=4, state_size=8,
        num_recurrent_per_block=1, num_routed_experts=4, top_k_experts=2,
        num_layers=2, vocab_size=100, enable_channel_gates=True,
        use_attn_res=False, dropout=0.0, use_quantile_balancing=False,
        default_e3_steps=2, max_latent_steps=4,
    )
    base.update(kw)
    return DAPHConfigV3(**base)


def test_e3_runs_more_latent_steps_than_e2():
    cfg = _cfg()
    model = DAPHHybridModelV3(cfg).eval()
    ids = torch.randint(0, 100, (2, 5))
    out2 = model(ids, effort_mode="fixed_2")
    out3 = model(ids, effort_mode="fixed_3")
    assert out2["logits"].shape == out3["logits"].shape
    steps2 = out2["compute_stats"]["latent_steps"]
    steps3 = out3["compute_stats"]["latent_steps"]
    assert all(s == 0 for s in steps2)
    assert all(s == 2 for s in steps3)
    # outputs should differ (refinement changes hidden)
    assert not torch.allclose(out2["logits"], out3["logits"], atol=1e-5)
    print("GATE E3 latent steps > E2 OK")


def test_e0_e2_compute_stats_differ():
    cfg = _cfg(use_global_attention=True)
    model = DAPHHybridModelV3(cfg).eval()
    ids = torch.randint(0, 100, (2, 4))
    o0 = model(ids, effort_mode="fixed_0")
    o2 = model(ids, effort_mode="fixed_2")
    assert o0["compute_stats"]["attention_executed"] == [False, False]
    assert o2["compute_stats"]["attention_executed"] == [True, True]
    assert o0["compute_stats"]["moe_executed"] == [False, False]
    assert o2["compute_stats"]["moe_executed"] == [True, True]
    print("GATE E0 vs E2 compute graphs differ OK")


def test_jsonl_vault_chain():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "events.jsonl")
        vault = AppendOnlyJSONLVault(path, experiment_id="exp-test")
        emitter = ExFusionEmitter(sink=vault, sequence_id="s1")
        cfg = _cfg(num_layers=1)
        model = DAPHHybridModelV3(cfg).eval()
        ids = torch.randint(0, 100, (1, 3))
        model(ids, emitter=emitter)
        assert len(vault) >= 2
        assert vault.verify_chain()
        # tamper
        with open(path, "a") as f:
            f.write('{"event_type":"forged","content_hash":"deadbeef","parent_hash":"0"}\n')
        # recount not updated; verify should fail on full file
        vault2 = AppendOnlyJSONLVault(path, experiment_id="exp-test")
        # chain includes forged line → should fail
        ok = vault2.verify_chain()
        assert ok is False
        print("GATE JSONL hash chain OK")


def test_train_smoke():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        ckpt = train_smoke(
            train_cfg=TrainConfig(
                steps=5, batch_size=2, seq_len=16, log_every=2,
                output_dir=td,
                seed=0, device="cpu",
            )
        )
        assert ckpt["final_loss"] is not None
        assert ckpt["steps"] == 5
        assert ckpt["param_count"] > 0
        print(f"GATE train smoke OK (loss={ckpt['final_loss']:.4f})")


def test_fisher_multi_expert_pipeline():
    # Same-sign experts with different magnitudes; Fisher weight must shift the blend
    tv = [
        {"w": torch.tensor([1.0, 1.0])},
        {"w": torch.tensor([3.0, 3.0])},
    ]
    w = torch.tensor([1.0, 1.0])
    f_hi1 = [{"w": torch.tensor([1e4, 1e4])}, {"w": torch.tensor([1.0, 1.0])}]
    f_hi2 = [{"w": torch.tensor([1.0, 1.0])}, {"w": torch.tensor([1e4, 1e4])}]
    m1 = merge_task_vectors_dare_ties_fisher(
        tv, w, dare_base_p=0.0, fisher_diagonals=f_hi1, use_fisher=True, generator=torch.Generator().manual_seed(0)
    )
    m2 = merge_task_vectors_dare_ties_fisher(
        tv, w, dare_base_p=0.0, fisher_diagonals=f_hi2, use_fisher=True, generator=torch.Generator().manual_seed(0)
    )
    # m1 leans toward expert1 (~1), m2 toward expert2 (~3)
    assert m1["w"].mean().item() < m2["w"].mean().item()
    assert (m1["w"] - m2["w"]).abs().max().item() > 0.5
    print("GATE multi-expert Fisher pipeline OK")


def test_latent_refine_unit():
    block = LatentRefineBlock(32, workspace_slots=4)
    x = torch.randn(2, 5, 32)
    y, ws = block(x, num_steps=3)
    assert y.shape == x.shape
    assert ws is not None and ws.shape == (2, 4)
    y0, _ = block(x, num_steps=0)
    assert torch.allclose(y0, x)
    print("GATE LatentRefineBlock OK")


if __name__ == "__main__":
    test_e3_runs_more_latent_steps_than_e2()
    test_e0_e2_compute_stats_differ()
    test_jsonl_vault_chain()
    test_train_smoke()
    test_fisher_multi_expert_pipeline()
    test_latent_refine_unit()
    print("\nAll v3.1 extended gates passed.")
