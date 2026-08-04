#!/usr/bin/env python3
import sys, os, tempfile, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.config import DAPHConfigV3
from daph.model import DAPHHybridModelV3
from daph.pretrained import (
    import_state_dict,
    build_qwen_key_map,
    research_config,
    PretrainedImportReport,
    save_adapted_checkpoint,
)
from daph.train import TrainConfig, train_smoke, sample_effort_mode
from daph.train_real import RealTrainConfig, train_adapt, load_jsonl_texts


def test_qwen_style_map_and_import():
    cfg = research_config(hidden_size=64, num_layers=2, num_heads=4, vocab_size=128, num_experts=4)
    # override to tiny
    cfg = DAPHConfigV3(
        hidden_size=64, latent_size=32, num_attention_heads=4, state_size=8,
        num_recurrent_per_block=1, num_routed_experts=4, top_k_experts=2,
        num_shared_experts=1, num_layers=2, vocab_size=128,
        use_attn_res=False, dropout=0.0, use_quantile_balancing=False,
        tie_word_embeddings=True,
    )
    model = DAPHHybridModelV3(cfg)
    # synthetic Qwen-like state dict
    H, V, L = 64, 128, 2
    src = {
        "model.embed_tokens.weight": torch.randn(V, H),
        "lm_head.weight": torch.randn(V, H),
        "model.norm.weight": torch.ones(H),
    }
    inter = 128  # matches shared.0.0 out
    for i in range(L):
        src[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(H, H)
        src[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(H, H)
        src[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(H, H)
        src[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(H, H)
        src[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(H)
        src[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(H)
        src[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(inter, H)
        src[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(inter, H)
        src[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(H, inter)
    km = build_qwen_key_map(model.state_dict(), src)
    assert "model.embed_tokens.weight" in km
    assert any("self_attn.q_proj" in k for k in km)
    report = import_state_dict(model, src, source_name="fake-qwen", zero_init_new=True)
    assert report.matched_parameters + report.transformed_parameters > 0
    assert report.coverage_percent > 5.0
    assert hasattr(report, 'exact_coverage_percent')
    assert report.partial_block_parameters == 0  # default: no unsafe block copy
    print(f"qwen map import OK coverage={report.coverage_percent:.1f}% matched={len(report.matched_keys)}")


def test_research_config_size():
    cfg = research_config()
    m = DAPHHybridModelV3(cfg)
    n = sum(p.numel() for p in m.parameters())
    assert n < 80_000_000  # research scale
    print(f"research config params={n:,}")


def test_multi_effort_smoke():
    with tempfile.TemporaryDirectory() as td:
        cfg = TrainConfig(
            steps=8, batch_size=2, seq_len=8,
            effort_mode="sample", effort_probs=(0.25, 0.25, 0.25, 0.25),
            seed=0, output_dir=td, log_every=100,
        )
        mcfg = DAPHConfigV3(
            hidden_size=32, latent_size=16, num_attention_heads=4, state_size=4,
            num_recurrent_per_block=1, num_routed_experts=2, top_k_experts=1,
            num_layers=1, vocab_size=32, use_attn_res=False, dropout=0.0,
            use_quantile_balancing=False, default_e3_steps=1,
        )
        ckpt = train_smoke(mcfg, cfg)
        assert sum(ckpt["effort_hist"].values()) == 8
        print(f"multi-effort smoke OK hist={ckpt['effort_hist']}")


def test_train_adapt_jsonl():
    with tempfile.TemporaryDirectory() as td:
        data = os.path.join(td, "train.jsonl")
        with open(data, "w") as f:
            for i in range(20):
                f.write(json.dumps({"text": f"hello world {i} " * 8}) + "\n")
        cfg = RealTrainConfig(
            steps=6, batch_size=2, seq_len=16, lr=1e-3, lr_pretrained=1e-4,
            warmup_steps=1, log_every=100, eval_every=1000, seed=0,
            effort_mode="sample",
            effort_schedule=("fixed_0", "fixed_1", "fixed_3"),
            data_path=data, output_dir=os.path.join(td, "out"),
            device="cpu",
        )
        mcfg = DAPHConfigV3(
            hidden_size=32, latent_size=16, num_attention_heads=4, state_size=4,
            num_recurrent_per_block=1, num_routed_experts=2, top_k_experts=1,
            num_shared_experts=1, num_layers=1, vocab_size=128,
            use_attn_res=False, dropout=0.0, use_quantile_balancing=False,
        )
        model = DAPHHybridModelV3(mcfg)
        out = train_adapt(model, cfg)
        assert sum(out["effort_hist"].values()) == 6
        assert out["effort_hist"] == {"fixed_0": 2, "fixed_1": 2, "fixed_3": 2}
        assert os.path.isfile(os.path.join(td, "out", "checkpoint_final.pt"))
        print(f"train_adapt jsonl OK hist={out['effort_hist']}")


if __name__ == "__main__":
    test_qwen_style_map_and_import()
    test_research_config_size()
    test_multi_effort_smoke()
    test_train_adapt_jsonl()
    print("\nAll pretrained/train_real tests passed.")
