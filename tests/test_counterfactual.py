#!/usr/bin/env python3
import sys, os, tempfile, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.config import DAPHConfigV3
from daph.model import DAPHHybridModelV3
from daph.counterfactual import (
    CounterfactualCollector,
    compute_utility,
    soft_targets,
    oracle_analysis,
    full_state_dict_digest,
    causal_ce_quality,
)


def _cfg():
    return DAPHConfigV3(
        hidden_size=64, latent_size=32, num_attention_heads=4, state_size=8,
        num_recurrent_per_block=1, num_routed_experts=4, top_k_experts=2,
        num_layers=2, vocab_size=50, use_attn_res=False, dropout=0.0,
        use_quantile_balancing=False, default_e3_steps=2,
    )


def test_compute_utility_tiebreak():
    u, best, argmax = compute_utility([0.5, 0.5, 0.5, 0.5], [0.3, 0.5, 0.7, 1.0], lambda_cost=0.0)
    assert best == 0
    assert argmax in range(4)
    print("utility + tiebreak OK")


def test_soft_targets():
    p = soft_targets([0.1, 0.9, 0.1, 0.1], temperature=0.1)
    assert abs(sum(p) - 1.0) < 1e-6
    assert p[1] == max(p)
    print("soft targets OK")


def test_full_digest_sensitive():
    cfg = _cfg()
    m = DAPHHybridModelV3(cfg).eval()
    d1 = full_state_dict_digest(m)
    # mutate last parameter
    last = list(m.parameters())[-1]
    with torch.no_grad():
        last.add_(1.0)
    d2 = full_state_dict_digest(m)
    assert d1 != d2, "digest must change when weights change"
    print("full state_dict digest sensitive OK")


def test_causal_ce_shift():
    B, L, V = 1, 5, 20
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    q, st = causal_ce_quality(logits, labels)
    assert st == "CE_PROXY_CAUSAL"
    assert 0.0 <= q <= 1.0
    # pad mask
    labels2 = labels.clone()
    labels2[:, -2:] = 0
    q2, _ = causal_ce_quality(logits, labels2, pad_id=0)
    assert 0.0 <= q2 <= 1.0
    print("causal CE quality OK")


def test_collector_restores_grad():
    cfg = _cfg()
    m = DAPHHybridModelV3(cfg)
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    with CounterfactualCollector(m, freeze=True) as coll:
        assert not m.training
        assert all(not p.requires_grad for p in m.parameters())
        tasks = [{"task_id": "t0", "input_ids": torch.randint(0, 50, (1, 6)),
                  "labels": torch.randint(0, 50, (6,))}]
        recs = coll.collect_many(tasks)
        assert len(recs) == 1
        assert recs[0].model_digest == coll.model_digest
        assert hasattr(recs[0], "argmax_effort")
    # restored
    assert m.training
    assert any(p.requires_grad for p in m.parameters())
    print("collector restore OK")


def test_collector_smoke():
    cfg = _cfg()
    model = DAPHHybridModelV3(cfg).eval()
    tasks = []
    for i in range(4):
        ids = torch.randint(0, 50, (1, 8))
        labels = torch.randint(0, 50, (8,))
        tasks.append({"task_id": f"t{i}", "input_ids": ids, "labels": labels})
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cf.jsonl")
        with CounterfactualCollector(model, lambda_cost=0.15, cost_mode="flops") as coll:
            recs = coll.collect_many(tasks, out_path=path)
        assert len(recs) == 4
        analysis = oracle_analysis(recs, n_bootstrap=50)
        assert "oracle_gap_lcb95" in analysis
        assert "oracle_effort_entropy" in analysis
        assert "has_routing_opportunity" in analysis
        print("collector smoke OK", analysis["best_effort_hist"])
        print("  gap", round(analysis["oracle_gap"], 4),
              "lcb95", round(analysis["oracle_gap_lcb95"], 4),
              "entropy", round(analysis["oracle_effort_entropy"], 3))




def test_bf16_digest():
    cfg = _cfg()
    m = DAPHHybridModelV3(cfg).eval()
    # cast one param to bf16 if available
    p = next(m.parameters())
    try:
        p.data = p.data.to(torch.bfloat16)
    except Exception:
        print("bf16 skip")
        return
    d = full_state_dict_digest(m)
    assert isinstance(d, str) and len(d) == 32
    print("bf16 digest OK")


def test_batch_size_rejected():
    cfg = _cfg()
    m = DAPHHybridModelV3(cfg).eval()
    with CounterfactualCollector(m) as coll:
        try:
            coll.collect_one({"task_id": "x", "input_ids": torch.randint(0, 50, (2, 6))})
            raise AssertionError("should reject B>1")
        except ValueError as e:
            assert "batch size 1" in str(e)
    print("batch size reject OK")


def test_raw_compute_present():
    cfg = _cfg()
    m = DAPHHybridModelV3(cfg).eval()
    with CounterfactualCollector(m) as coll:
        r = coll.collect_one({
            "task_id": "t", "input_ids": torch.randint(0, 50, (1, 6)),
            "labels": torch.randint(0, 50, (6,)),
        })
    assert hasattr(r, "raw_compute")
    assert hasattr(r, "task_digest")
    assert r.raw_compute[3] >= r.raw_compute[0]  # E3 >= E0 absolute
    assert abs(r.compute[3] - 1.0) < 1e-6  # normalized E3 ~ 1
    print("raw_compute + task_digest OK")


if __name__ == "__main__":
    test_compute_utility_tiebreak()
    test_soft_targets()
    test_full_digest_sensitive()
    test_causal_ce_shift()
    test_collector_restores_grad()
    test_collector_smoke()
    test_bf16_digest()
    test_batch_size_rejected()
    test_raw_compute_present()
    print("\nAll counterfactual tests passed.")
