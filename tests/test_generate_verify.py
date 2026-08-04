#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.config import DAPHConfigV3
from daph.model import DAPHHybridModelV3
from daph.verifiers import ExactMatchVerifier, NumericVerifier, FinalAnswerVerifier, make_quality_fn
from daph.counterfactual import CounterfactualCollector


def _cfg(**kw):
    base = dict(
        hidden_size=64, latent_size=32, num_attention_heads=4, state_size=8,
        num_recurrent_per_block=1, num_routed_experts=4, top_k_experts=2,
        num_layers=2, vocab_size=50, use_attn_res=False, dropout=0.0,
        use_quantile_balancing=False, default_e3_steps=2, use_global_attention=True,
    )
    base.update(kw)
    return DAPHConfigV3(**base)


def test_generate_greedy():
    model = DAPHHybridModelV3(_cfg()).eval()
    ids = torch.randint(0, 50, (1, 4))
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=5, effort_mode="fixed_2")
    assert out["sequences"].shape[1] == 4 + 5
    assert out["generated_ids"].shape == (1, 5)
    print("generate greedy OK")


def test_padded_equals_unpadded():
    """Critical: pad must not change greedy continuation for SSM and KDA, all efforts."""
    prompt = torch.tensor([[3, 7, 11]])
    padded = torch.tensor([[3, 7, 11, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    for rtype in ("ssm", "kda"):
        for e in range(4):
            torch.manual_seed(0)
            cfg = _cfg(recurrent_type=rtype, kda_num_heads=4)
            model = DAPHHybridModelV3(cfg).eval()
            with torch.no_grad():
                a = model.generate(prompt, max_new_tokens=5, effort_mode=f"fixed_{e}")
                b = model.generate(
                    padded, attention_mask=mask, max_new_tokens=5, effort_mode=f"fixed_{e}"
                )
            assert torch.equal(a["generated_ids"], b["generated_ids"]), (
                f"{rtype} E{e} pad changed generation: "
                f"{a['generated_ids'].tolist()} vs {b['generated_ids'].tolist()}"
            )
    print("padded == unpadded generation OK (SSM+KDA, E0-E3)")


def test_generate_all_efforts():
    model = DAPHHybridModelV3(_cfg()).eval()
    ids = torch.randint(0, 50, (1, 3))
    with torch.no_grad():
        for e in range(4):
            out = model.generate(ids, max_new_tokens=3, effort_mode=f"fixed_{e}")
            assert out["generated_ids"].shape[1] == 3
    print("generate all efforts OK")


def test_generate_ssm_kda():
    for rtype in ("ssm", "kda"):
        model = DAPHHybridModelV3(_cfg(recurrent_type=rtype, kda_num_heads=4)).eval()
        ids = torch.randint(0, 50, (1, 3))
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=4, effort_mode="fixed_2")
        assert out["sequences"].shape[1] == 7
    print("generate SSM/KDA OK")


def test_exact_match_strict():
    v = ExactMatchVerifier()
    assert v({"generated_text": "42"}, {"expected": "42"}) == (1.0, "CORRECT")
    assert v({"generated_text": "142"}, {"expected": "42"}) == (0.0, "INCORRECT")
    # token ids alone are unverifiable
    assert v({"generated_ids": torch.tensor([[5, 42]])}, {"expected": "42"}) == (0.0, "UNVERIFIABLE")
    print("ExactMatch strict + no-ID OK")


def test_numeric_requires_text():
    v = NumericVerifier()
    assert v({"generated_text": "answer is 42"}, {"expected": 42}) == (1.0, "CORRECT")
    assert v({"generated_ids": torch.tensor([[5, 42]])}, {"expected": 42}) == (0.0, "UNVERIFIABLE")
    print("NumericVerifier text-only OK")


def test_final_answer_verifier():
    v = FinalAnswerVerifier()
    assert v({"generated_text": "thinking...\nAnswer: 42"}, {"expected": "42"})[0] == 1.0
    print("FinalAnswerVerifier OK")


def test_collect_generate_with_verifier():
    model = DAPHHybridModelV3(_cfg()).eval()
    # without generated_text, should be UNVERIFIABLE not false correct
    v = ExactMatchVerifier()
    task = {
        "task_id": "g0",
        "input_ids": torch.randint(0, 50, (1, 4)),
        "expected": "42",
        "verifier_spec": {"name": "exact"},
    }
    with CounterfactualCollector(model, quality_fn=make_quality_fn(v)) as coll:
        r = coll.collect_one_generate(task, max_new_tokens=3)
    assert all(s == "UNVERIFIABLE" for s in r.verifier_status)
    print("collect_one_generate UNVERIFIABLE without text OK")


if __name__ == "__main__":
    test_generate_greedy()
    test_padded_equals_unpadded()
    test_generate_all_efforts()
    test_generate_ssm_kda()
    test_exact_match_strict()
    test_numeric_requires_text()
    test_final_answer_verifier()
    test_collect_generate_with_verifier()
    print("\nAll generate/verify tests passed.")
