#!/usr/bin/env python3
import sys, os, tempfile, copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from daph.effort import EffortController
from daph.counterfactual import EffortCounterfactual, CounterfactualCollector
from daph.config import DAPHConfigV3
from daph.model import DAPHHybridModelV3
from daph.verifiers import ExactMatchVerifier, make_quality_fn
from daph.policy_trainer import (
    EffortPolicyTrainer,
    ShamEffortController,
    EffortPolicyArtifact,
    effort_frequency_matched_random,
    compute_matched_random,
    MatchedRandomResult,
    dataset_digest,
    evaluate_policy_utility,
    gap_capture,
    make_split_manifest,
    apply_split,
    validate_counterfactual_dataset,
    install_effort_policy,
    make_experiment_manifest,
    make_leave_family_out_manifest,
    PolicyTrainingConfig,
    TrainingReceipt,
)


def _fake_records(n=32, h=64, id_offset=0):
    from daph.counterfactual import compute_utility
    recs = []
    for i in range(n):
        peak = i % 4
        q = [0.1, 0.1, 0.1, 0.1]
        q[peak] = 0.9
        hvec = (torch.randn(h) * 0.05).tolist()
        for k in range(4):
            hvec[k] = 1.0 if k == peak else 0.0
        scale = 1.0 + (i % 5)
        raw = (0.3 * scale, 0.5 * scale, 0.7 * scale, 1.0 * scale)
        compute = (0.3, 0.5, 0.7, 1.0)
        utils, best, argmax = compute_utility(q, compute, lambda_cost=0.15, tie_epsilon=0.01)
        recs.append(
            EffortCounterfactual(
                task_id=f"t{id_offset + i}",
                input_digest="x",
                task_digest=f"td{id_offset + i}",
                probe_hidden=hvec,
                quality=tuple(q),
                compute=compute,
                raw_compute=raw,
                utility=utils,
                best_effort=best,
                argmax_effort=argmax,
                verifier_status=("CORRECT",) * 4,
                model_digest="mdigest",
                config_digest="c",
                lambda_cost=0.15,
                tie_epsilon=0.01,
            )
        )
    return recs


def test_seed_reproducibility():
    recs = _fake_records(32, h=64)
    def train_once(seed):
        torch.manual_seed(0)  # same init
        ctrl = EffortController(64, num_levels=4)
        # force identical init
        sd0 = {k: v.clone() for k, v in ctrl.state_dict().items()}
        t1 = EffortPolicyTrainer(ctrl, lr=1e-2, temperature=0.05, seed=seed)
        for _ in range(5):
            t1.train_epoch(recs, batch_size=8)
        return t1.controller.state_dict()

    # reset and train twice with same seed → same weights
    torch.manual_seed(0)
    c1 = EffortController(64, num_levels=4)
    sd_init = {k: v.clone() for k, v in c1.state_dict().items()}
    t1 = EffortPolicyTrainer(c1, lr=1e-2, seed=42)
    for _ in range(5):
        t1.train_epoch(recs, batch_size=8)
    d1 = {k: v.clone() for k, v in t1.controller.state_dict().items()}

    torch.manual_seed(0)
    c2 = EffortController(64, num_levels=4)
    c2.load_state_dict(sd_init)
    t2 = EffortPolicyTrainer(c2, lr=1e-2, seed=42)
    for _ in range(5):
        t2.train_epoch(recs, batch_size=8)
    d2 = t2.controller.state_dict()
    for k in d1:
        assert torch.allclose(d1[k], d2[k]), k
    print("training_seed actually used OK")


def test_policy_learns_held_out():
    train = _fake_records(64, h=64, id_offset=0)
    test = _fake_records(32, h=64, id_offset=1000)
    ctrl = EffortController(64, num_levels=4)
    trainer = EffortPolicyTrainer(ctrl, lr=1e-2, temperature=0.05, seed=0)
    before = trainer.evaluate(test)
    for _ in range(40):
        trainer.train_epoch(train, batch_size=16)
    after = trainer.evaluate(test)
    assert after.top1_acc_argmax > before.top1_acc_argmax or after.top1_acc_argmax >= 0.45
    print(f"policy held-out OK after_acc={after.top1_acc_argmax:.2f}")


def test_splits_no_leakage():
    recs = _fake_records(50)
    m = make_split_manifest(recs, seed=7, train_frac=0.6, val_frac=0.2)
    m.assert_no_leakage()
    tr, va, te = apply_split(recs, m)
    assert len(tr) + len(va) + len(te) == len(recs)
    print(f"split OK train={len(tr)} val={len(va)} test={len(te)}")


def test_validate_dataset():
    recs = _fake_records(10)
    good = validate_counterfactual_dataset(recs, require_all_verified=True, expected_model_digest="mdigest")
    assert len(good) == 10
    d = recs[0].to_dict()
    d["verifier_status"] = ("UNVERIFIABLE",) * 4
    bad = [EffortCounterfactual(**d)] + recs[1:]
    report = validate_counterfactual_dataset(bad, require_all_verified=True, return_report=True)
    assert report.accepted == 9 and report.dropped_unverifiable == 1
    # mixed config must fail
    d2 = recs[1].to_dict()
    d2["config_digest"] = "other_cfg"
    mixed = [recs[0], EffortCounterfactual(**d2)] + list(recs[2:])
    try:
        validate_counterfactual_dataset(mixed)
        raise AssertionError("should reject mixed config")
    except ValueError as e:
        assert "mixed_config" in str(e) or "config" in str(e).lower()
    # duplicate task_digest rejected by split
    try:
        make_split_manifest([recs[0], recs[0]])
        raise AssertionError("should reject dup")
    except ValueError:
        pass
    # mixed projection rejected
    d3 = recs[2].to_dict()
    d3["projection_digest"] = "projA"
    d4 = recs[3].to_dict()
    d4["projection_digest"] = "projB"
    try:
        validate_counterfactual_dataset(
            [EffortCounterfactual(**d3), EffortCounterfactual(**d4)] + list(recs[4:])
        )
        raise AssertionError("should reject mixed projection")
    except ValueError as e:
        assert "projection" in str(e).lower()
    print("validate_counterfactual_dataset OK")


def test_policy_artifact_and_install():
    from daph.policy_trainer import PolicyTrainingConfig, source_tree_digest
    train = _fake_records(16)
    val = _fake_records(8, id_offset=200)
    ctrl = EffortController(64, num_levels=4)
    cfg = PolicyTrainingConfig(seed=3, epochs=4, batch_size=8, feature_spec="hidden", lr=1e-2)
    trainer = EffortPolicyTrainer(ctrl, config=cfg)
    trainer.authorize_policy_training(
        {"qualified": True}, {"has_routing_opportunity": True}
    )
    m, receipt = trainer.fit(train, val)
    assert isinstance(receipt, type(trainer._last_receipt))
    assert receipt.epochs_requested == 4
    assert receipt.epochs_completed == 4
    assert receipt.batch_size == 8
    assert receipt.examples_seen == 16 * 4
    art, sd = trainer.build_artifact(
        base_model_digest="abc123",
        train_records=train,
        validation_records=val,
        metrics=m,
        split_manifest_digest="splitx",
    )
    assert art.training_seed == 3
    assert art.training_config_digest == cfg.digest()
    assert art.training_receipt is not None
    assert art.training_receipt["epochs_completed"] == 4
    assert art.source_digest == source_tree_digest()
    assert art.environment.get("torch")
    fresh = EffortController(64, num_levels=4)
    install_effort_policy(fresh, art, sd, base_model_digest="abc123")
    for k in sd:
        assert torch.allclose(fresh.state_dict()[k], sd[k])
    try:
        install_effort_policy(fresh, art, sd, base_model_digest="wrong")
        raise AssertionError("should reject")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "pol.pt")
        art.save(path, sd)
        art2, sd2 = EffortPolicyArtifact.load(path)
        assert art2.training_receipt["epochs_completed"] == 4
    print("artifact + install + fit receipt OK")





def test_family_metadata_propagated():
    from daph.config import DAPHConfigV3
    from daph.model import DAPHHybridModelV3
    from daph.counterfactual import CounterfactualCollector
    cfg = DAPHConfigV3(
        hidden_size=64, latent_size=32, num_attention_heads=4, state_size=8,
        num_recurrent_per_block=1, num_routed_experts=4, top_k_experts=2,
        num_layers=2, vocab_size=50, use_attn_res=False, dropout=0.0,
        use_quantile_balancing=False, default_e3_steps=1,
    )
    model = DAPHHybridModelV3(cfg).eval()
    with CounterfactualCollector(model) as coll:
        r = coll.collect_one({
            "task_id": "fam1",
            "input_ids": torch.randint(0, 50, (1, 4)),
            "labels": torch.randint(0, 50, (1, 4)),
            "task_family": "arith",
            "template_id": "add_v1",
            "difficulty_bucket": "easy",
            "generator_version": "g1",
        })
    assert r.task_family == "arith"
    assert r.template_id == "add_v1"
    assert r.difficulty_bucket == "easy"
    assert r.generator_version == "g1"
    print("family metadata propagated OK")


def test_experiment_manifest_and_lfo():
    recs = _fake_records(40)
    # attach families
    from daph.counterfactual import EffortCounterfactual
    tagged = []
    for i, r in enumerate(recs):
        d = r.to_dict()
        d["task_family"] = "arith" if i < 20 else "pattern"
        d["template_id"] = f"tpl{i % 5}"
        tagged.append(EffortCounterfactual(**d))
    m = make_experiment_manifest(tagged, seed=1)
    m.assert_disjoint()
    assert len(m.all_digests()) == 40
    lfo = make_leave_family_out_manifest(
        tagged, family_key=lambda r: r.task_family or "x", held_out_family="pattern", seed=2
    )
    lfo.assert_disjoint()
    assert set(lfo.ood_task_digests) == {r.task_digest for r in tagged if r.task_family == "pattern"}
    assert lfo.ood_is_true_ood is True
    m2 = make_experiment_manifest(tagged, seed=3)
    assert m2.ood_is_true_ood is False
    print(f"experiment manifest OK Q={len(m.qualification_task_digests)} OOD={len(lfo.ood_task_digests)}")


def test_require_fit_receipt():
    train = _fake_records(8)
    ctrl = EffortController(64, num_levels=4)
    trainer = EffortPolicyTrainer(ctrl, seed=1)
    trainer.train_epoch(train, batch_size=4)  # manual, no fit
    m = trainer.evaluate(train)
    try:
        trainer.build_artifact(
            base_model_digest="x",
            train_records=train,
            metrics=m,
            require_training_receipt=True,
        )
        raise AssertionError("should require fit")
    except ValueError as e:
        assert "TrainingReceipt" in str(e) or "fit" in str(e).lower()
    # explicit opt-out allowed
    art, _ = trainer.build_artifact(
        base_model_digest="x",
        train_records=train,
        metrics=m,
        require_training_receipt=False,
    )
    assert art.training_status == "MANUAL_UNVERIFIED"
    print("require fit receipt OK")


def test_compute_matched_random():
    chosen = [3, 0, 3, 0, 2, 1]
    raw = [
        [10, 20, 50, 1000],
        [5, 8, 10, 12],
        [10, 20, 50, 1000],
        [5, 8, 10, 12],
        [20, 30, 40, 50],
        [15, 20, 25, 30],
    ]
    result = compute_matched_random(chosen, raw, seed=0, n_candidates=500, tol=0.15)
    assert result.matched is True
    try:
        compute_matched_random([3, 0], [[1, 2, 3, 1000], [1, 2, 3, 4]],
                               seed=0, n_candidates=0, tol=1e-9, require_match=True)
        raise AssertionError("should raise")
    except ValueError:
        pass
    print("compute-matched random OK")


def test_tokenizer_end_to_end():
    class FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return "42"
    cfg = DAPHConfigV3(
        hidden_size=64, latent_size=32, num_attention_heads=4, state_size=8,
        num_recurrent_per_block=1, num_routed_experts=4, top_k_experts=2,
        num_layers=2, vocab_size=50, use_attn_res=False, dropout=0.0,
        use_quantile_balancing=False, default_e3_steps=1,
    )
    model = DAPHHybridModelV3(cfg).eval()
    with CounterfactualCollector(model, tokenizer=FakeTok(), quality_fn=make_quality_fn(ExactMatchVerifier())) as coll:
        r = coll.collect_one_generate(
            {"task_id": "t0", "input_ids": torch.randint(0, 50, (1, 4)), "expected": "42"},
            max_new_tokens=3,
        )
    assert all(s == "CORRECT" for s in r.verifier_status)
    print("tokenizer end-to-end OK")


if __name__ == "__main__":
    test_seed_reproducibility()
    test_policy_learns_held_out()
    test_splits_no_leakage()
    test_validate_dataset()
    test_policy_artifact_and_install()
    test_require_fit_receipt()
    test_family_metadata_propagated()
    test_experiment_manifest_and_lfo()
    test_compute_matched_random()
    test_tokenizer_end_to_end()
    print("\nAll policy trainer tests passed.")
