"""Hard gates for the canonical physical E0 < E1 < E2 < E3 hierarchy."""

import json
import os
import tempfile

import torch

from daph.qwen_compat import QwenCompatModel
from daph.qwen_exfusion import (
    QwenExFusionBlock,
    augment_qwen_compat_model,
    gate0b_exact_parity,
    load_qwen_exfusion_checkpoint,
    prepare_exfusion_for_training,
)
from daph.pretrained import save_adapted_checkpoint
from daph.counterfactual import CounterfactualCollector, _tensor_raw_bytes, full_state_dict_digest
from daph.train_real import (
    RealTrainConfig, TrainingStageConfig, apply_training_stage,
    distillation_loss, finite_and_clip_gradients, train_adapt,
)


def _model():
    torch.manual_seed(7)
    compat = QwenCompatModel(96, 32, 4, 4, 2, 64)
    return compat, augment_qwen_compat_model(
        compat, num_routed_experts=2, top_k=1, default_e3_steps=1
    )


def test_effort_compute_ordering_and_exact_e2():
    compat, model = _model()
    ids = torch.randint(0, 96, (2, 9))
    receipts = [model.compute_receipt(ids, f"fixed_{e}") for e in range(4)]
    costs = [r.estimated_compute for r in receipts]
    layers = [r.executed_layer_count for r in receipts]
    assert costs[0] < costs[1] < costs[2] < costs[3]
    assert layers[0] < layers[1] < layers[2] == layers[3]
    assert receipts[2].normalized_compute_cost == 1.0
    assert gate0b_exact_parity(compat, model, ids)["decision"] == "PASS_EXACT"


def test_disabled_branches_are_not_called():
    _, model = _model()
    ids = torch.randint(0, 96, (1, 7))
    calls = {"rec": 0, "moe": 0, "latent": 0}
    for layer in model.layers:
        for key, module in (("rec", layer.recurrent), ("moe", layer.routed_moe), ("latent", layer.latent_refine)):
            original = module.forward
            def wrapped(*args, _key=key, _original=original, **kwargs):
                calls[_key] += 1
                return _original(*args, **kwargs)
            module.forward = wrapped
    model(ids, effort_mode="fixed_0")
    model(ids, effort_mode="fixed_1")
    model(ids, effort_mode="fixed_2")
    assert calls == {"rec": 0, "moe": 0, "latent": 0}
    model(ids, effort_mode="fixed_3")
    assert calls == {"rec": 0, "moe": 0, "latent": 1}


def test_shallow_exit_backprop_and_distillation_are_finite():
    _, model = _model()
    ids = torch.randint(0, 96, (2, 8))
    labels = ids.clone()
    student = model(ids, effort_mode="fixed_0")
    with torch.no_grad():
        teacher = model(ids, effort_mode="fixed_2")
    loss, pieces = distillation_loss(student, teacher, labels, beta=0.7, temperature=2.0)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(v)) for v in pieces.values())
    loss.backward()
    assert model.layers[0].base.mlp.down_proj.weight.grad is not None
    assert model.layers[-1].base.mlp.down_proj.weight.grad is None


def test_refinement_scales_delta_not_full_representation():
    block = QwenExFusionBlock(32, 4, 2, 64, num_routed_experts=2, top_k=1)
    class AddTwo(torch.nn.Module):
        def forward(self, x, num_steps=1):
            return x + 2.0, None
    block.latent_refine = AddTwo()
    block.latent_scale.data.fill_(0.5)
    x = torch.randn(1, 5, 32)
    base, _, _ = block.base(x)
    out, _, _, _ = block(
        x, use_recurrent=False, use_routed_moe=False,
        use_attn_res=False, latent_steps=1,
    )
    expected_scale = 0.01 * torch.tanh(torch.tensor(0.5 / 0.01))
    assert torch.allclose(out, base + 2.0 * expected_scale, atol=1e-6)
    assert float((out - base).detach().abs().max()) <= 0.020001


def test_training_init_is_explicit_and_enables_augmentation_gradients():
    _, model = _model()
    receipt = prepare_exfusion_for_training(model, gate0b_passed=True, epsilon=1e-3)
    assert receipt.backbone_unchanged and receipt.changed_scale_names
    ids = torch.randint(0, 96, (2, 7))
    model(ids, effort_mode="fixed_3").sum().backward()
    insertion = model.e3_region.insertion_layer
    assert receipt.changed_scale_names == (f"layers.{insertion}.latent_scale",)
    assert model.layers[insertion].latent_refine.fc2.weight.grad is not None
    assert model.layers[insertion].latent_refine.fc2.weight.grad.abs().sum() > 0
    assert model.layers[0].latent_refine.fc2.weight.grad is None


def test_hidden_state_distillation_is_finite_and_backpropagates():
    _, model = _model()
    ids = torch.randint(0, 96, (2, 8))
    labels = ids.clone()
    student = model(ids, effort_mode="fixed_0", return_hidden_state=True)
    with torch.no_grad():
        teacher = model(ids, effort_mode="fixed_2", return_hidden_state=True)
    loss, pieces = distillation_loss(
        student["logits"], teacher["logits"], labels,
        beta=0.7, temperature=2.0,
        student_hidden=student["hidden_state"],
        teacher_hidden=teacher["hidden_state"], hidden_weight=1.0,
    )
    assert torch.isfinite(loss)
    assert pieces["hidden_mse"] > 0
    loss.backward()
    assert model.layers[0].base.mlp.down_proj.weight.grad is not None


def test_parameter_provenance_is_exact_names():
    _, model = _model()
    provenance = model.parameter_provenance
    names = {n for n, _ in model.named_parameters()}
    assert provenance is not None
    assert set(provenance.imported_parameter_names).isdisjoint(provenance.new_parameter_names)
    assert set(provenance.imported_parameter_names) | set(provenance.new_parameter_names) == names
    assert all(name.endswith("_scale") for name in provenance.scale_parameter_names)
    insertion = model.e3_region.insertion_layer
    assert provenance.e3_scale_parameter_names == (f"layers.{insertion}.latent_scale",)
    assert all(name.startswith(f"layers.{insertion}.latent_refine.") for name in provenance.e3_refinement_parameter_names)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "canonical.pt")
        save_adapted_checkpoint(model, path)
        loaded = load_qwen_exfusion_checkpoint(path)
        ids = torch.randint(0, 96, (1, 6))
        assert torch.equal(model(ids, effort_mode="fixed_2"), loaded(ids, effort_mode="fixed_2"))
        assert loaded.parameter_provenance == provenance


def test_scalar_digest_and_qwen_counterfactual_probe_are_end_to_end():
    """Scalar ExFusion scales must not prevent canonical collection."""
    _, model = _model()
    assert _tensor_raw_bytes(torch.tensor(0.0)) == _tensor_raw_bytes(torch.tensor(0.0))
    assert full_state_dict_digest(model)
    task = {
        "task_id": "scalar-digest-regression",
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[1, 2, 3, 4, 5]]),
    }
    with CounterfactualCollector(model) as collector:
        record = collector.collect_one(task)
    assert len(record.probe_hidden) == model.hidden_size
    assert record.compute[0] < record.compute[1] < record.compute[2] < record.compute[3]
    assert record.compute_receipts is not None


def test_adaptive_qwen_dispatch_reuses_internal_probe_and_executes_selected_arm():
    _, model = _model()
    ids = torch.randint(0, 96, (1, 7))
    mask = torch.ones_like(ids)
    embedding = model.embed(ids)
    probe_h, _, decision = model.compute_effort_probe(embedding, mask)
    assert decision.source_position == "post_qwen_probe"
    assert not torch.equal(probe_h, embedding)

    adaptive_e0 = model(
        ids, attention_mask=mask, effort_mode="adaptive",
        effort_levels_override=torch.tensor([0]), return_compute_receipt=True,
    )
    fixed_e0 = model(ids, attention_mask=mask, effort_mode="fixed_0", return_compute_receipt=True)
    assert torch.equal(adaptive_e0["logits"], fixed_e0["logits"])
    assert adaptive_e0["effort_decision"]["levels"] == [0]
    assert adaptive_e0["compute_stats"]["executed_layer_count"] == fixed_e0["compute_stats"]["executed_layer_count"]

    adaptive_e3 = model(
        ids, attention_mask=mask, effort_mode="adaptive",
        effort_levels_override=torch.tensor([3]), return_compute_receipt=True,
    )
    fixed_e3 = model(ids, attention_mask=mask, effort_mode="fixed_3", return_compute_receipt=True)
    assert torch.equal(adaptive_e3["logits"], fixed_e3["logits"])
    assert adaptive_e3["effort_decision"]["levels"] == [3]
    assert adaptive_e3["compute_stats"]["latent_steps"] == fixed_e3["compute_stats"]["latent_steps"]


def test_adaptive_qwen_partitions_mixed_batch_by_selected_effort():
    _, model = _model()
    ids = torch.randint(0, 96, (2, 6))
    out = model(
        ids, effort_mode="adaptive", effort_levels_override=torch.tensor([0, 3]),
        return_compute_receipt=True,
    )
    assert out["effort_decision"]["levels"] == [0, 3]
    assert out["compute_stats"]["per_sample_compute"][0] < out["compute_stats"]["per_sample_compute"][1]


def test_explicit_stage_groups_and_gradient_remainder_resume():
    compat = QwenCompatModel(128, 16, 4, 4, 2, 32)
    model = augment_qwen_compat_model(
        compat, num_routed_experts=2, top_k=1,
        use_shallow_continuation=True, default_e3_steps=1,
    )
    stage = TrainingStageConfig(
        name="continuation", steps=4,
        train_parameter_groups=("continuation", "scales"),
        freeze_parameter_groups=("imported",),
        effort_sampling=(1.0, 0.0, 0.0, 0.0),
    )
    membership = apply_training_stage(model, stage)
    assert membership["trained"]
    assert all("_continuation." in n or n.endswith("_scale") for n in membership["trained"])
    with tempfile.TemporaryDirectory() as td:
        data = os.path.join(td, "train.jsonl")
        with open(data, "w", encoding="utf-8") as f:
            for i in range(8):
                f.write(json.dumps({"text": f"small training row {i}"}) + "\n")
        cfg = RealTrainConfig(
            steps=2, batch_size=1, seq_len=8, grad_accum=3,
            warmup_steps=1, log_every=99, eval_every=99,
            effort_mode="fixed_0", data_path=data,
            output_dir=os.path.join(td, "first"), stages=(stage,),
        )
        first = train_adapt(model, cfg)
        assert first["optimizer_steps_completed"] == 1
        assert first["next_micro_step"] == 2
        resumed_model = augment_qwen_compat_model(
            compat, num_routed_experts=2, top_k=1,
            use_shallow_continuation=True, default_e3_steps=1,
        )
        resumed = train_adapt(
            resumed_model,
            RealTrainConfig(
                **{**cfg.__dict__, "steps": 3, "resume": os.path.join(td, "first", "checkpoint_final.pt"),
                   "output_dir": os.path.join(td, "second")}
            ),
        )
        assert resumed["optimizer_steps_completed"] == 2
        assert resumed["next_micro_step"] == 3


def test_stable_gradient_clip_checks_values_not_reduction_overflow():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    norm = finite_and_clip_gradients([parameter], max_norm=1.0)
    assert norm == 5.0
    assert torch.allclose(parameter.grad, torch.tensor([0.6, 0.8]))
    parameter.grad = torch.tensor([float("nan"), 0.0])
    assert not torch.isfinite(torch.tensor(finite_and_clip_gradients([parameter], max_norm=1.0)))
