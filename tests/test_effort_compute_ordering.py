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
from daph.train_real import (
    RealTrainConfig, TrainingStageConfig, apply_training_stage,
    distillation_loss, train_adapt,
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
    assert calls["rec"] == 4 and calls["moe"] == 4 and calls["latent"] == 4


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
    assert torch.allclose(out, base + 1.0, atol=1e-6)


def test_training_init_is_explicit_and_enables_augmentation_gradients():
    _, model = _model()
    receipt = prepare_exfusion_for_training(model, gate0b_passed=True, epsilon=1e-3)
    assert receipt.backbone_unchanged and receipt.changed_scale_names
    ids = torch.randint(0, 96, (2, 7))
    model(ids, effort_mode="fixed_3").sum().backward()
    assert model.layers[0].latent_refine.fc2.weight.grad is not None
    assert model.layers[0].latent_refine.fc2.weight.grad.abs().sum() > 0


def test_parameter_provenance_is_exact_names():
    _, model = _model()
    provenance = model.parameter_provenance
    names = {n for n, _ in model.named_parameters()}
    assert provenance is not None
    assert set(provenance.imported_parameter_names).isdisjoint(provenance.new_parameter_names)
    assert set(provenance.imported_parameter_names) | set(provenance.new_parameter_names) == names
    assert all(name.endswith("_scale") for name in provenance.scale_parameter_names)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "canonical.pt")
        save_adapted_checkpoint(model, path)
        loaded = load_qwen_exfusion_checkpoint(path)
        ids = torch.randint(0, 96, (1, 6))
        assert torch.equal(model(ids, effort_mode="fixed_2"), loaded(ids, effort_mode="fixed_2"))
        assert loaded.parameter_provenance == provenance


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
