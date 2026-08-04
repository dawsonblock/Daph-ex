"""Research-build gates for middle-layer E3, profiling, mining and routing."""

import json

import pytest
import torch

from daph.counterfactual import CounterfactualCollector, _tensor_raw_bytes, full_state_dict_digest
from daph.e3_architecture import E3RefinementConfig, resolve_e3_region, sparse_profile_indices
from daph.e3_experiment import (
    active_refinement_layer,
    dose_response_variants,
    E2DifficultyBandConfig,
    location_ablation_variants,
    select_mixed_success_tasks,
    set_refinement_steps,
)
from daph.e3_metrics import E3QualificationConfig, e3_pair_metrics, qualify_e3_pairs
from daph.e3_training import E3StageConfig, configure_e3_training, e3_verified_objective
from daph.hard_case import E3HardCaseMiner, HardCaseMiningConfig
from daph.layer_contribution import LayerContributionConfig, LayerContributionProfiler, profile_selection_payload
from daph.policy_trainer import EffortPolicyArtifact, EffortPolicyTrainer, _state_dict_digest
from daph.pretrained import save_adapted_checkpoint
from daph.qwen_compat import QwenCompatModel
from daph.qwen_exfusion import augment_qwen_compat_model, gate0b_exact_parity, load_qwen_exfusion_checkpoint
from daph.verifiers import ExactMatchVerifier, make_quality_fn
from daph.train_real import TextBatcher, load_jsonl_training_records


def make_model(e3_config=None, layers=10):
    torch.manual_seed(31)
    compat = QwenCompatModel(80, 24, layers, 4, 2, 48)
    model = augment_qwen_compat_model(
        compat, num_routed_experts=2, top_k=1,
        e0_layer_count=max(2, layers // 2), e1_layer_count=max(3, layers * 3 // 4),
        e3_config=e3_config,
    )
    return compat, model


def test_tensor_hashing_all_shapes_and_empty():
    tensors = [
        torch.tensor(1.0), torch.tensor(1, dtype=torch.int64), torch.arange(3),
        torch.arange(6).reshape(2, 3), torch.empty(0),
    ]
    for tensor in tensors:
        assert _tensor_raw_bytes(tensor) == _tensor_raw_bytes(tensor.clone())


def test_middle_region_is_zero_based_40_60_and_deterministic():
    region = resolve_e3_region(E3RefinementConfig(), 24)
    assert region.selected_layers == tuple(range(9, 15))
    assert region.insertion_layer == 12
    assert sparse_profile_indices(28) == tuple(sorted(set(sparse_profile_indices(28))))
    profiled = resolve_e3_region(E3RefinementConfig(
        e3_refinement_mode="profiled_middle_recurrent", e3_region_selection="profiled",
        e3_profiled_layers=[7, 5, 6], source_profile_digest="abc123",
    ), 24)
    assert profiled.selected_layers == (5, 6, 7)
    assert profiled.source_profile_digest == "abc123"


def test_middle_refiner_location_zero_gate_and_receipt():
    compat, model = make_model()
    ids = torch.randint(0, 80, (1, 6))
    calls = []
    for index, layer in enumerate(model.layers):
        original = layer.latent_refine.forward
        layer.latent_refine.forward = lambda *a, _i=index, _f=original, **kw: (calls.append(_i) or _f(*a, **kw))
    e2 = model(ids, effort_mode="fixed_2")
    e3 = model(ids, effort_mode="fixed_3", return_compute_receipt=True)
    assert torch.equal(e2, e3["logits"])
    assert calls == [model.e3_region.insertion_layer]
    receipt = e3["compute_receipt"]
    assert receipt.middle_refinement_steps == model.e3_config.e3_refine_steps
    assert receipt.middle_refiner_calls == 1
    assert receipt.normalized_compute_cost > 1.0
    assert gate0b_exact_parity(compat, model, ids)["passed"]


def test_final_and_middle_variants_execute_different_locations():
    ids = torch.randint(0, 80, (1, 5))
    locations = []
    for mode in ("middle_recurrent", "final_refine"):
        _, model = make_model(E3RefinementConfig(e3_refinement_mode=mode, e3_refine_steps=1))
        calls = []
        for index, layer in enumerate(model.layers):
            original = layer.latent_refine.forward
            layer.latent_refine.forward = lambda *a, _i=index, _f=original, **kw: (calls.append(_i) or _f(*a, **kw))
        model(ids, effort_mode="fixed_3")
        locations.append(calls[0])
    assert locations[0] != locations[1]
    assert locations[1] == 9


def test_repeated_layer_shares_weights_is_zero_gated_and_counted():
    config = E3RefinementConfig(
        e3_refinement_mode="middle_repeat", e3_reuse_pretrained_layers=True,
        e3_reuse_layers=[4, 5], e3_repeat_count=2,
    )
    _, model = make_model(config)
    parameter_id = id(model.layers[4].base.mlp.down_proj.weight)
    ids = torch.randint(0, 80, (1, 5))
    e2 = model(ids, effort_mode="fixed_2")
    out = model(ids, effort_mode="fixed_3", return_compute_receipt=True)
    assert torch.equal(e2, out["logits"])
    assert id(model.layers[4].base.mlp.down_proj.weight) == parameter_id
    assert out["compute_receipt"].repeated_pretrained_layer_calls == 4
    assert out["compute_receipt"].attention_calls == len(model.layers) + 4


def test_probe_is_internal_continuable_and_adaptive_requires_verified_policy():
    _, model = make_model()
    ids = torch.randint(0, 80, (1, 6))
    embeddings = model.embed(ids)
    probe = model.compute_effort_probe(ids)
    assert probe.executed_layers <= model._layer_count(0)
    assert not torch.equal(probe.probe_hidden, embeddings)
    assert probe.compute_receipt.executed_layer_count == probe.executed_layers
    with pytest.raises(RuntimeError, match="VERIFIED_FIT"):
        model(ids, effort_mode="adaptive")


def test_installed_verified_controller_dispatches_physical_e3():
    _, model = make_model()
    with torch.no_grad():
        model.effort_controller.net[-1].weight.zero_()
        model.effort_controller.net[-1].bias.fill_(-20)
        model.effort_controller.net[-1].bias[3] = 20
    state = {key: value.detach().clone() for key, value in model.effort_controller.state_dict().items()}
    artifact = EffortPolicyArtifact(
        policy_version="test", base_model_digest="base", train_dataset_digest="train",
        validation_dataset_digest="val", split_manifest_digest="split", feature_dim=model.hidden_size,
        feature_spec="hidden", temperature=0.1, training_seed=1, training_config_digest="cfg",
        initial_state_dict_digest="init", metrics={}, state_dict_digest=_state_dict_digest(state),
        training_status="VERIFIED_FIT",
    )
    model.install_effort_policy(artifact, state, base_model_digest="base")
    out = model(torch.randint(0, 80, (1, 6)), effort_mode="adaptive", return_compute_receipt=True)
    assert out["chosen_effort"] == 3
    assert out["compute_receipt"].middle_refinement_steps > 0
    assert out["compute_stats"]["probe_compute_included"]


def test_verified_controller_survives_checkpoint_round_trip(tmp_path):
    _, model = make_model()
    with torch.no_grad():
        model.effort_controller.net[-1].weight.zero_()
        model.effort_controller.net[-1].bias.fill_(-20)
        model.effort_controller.net[-1].bias[3] = 20
    state = {key: value.detach().clone() for key, value in model.effort_controller.state_dict().items()}
    artifact = EffortPolicyArtifact(
        policy_version="test", base_model_digest="base", train_dataset_digest="train",
        validation_dataset_digest="val", split_manifest_digest="split", feature_dim=model.hidden_size,
        feature_spec="hidden", temperature=0.1, training_seed=1, training_config_digest="cfg",
        initial_state_dict_digest="init", metrics={}, state_dict_digest=_state_dict_digest(state),
        training_status="VERIFIED_FIT",
    )
    model.install_effort_policy(artifact, state, base_model_digest="base")
    path = tmp_path / "verified-policy.pt"
    save_adapted_checkpoint(model, str(path))
    loaded = load_qwen_exfusion_checkpoint(str(path))
    out = loaded(torch.randint(0, 80, (1, 6)), effort_mode="adaptive", return_compute_receipt=True)
    assert loaded.has_verified_effort_policy()
    assert out["chosen_effort"] == 3
    assert out["policy_logits"] is not None

    tampered_payload = torch.load(path, map_location="cpu", weights_only=False)
    controller_key = next(key for key in tampered_payload["state_dict"] if key.startswith("effort_controller."))
    tampered_payload["state_dict"][controller_key].view(-1)[0].add_(1.0)
    tampered_path = tmp_path / "tampered-policy.pt"
    torch.save(tampered_payload, tampered_path)
    with pytest.raises(ValueError, match="controller digest mismatch"):
        load_qwen_exfusion_checkpoint(str(tampered_path))

    with torch.no_grad():
        model.effort_controller.net[-1].bias.add_(1.0)
    with pytest.raises(RuntimeError, match="changed after verified policy installation"):
        save_adapted_checkpoint(model, str(tmp_path / "stale-policy.pt"))


def test_unverified_research_override_retains_policy_logits():
    _, model = make_model()
    out = model(
        torch.randint(0, 80, (1, 6)), effort_mode="adaptive",
        allow_unverified_policy=True, return_compute_receipt=True,
    )
    assert out["policy_logits"] is not None
    assert out["policy_logits"].shape == (1, 4)


def test_collector_uses_internal_probe_and_records_e3_metadata():
    _, model = make_model()
    calls = 0
    original = model.layers[0].forward
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    model.layers[0].forward = counted
    task = {"task_id": "one", "input_ids": [[1, 2, 3, 4]], "labels": [[1, 2, 3, 4]]}
    with CounterfactualCollector(model) as collector:
        record = collector.collect_one(task)
    assert record.probe_source == "internal_qwen_probe"
    assert record.compute_receipts[3]["e3_variant"] == "middle_recurrent"
    assert full_state_dict_digest(model)
    assert calls == 1  # common probe prefix is collected once and reused by all arms


class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)


class DummyProfileModel(torch.nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        self.layers = torch.nn.ModuleList(DummyLayer() for _ in range(depth))
        self.score = 0.0


def test_layer_profiler_contribution_negative_status_and_persistence(tmp_path):
    model = DummyProfileModel()
    scores = {0: -0.2, 1: 0.1, 2: 0.7, 3: 1.2, 4: 0.3, 5: 0.0}
    config = LayerContributionConfig(profile_mode="full", training_steps=2, best_contiguous_width=2)
    profiler = LayerContributionProfiler(model, config)

    def adapt(candidate, layer, objective, steps, seed):
        candidate.score = 1.0 if layer is None else scores[layer]
        if layer is not None:
            with torch.no_grad():
                candidate.layers[layer].base.weight.add_(0.01)

    report = profiler.run(lambda candidate: candidate.score, adapt, full_reference_adapter=adapt)
    assert report.profile_status == "FULL_PROFILE"
    assert report.results[0].layer_contribution < 0
    assert report.ranking[0] == 3
    payload = profile_selection_payload(report, strategy="best_contiguous", contiguous_width=2)
    assert payload["e3_profiled_layers"] == report.best_contiguous_region
    assert payload["source_profile_digest"] == report.digest()
    profiler.save(report, str(tmp_path))
    assert len((tmp_path / "per_layer_results.jsonl").read_text().splitlines()) == 6
    partial_model = DummyProfileModel(depth=12)
    partial = LayerContributionProfiler(partial_model, LayerContributionConfig(profile_mode="sparse"))
    def adapt_partial(candidate, layer, objective, steps, seed):
        candidate.score = 1.0 if layer is None else layer / 12
    partial_report = partial.run(lambda candidate: candidate.score, adapt_partial, score_full=1.0)
    assert partial_report.profile_status == "PARTIAL_PROFILE"


def test_layer_profiler_rejects_non_improving_full_reference():
    model = DummyProfileModel()
    original_flags = [parameter.requires_grad for parameter in model.parameters()]
    profiler = LayerContributionProfiler(model, LayerContributionConfig(profile_mode="full"))

    def adapt(candidate, layer, objective, steps, seed):
        candidate.score = -0.1 if layer is None else 0.01

    with pytest.raises(ValueError, match="did not improve"):
        profiler.run(lambda candidate: candidate.score, adapt, full_reference_adapter=adapt)
    assert [parameter.requires_grad for parameter in model.parameters()] == original_flags


def test_rescue_regression_metrics_and_statistical_gate():
    prototype_pairs = [
        {"task_id": "a", "e2_correct": False, "e3_correct": True, "quality_e2": 0.0, "quality_e3": 1.0, "compute_e2": 1.0, "compute_e3": 1.0, "task_family": "math", "template_id": "math-a", "difficulty_bucket": "hard"},
        {"task_id": "b", "e2_correct": False, "e3_correct": True, "quality_e2": 0.0, "quality_e3": 1.0, "compute_e2": 1.0, "compute_e3": 1.0, "task_family": "math", "template_id": "math-b", "difficulty_bucket": "hard"},
        {"task_id": "c", "e2_correct": False, "e3_correct": True, "quality_e2": 0.0, "quality_e3": 1.0, "compute_e2": 1.0, "compute_e3": 1.0, "task_family": "code", "template_id": "code-a", "difficulty_bucket": "easy"},
    ]
    pairs = [
        {**prototype_pairs[index % len(prototype_pairs)], "task_id": f"{prototype_pairs[index % len(prototype_pairs)]['task_id']}-{index}", "template_id": f"template-{index}"}
        for index in range(24)
    ]
    metrics = e3_pair_metrics(pairs)
    assert metrics["rescue_count"] == 24 and metrics["regression_count"] == 0
    qualified = qualify_e3_pairs(pairs, E3QualificationConfig(bootstrap_samples=100, seed=1))
    assert qualified["qualified"] and qualified["quality_lcb95"] > 0


def test_hard_case_miner_labels_and_configurable_curriculum():
    _, model = make_model(layers=4)
    tasks = [
        {"task_id": "bad", "input_ids": [1, 2, 3], "expected": False},
        {"task_id": "good", "input_ids": [2, 3, 4], "expected": True},
    ]
    miner = E3HardCaseMiner(
        model, lambda _out, task: {"correct": task["expected"], "reward": float(task["expected"])},
        HardCaseMiningConfig(hard_failure_ratio=0.5, hard_uncertain_ratio=0.0, easy_correct_ratio=0.5),
    )
    records = miner.mine(tasks)
    assert [record.category for record in records] == ["HARD_FAILURE", "EASY_CORRECT"]
    assert len(miner.sample(records, 4)) == 4


def test_hard_case_miner_uses_model_device():
    _, model = make_model(layers=4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    observed = []

    def verify(output, task):
        observed.append(output["logits"].device)
        return {"correct": True}

    miner = E3HardCaseMiner(model, verify, HardCaseMiningConfig())
    miner.mine([{"task_id": "device", "input_ids": [1, 2, 3]}])
    assert len(observed) == 1
    assert observed[0].type == device.type


def test_hard_case_miner_supports_project_verifier_contract_and_generation():
    _, model = make_model(layers=4)

    class FixedTokenizer:
        def batch_decode(self, sequences, skip_special_tokens=True):
            return ["42"] * sequences.size(0)

    miner = E3HardCaseMiner(
        model,
        make_quality_fn(ExactMatchVerifier()),
        HardCaseMiningConfig(max_new_tokens=1),
        tokenizer=FixedTokenizer(),
    )
    records = miner.mine([{"task_id": "verified", "input_ids": [1, 2, 3], "expected": "42"}])
    assert records[0].e2_correct
    assert records[0].e2_verifier_reward == 1.0
    assert records[0].e2_answer == "42"

    without_tokenizer = E3HardCaseMiner(
        model, make_quality_fn(ExactMatchVerifier()), HardCaseMiningConfig(max_new_tokens=1),
    )
    with pytest.raises(ValueError, match="requires a verifiable E2 result"):
        without_tokenizer.mine([{"task_id": "unverifiable", "input_ids": [1, 2, 3], "expected": "42"}])


def test_policy_fit_is_blocked_until_arm_and_oracle_gates_pass():
    controller = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Softmax(dim=-1))
    trainer = EffortPolicyTrainer(controller)
    with pytest.raises(RuntimeError, match="Policy training blocked"):
        trainer.fit([])
    with pytest.raises(RuntimeError, match="effort arms"):
        trainer.authorize_policy_training({"qualified": False}, {"has_routing_opportunity": True})
    trainer.authorize_policy_training({"qualified": True}, {"has_routing_opportunity": True})


def test_dose_and_location_ablation_contracts_are_explicit():
    assert [item.refinement_steps for item in dose_response_variants()] == [0, 1, 2, 4, 8]
    locations = location_ablation_variants()
    assert [item.name for item in locations] == ["EARLY", "MIDDLE", "LATE", "FINAL"]
    assert all(not item.strong_e2_distillation for item in locations)


def test_real_ablation_harness_targets_active_location_and_step_count():
    _, middle = make_model(E3RefinementConfig(e3_refinement_mode="middle_recurrent", e3_refine_steps=1))
    _, final = make_model(E3RefinementConfig(e3_refinement_mode="final_refine", e3_refine_steps=1))
    assert active_refinement_layer(middle) == middle.e3_region.insertion_layer
    assert active_refinement_layer(final) == len(final.layers) - 1
    assert active_refinement_layer(middle) != active_refinement_layer(final)
    set_refinement_steps(middle, 4)
    out = middle(torch.randint(0, 80, (1, 5)), effort_mode="fixed_3", return_compute_receipt=True)
    assert middle.e3_config.e3_refine_steps == 4
    assert out["compute_receipt"].middle_refinement_steps == 4


def test_per_example_research_step_override_records_actual_steps():
    _, middle = make_model(E3RefinementConfig(e3_refinement_mode="middle_recurrent", e3_refine_steps=1))
    ids = torch.randint(0, 80, (1, 5))
    out = middle(
        ids, effort_mode="fixed_3", e3_refinement_steps_override=4,
        return_compute_receipt=True,
    )
    assert out["compute_receipt"].middle_refinement_steps == 4
    assert out["compute_stats"]["research_step_override"]
    with pytest.raises(ValueError, match="batch_size=1"):
        middle(ids.repeat(2, 1), effort_mode="fixed_3", e3_refinement_steps_override=2)


def test_answer_only_batcher_masks_every_prompt_and_padding_token(tmp_path):
    path = tmp_path / "answer-only.jsonl"
    path.write_text(json.dumps({"prompt": "abc", "answer": "de"}) + "\n")
    records = load_jsonl_training_records(str(path), answer_only_loss=True)

    class TinyTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}[char] for char in text]}

    batcher = TextBatcher(
        records, tokenizer=TinyTokenizer(), seq_len=8, batch_size=1,
        device=torch.device("cpu"), answer_only_loss=True,
    )
    ids, labels, mask = next(batcher)
    assert ids.tolist() == [[1, 2, 3, 4, 5, 0, 0, 0]]
    assert labels.tolist() == [[-100, -100, -100, 4, 5, -100, -100, -100]]
    assert mask.tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]]


def test_mixed_success_calibration_is_deterministic_and_non_degenerate():
    tasks = [{"task_id": f"t{i}"} for i in range(10)]
    outcomes = [{"task_id": f"t{i}", "e2_correct": i < 5} for i in range(10)]
    config = E2DifficultyBandConfig(target_size=6, seed=7)
    selected, report = select_mixed_success_tasks(tasks, outcomes, config)
    repeated, repeated_report = select_mixed_success_tasks(tasks, outcomes, config)
    assert selected == repeated and report == repeated_report
    assert report["selected_successes"] == 3
    assert report["selected_failures"] == 3
    assert report["selected_e2_accuracy"] == 0.5


def test_e3_stage_a_freezes_e2_and_task_loss_is_primary():
    _, model = make_model()
    receipt, groups = configure_e3_training(model, E3StageConfig(regression_guard_weight=0.01))
    assert receipt.changed_scale_names
    assert not groups["middle_layers"]
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if name in model.parameter_provenance.imported_parameter_names)
    ids = torch.randint(0, 80, (1, 5))
    e3 = model(ids, effort_mode="fixed_3", return_compute_receipt=True)
    with torch.no_grad():
        e2_logits = model(ids, effort_mode="fixed_2")
    task_loss = lambda output, task: output["logits"].square().mean()
    total, pieces = e3_verified_objective(
        e3, {"logits": e2_logits}, {}, task_loss, regression_guard_weight=0.01,
    )
    total.backward()
    assert pieces["regression_guard_weight"] == 0.01
    assert model.layers[model.e3_region.insertion_layer].latent_refine.fc2.weight.grad is not None
