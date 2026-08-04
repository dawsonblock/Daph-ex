import json
import gzip
import importlib.util
from dataclasses import replace

import pytest
import torch

from daph.verifiers import NumericVerifier
from daph_metareasoner import (
    Action,
    ActionReceipt,
    ActionValueEnsemble,
    BranchResult,
    CollectionConfig,
    ConservativeVOCPolicy,
    CounterfactualExperienceCollector,
    FixedRuntimePolicy,
    LoopGuard,
    OnPathExecutor,
    OracleGateConfig,
    PolicyConfig,
    ReasoningState,
    RuntimeLimits,
    StateVectorizer,
    ThresholdRuntimePolicy,
    Task,
    UtilityConfig,
    ValueTrainingConfig,
    build_split_manifest,
    load_records,
    oracle_value_study,
    paired_policy_gate,
    ProbeTrainingConfig,
    probe_signal_gate,
    predictor_policy,
    records_digest,
    train_value_ensemble,
    train_probe,
)


def make_state(task_id, answer, *, step=0, signal=0.0, confidence=0.5, history=()):
    hidden = {
        "25": (signal, 0.1),
        "50": (signal, 0.2),
        "75": (signal, 0.3),
        "100": (signal, 0.4),
    }
    return ReasoningState.create(
        task_id=task_id,
        step=step,
        answer=str(answer),
        prompt=f"task {task_id}",
        action_history=tuple(history),
        hidden_by_depth=hidden,
        hidden_final_token=hidden["100"],
        answer_confidence=confidence,
        budget_remaining=1.0,
    )


class MockAdapter:
    model_digest = "model-digest"
    environment_digest = "environment-digest"

    def __init__(self):
        self.calls = []

    def initial_state(self, task, *, budget):
        signal = float(task.metadata.get("signal", 0.0))
        state = make_state(
            task.task_id, task.metadata.get("initial", "0"),
            signal=signal, confidence=float(task.metadata.get("confidence", 0.5)),
        )
        return replace(state, budget_remaining=budget)

    def execute(self, task, state, action):
        self.calls.append((state.state_id, action.value))
        receipt = ActionReceipt(
            action=action.value,
            latency_ms=0.0 if action is Action.STOP else 10.0,
            input_tokens=0 if action is Action.STOP else 5,
            output_tokens=0 if action is Action.STOP else 1,
            normalized_compute=0.0 if action is Action.STOP else 0.1,
            model_digest=self.model_digest,
            environment_digest=self.environment_digest,
        )
        if action is Action.STOP:
            next_state = state
        else:
            answer = str(task.metadata.get("outcomes", {}).get(action.value, state.answer))
            signal = float(task.metadata.get("next_signal", -1.0))
            next_state = make_state(
                task.task_id, answer, step=state.step + 1, signal=signal,
                confidence=0.9, history=state.action_history + (action.value,),
            )
            next_state = replace(
                next_state,
                answer_changed=answer != state.answer,
                repeated_answer_count=0 if answer != state.answer else state.repeated_answer_count + 1,
                hidden_cosine_previous=0.999 if answer == state.answer else 0.5,
                budget_remaining=max(0.0, state.budget_remaining - 0.05),
            )
        return BranchResult(action, state.state_id, next_state, receipt)


def task(index, initial, expected, outcomes, *, signal=0.0, split="experience", family="math"):
    return Task(
        task_id=f"t{index}", prompt=f"What is task {index}?", expected=str(expected),
        family_id=family, split=split, template_id=f"template-{index}",
        generator_seed=str(index),
        metadata={"initial": str(initial), "outcomes": outcomes, "signal": signal},
    )


def collect(tasks, *, max_depth=0):
    adapter = MockAdapter()
    collector = CounterfactualExperienceCollector(
        adapter, NumericVerifier(), UtilityConfig(),
        CollectionConfig(max_depth=max_depth, max_states_per_task=16),
    )
    return adapter, collector.collect_many(tasks)


def test_counterfactual_branches_are_isolated_and_receipts_are_immutable(tmp_path):
    adapter, records = collect([
        task(1, 0, 2, {"THINK": 1, "VERIFY": 2, "DECOMPOSE": 3}),
    ])
    state_ids = {record.state.state_id for record in records}
    assert len(records) == 4 and len(state_ids) == 1
    assert [state_id for state_id, _ in adapter.calls] == [records[0].state.state_id] * 4
    assert {record.answer_after for record in records} == {"0", "1", "2", "3"}
    receipt = CounterfactualExperienceCollector.save(records, tmp_path / "experience.jsonl")
    restored = load_records(tmp_path / "experience.jsonl")
    assert records_digest(restored) == receipt["records_digest"]
    compressed = tmp_path / "experience.jsonl.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        handle.write((tmp_path / "experience.jsonl").read_text())
    assert records_digest(load_records(compressed)) == receipt["records_digest"]


def test_counterfactual_collector_refuses_silently_truncated_state_trees():
    adapter = MockAdapter()
    collector = CounterfactualExperienceCollector(
        adapter, NumericVerifier(), UtilityConfig(),
        CollectionConfig(max_depth=2, max_states_per_task=2),
    )
    with pytest.raises(RuntimeError, match="refusing a partial counterfactual table"):
        collector.collect_task(task(1, 0, 1, {"THINK": 1, "VERIFY": 1, "DECOMPOSE": 1}))


def test_utility_cost_is_subtracted_exactly_once():
    _, records = collect([
        task(1, 0, 1, {"THINK": 1, "VERIFY": 0, "DECOMPOSE": 0}),
    ])
    by_action = {record.action: record for record in records}
    think = by_action["THINK"]
    assert think.delta_quality == 1.0
    assert think.action_cost == 0.02
    assert think.delta_utility == pytest.approx(0.98)
    verify = by_action["VERIFY"]
    assert verify.delta_quality == 0.0
    assert verify.delta_utility == pytest.approx(-0.04)


def test_oracle_gate_detects_conditional_value_before_controller_training():
    tasks = [
        task(1, 0, 1, {"THINK": 1, "VERIFY": 0, "DECOMPOSE": 0}),
        task(2, 0, 1, {"THINK": 0, "VERIFY": 1, "DECOMPOSE": 0}),
        task(3, 1, 1, {"THINK": 0, "VERIFY": 1, "DECOMPOSE": 1}),
        task(4, 0, 1, {"THINK": 0, "VERIFY": 0, "DECOMPOSE": 1}),
    ]
    _, records = collect(tasks)
    report = oracle_value_study(records, OracleGateConfig(
        min_oracle_gain_over_fixed=0.01, bootstrap_samples=500, seed=7,
    ))
    assert report["oracle_gain_over_fixed"] > 0.1
    assert report["oracle_gain_lcb"] > 0.0
    assert report["controller_training_allowed"] is True
    assert set(report["oracle_action_frequency"]) >= {"STOP", "THINK", "VERIFY", "DECOMPOSE"}


class StaticPredictor:
    def __init__(self, means, uncertainty=None):
        self.means = means
        self.uncertainty = uncertainty or {action.value: 0.0 for action in Action}

    def predict(self, state):
        return dict(self.means), dict(self.uncertainty)


def test_conservative_policy_stops_when_lcb_does_not_clear_cost():
    means = {"STOP": 0.0, "THINK": 0.03, "VERIFY": 0.10, "DECOMPOSE": 0.04}
    uncertainty = {"STOP": 0.0, "THINK": 0.02, "VERIFY": 0.08, "DECOMPOSE": 0.02}
    policy = ConservativeVOCPolicy(
        StaticPredictor(means, uncertainty),
        config=PolicyConfig(uncertainty_beta=1.0),
    )
    decision = policy.decide(make_state("x", 0))
    assert decision.action == "STOP"
    assert decision.stop_reason == "STOP_NON_POSITIVE_VOC"


class StepPredictor:
    def predict(self, state):
        if state.step == 0:
            means = {"STOP": 0.0, "THINK": 0.0, "VERIFY": 1.0, "DECOMPOSE": 0.0}
        else:
            means = {action.value: 0.0 for action in Action}
        return means, {action.value: 0.0 for action in Action}


def test_on_path_executor_runs_only_the_selected_branch():
    adapter = MockAdapter()
    policy = ConservativeVOCPolicy(StepPredictor())
    executor = OnPathExecutor(adapter, policy, RuntimeLimits(max_steps=3, max_cost=0.2))
    result = executor.run(task(1, 0, 1, {"THINK": 0, "VERIFY": 1, "DECOMPOSE": 0}))
    assert [action for _, action in adapter.calls] == ["VERIFY"]
    assert result.answer == "1"
    assert result.total_steps == 1


def test_fixed_depth_and_threshold_runtime_controls_are_real_policies():
    adapter = MockAdapter()
    fixed = OnPathExecutor(
        adapter, FixedRuntimePolicy("THINK", max_actions=2),
        RuntimeLimits(max_steps=4, max_cost=0.2),
    )
    fixed.run(task(1, 0, 1, {"THINK": 2, "VERIFY": 1, "DECOMPOSE": 0}))
    assert [action for _, action in adapter.calls] == ["THINK", "THINK"]
    adapter.calls.clear()
    threshold = OnPathExecutor(
        adapter,
        ThresholdRuntimePolicy("VERIFY", feature="confidence", threshold=0.75),
        RuntimeLimits(max_steps=4, max_cost=0.2),
    )
    threshold.run(task(2, 0, 1, {"VERIFY": 1}, signal=0.0))
    assert [action for _, action in adapter.calls] == ["VERIFY"]


def test_loop_guard_blocks_repetition_recurrence_and_abab_cycles():
    guard = LoopGuard(RuntimeLimits(action_repeat_limit=2, unchanged_answer_limit=2))
    state = replace(
        make_state("x", 1), answer_changed=False,
        repeated_answer_count=2, hidden_cosine_previous=0.9999,
    )
    assert guard.blocked_actions([state], ["VERIFY", "VERIFY"]) == {
        "THINK", "VERIFY", "DECOMPOSE",
    }
    assert guard.blocked_actions([], ["VERIFY", "THINK", "VERIFY", "THINK"]) == {
        "THINK", "VERIFY", "DECOMPOSE",
    }


def test_split_manifest_rejects_exact_prompt_and_ood_family_leakage():
    base = task(1, 0, 1, {}, split="experience", family="math")
    duplicate = replace(base, task_id="duplicate", split="test")
    with pytest.raises(ValueError, match="Exact prompt leakage"):
        build_split_manifest([base, duplicate])
    ood = replace(
        task(2, 0, 1, {}, split="ood", family="math"), prompt="different",
    )
    with pytest.raises(ValueError, match="OOD family leakage"):
        build_split_manifest([base, ood])
    reused_template = replace(
        task(3, 0, 1, {}, split="test", family="math"),
        prompt="new wording", template_id=base.template_id, generator_seed="new-seed",
    )
    with pytest.raises(ValueError, match="Template leakage"):
        build_split_manifest([base, reused_template])


def _synthetic_training_records(count=24, offset=0, split="experience"):
    tasks = []
    for local_index in range(count):
        index = offset + local_index
        helps = local_index % 2 == 0
        tasks.append(task(
            index, 0, 1,
            {"THINK": 1 if helps else 0, "VERIFY": 0, "DECOMPOSE": 0},
            signal=1.0 if helps else -1.0,
            split=split,
        ))
    return collect(tasks)[1]


def test_hidden_probe_signal_gate_beats_cheap_proxy_on_independent_states():
    train = _synthetic_training_records(40)
    validation = _synthetic_training_records(20, offset=100, split="validation")
    config = ProbeTrainingConfig(epochs=80, lr=0.03, seed=3)
    _, _, hidden = train_probe(train, validation, StateVectorizer("hidden_runtime"), config)
    _, _, cheap = train_probe(train, validation, StateVectorizer("cheap"), config)
    gate = probe_signal_gate(hidden, cheap)
    assert hidden["auroc"] > 0.95
    assert hidden["auroc"] > cheap["auroc"] + 0.03
    assert gate["value_controller_training_allowed"] is True


def test_single_class_probe_metrics_are_valid_json_nulls():
    train = _synthetic_training_records(20)
    validation_tasks = [
        task(
            200 + index, 1, 1,
            {"THINK": 1, "VERIFY": 1, "DECOMPOSE": 1},
            signal=-1.0, split="validation",
        )
        for index in range(6)
    ]
    validation = collect(validation_tasks)[1]
    _, _, metrics = train_probe(
        train, validation, StateVectorizer("hidden_runtime"),
        ProbeTrainingConfig(epochs=5, seed=2),
    )
    assert metrics["auroc"] is None and metrics["auprc"] is None
    assert "NaN" not in json.dumps(metrics)
    assert probe_signal_gate(metrics, metrics)["qualified"] is False


def test_value_ensemble_round_trip_and_tamper_detection(tmp_path):
    records = _synthetic_training_records()
    config = ValueTrainingConfig(
        epochs=8, hidden_dim=32, second_hidden_dim=16,
        batch_size=16, ensemble_size=2, seed=4,
    )
    ensemble = train_value_ensemble(records, StateVectorizer("hidden_runtime"), config)
    target = tmp_path / "controller.pt"
    ensemble.save(
        target, training_digest=records_digest(records), config=config,
        base_model_digest="model-digest",
    )
    restored, artifact = ActionValueEnsemble.load(target)
    state = records[0].state
    assert restored.predict(state)[0].keys() == ensemble.predict(state)[0].keys()
    _, _, correctness, correctness_uncertainty = restored.predict_state(state)
    assert 0.0 <= correctness <= 1.0
    assert correctness_uncertainty >= 0.0
    assert artifact["training_status"] == "UNVERIFIED_FIT"
    assert artifact["base_model_digest"] == "model-digest"

    payload = torch.load(target, weights_only=False)
    first = next(iter(payload["state_dicts"][0]))
    payload["state_dicts"][0][first].view(-1)[0] += 1.0
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="state_dict digest mismatch"):
        ActionValueEnsemble.load(tampered)


def test_paired_policy_gate_requires_strictly_positive_lcb():
    _, records = collect([
        task(1, 0, 1, {"THINK": 1, "VERIFY": 0, "DECOMPOSE": 0}),
        task(2, 0, 1, {"THINK": 0, "VERIFY": 0, "DECOMPOSE": 0}),
    ])
    learned = lambda rows: "THINK"
    control = lambda rows: "STOP"
    report = paired_policy_gate(records, learned, control, bootstrap_samples=500, seed=1)
    assert report["mean_utility_delta"] > 0.0
    assert report["utility_delta_lcb"] < 0.0
    assert report["qualified"] is False


def test_cli_modules_import_without_optional_transformers_dependency():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    for index, name in enumerate((
        "collect_voc_experience.py",
        "run_voc_on_path.py",
        "run_voc_policy_suite.py",
    )):
        spec = importlib.util.spec_from_file_location(f"voc_cli_{index}", root / "scripts" / name)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
