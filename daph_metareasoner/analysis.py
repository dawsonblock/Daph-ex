"""Oracle, fixed-policy, confidence, sham, and uncertainty-aware evaluation."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from .schema import Action, ExperienceRecord


def group_by_state(records: Iterable[ExperienceRecord]) -> Dict[str, Dict[str, ExperienceRecord]]:
    grouped: Dict[str, Dict[str, ExperienceRecord]] = defaultdict(dict)
    for record in records:
        if record.action in grouped[record.state.state_id]:
            raise ValueError(f"Duplicate action {record.action} for state {record.state.state_id}")
        grouped[record.state.state_id][record.action] = record
    for state_id, rows in grouped.items():
        missing = {action.value for action in Action} - set(rows)
        if missing:
            raise ValueError(f"State {state_id} is missing actions: {sorted(missing)}")
    return dict(grouped)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def bootstrap_lcb(
    paired_deltas: Sequence[float], *, confidence: float = 0.95,
    samples: int = 2000, seed: int = 42,
) -> float:
    if not paired_deltas:
        raise ValueError("Cannot bootstrap an empty sample")
    rng = random.Random(seed)
    n = len(paired_deltas)
    means = sorted(
        _mean([paired_deltas[rng.randrange(n)] for _ in range(n)])
        for _ in range(max(1, samples))
    )
    index = min(len(means) - 1, int((1.0 - confidence) * len(means)))
    return means[max(0, index)]


@dataclass(frozen=True)
class OracleGateConfig:
    min_oracle_gain_over_fixed: float = 0.02
    confidence: float = 0.95
    bootstrap_samples: int = 2000
    seed: int = 42


def oracle_value_study(
    records: Sequence[ExperienceRecord],
    config: OracleGateConfig = OracleGateConfig(),
) -> Dict[str, Any]:
    grouped = group_by_state(records)
    state_rows = list(grouped.values())
    fixed = {
        action.value: _mean([rows[action.value].delta_utility for rows in state_rows])
        for action in Action
    }
    best_fixed_action = max(fixed, key=fixed.get)
    best_fixed = fixed[best_fixed_action]
    oracle_values = [max(row.delta_utility for row in rows.values()) for rows in state_rows]
    oracle_actions = [max(rows, key=lambda name: rows[name].delta_utility) for rows in state_rows]
    oracle = _mean(oracle_values)
    paired = [value - rows[best_fixed_action].delta_utility for value, rows in zip(oracle_values, state_rows)]
    lcb = bootstrap_lcb(
        paired, confidence=config.confidence,
        samples=config.bootstrap_samples, seed=config.seed,
    )
    gain = oracle - best_fixed
    qualified = gain > config.min_oracle_gain_over_fixed and lcb > 0.0
    return {
        "states": len(state_rows),
        "fixed_action_utility": fixed,
        "best_fixed_action": best_fixed_action,
        "best_fixed_utility": best_fixed,
        "immediate_stop_utility": fixed[Action.STOP.value],
        "oracle_utility": oracle,
        "oracle_gain_over_fixed": gain,
        "oracle_gain_lcb": lcb,
        "oracle_action_frequency": dict(Counter(oracle_actions)),
        "conditional_value_exists": qualified,
        "controller_training_allowed": qualified,
        "threshold": config.min_oracle_gain_over_fixed,
    }


Policy = Callable[[Mapping[str, ExperienceRecord]], str]


def evaluate_offline_policy(
    records: Sequence[ExperienceRecord], policy: Policy,
) -> Dict[str, Any]:
    grouped = group_by_state(records)
    chosen = []
    oracle = []
    stop_optimal = []
    harmful = waste = 0
    for rows in grouped.values():
        action = policy(rows)
        if action not in rows:
            raise ValueError(f"Policy selected unavailable action {action}")
        record = rows[action]
        chosen.append(record.delta_utility)
        oracle.append(max(row.delta_utility for row in rows.values()))
        stop_is_optimal = rows[Action.STOP.value].delta_utility >= max(row.delta_utility for row in rows.values())
        stop_optimal.append((action == Action.STOP.value, stop_is_optimal))
        harmful += int(record.delta_quality < 0.0)
        waste += int(record.delta_quality == 0.0 and record.action_cost > 0.0)
    stops = sum(pred for pred, _ in stop_optimal)
    truly_stop = sum(actual for _, actual in stop_optimal)
    correct_stops = sum(pred and actual for pred, actual in stop_optimal)
    return {
        "states": len(chosen),
        "mean_utility": _mean(chosen),
        "mean_regret": _mean([top - got for top, got in zip(oracle, chosen)]),
        "stop_precision": correct_stops / max(stops, 1),
        "stop_recall": correct_stops / max(truly_stop, 1),
        "harmful_continuation_rate": harmful / max(len(chosen), 1),
        "waste_rate": waste / max(len(chosen), 1),
    }


def predictor_policy(controller: Any) -> Policy:
    """Adapt an on-path controller to offline scoring without oracle visibility."""
    def policy(rows: Mapping[str, ExperienceRecord]) -> str:
        state = next(iter(rows.values())).state
        return str(controller.decide(state).action)
    return policy


def paired_policy_gate(
    records: Sequence[ExperienceRecord],
    learned: Policy,
    control: Policy,
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    grouped = group_by_state(records)
    deltas = []
    learned_actions: Counter[str] = Counter()
    control_actions: Counter[str] = Counter()
    for rows in grouped.values():
        learned_action = learned(rows)
        control_action = control(rows)
        learned_actions[learned_action] += 1
        control_actions[control_action] += 1
        deltas.append(rows[learned_action].delta_utility - rows[control_action].delta_utility)
    mean_delta = _mean(deltas)
    lcb = bootstrap_lcb(
        deltas, confidence=confidence, samples=bootstrap_samples, seed=seed,
    )
    return {
        "states": len(deltas),
        "mean_utility_delta": mean_delta,
        "utility_delta_lcb": lcb,
        "confidence": confidence,
        "learned_action_frequency": dict(learned_actions),
        "control_action_frequency": dict(control_actions),
        "qualified": mean_delta > 0.0 and lcb > 0.0,
    }


def probe_signal_gate(
    hidden_metrics: Mapping[str, float | None],
    cheap_metrics: Mapping[str, float | None],
    *,
    minimum_hidden_auroc: float = 0.65,
    minimum_margin: float = 0.03,
) -> Dict[str, Any]:
    hidden_raw = hidden_metrics["auroc"]
    cheap_raw = cheap_metrics["auroc"]
    hidden = None if hidden_raw is None else float(hidden_raw)
    cheap = None if cheap_raw is None else float(cheap_raw)
    margin = None if hidden is None or cheap is None else hidden - cheap
    qualified = (
        hidden is not None and cheap is not None and margin is not None
        and math.isfinite(hidden) and math.isfinite(cheap)
        and hidden > minimum_hidden_auroc and margin > minimum_margin
    )
    return {
        "hidden_auroc": hidden,
        "cheap_proxy_auroc": cheap,
        "hidden_margin": margin,
        "minimum_hidden_auroc": minimum_hidden_auroc,
        "minimum_margin": minimum_margin,
        "qualified": qualified,
        "value_controller_training_allowed": qualified,
    }


def confidence_threshold_policy(threshold: float, continuation: str) -> Policy:
    def policy(rows: Mapping[str, ExperienceRecord]) -> str:
        state = next(iter(rows.values())).state
        return Action.STOP.value if state.answer_confidence >= threshold else continuation
    return policy


def entropy_threshold_policy(threshold: float, continuation: str) -> Policy:
    def policy(rows: Mapping[str, ExperienceRecord]) -> str:
        state = next(iter(rows.values())).state
        return continuation if state.answer_entropy >= threshold else Action.STOP.value
    return policy


def fixed_action_policy(action: str) -> Policy:
    if action not in {candidate.value for candidate in Action}:
        raise ValueError(f"Unknown fixed action: {action}")

    def policy(_: Mapping[str, ExperienceRecord]) -> str:
        return action
    return policy


def prompt_length_policy(threshold: int, continuation: str) -> Policy:
    def policy(rows: Mapping[str, ExperienceRecord]) -> str:
        state = next(iter(rows.values())).state
        return continuation if len(state.prompt) >= threshold else Action.STOP.value
    return policy


def answer_stability_policy(repeats: int, continuation: str) -> Policy:
    def policy(rows: Mapping[str, ExperienceRecord]) -> str:
        state = next(iter(rows.values())).state
        return Action.STOP.value if state.repeated_answer_count >= repeats else continuation
    return policy


def frequency_matched_random_policy(
    frequencies: Mapping[str, float], *, seed: int = 42,
) -> Policy:
    rng = random.Random(seed)
    actions = list(frequencies)
    weights = [float(frequencies[action]) for action in actions]

    def policy(_: Mapping[str, ExperienceRecord]) -> str:
        return rng.choices(actions, weights=weights, k=1)[0]
    return policy


def fit_family_lookup_policy(
    train_records: Sequence[ExperienceRecord], *, fallback_action: str = Action.STOP.value,
) -> Policy:
    grouped = group_by_state(train_records)
    values: Dict[str, Dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rows in grouped.values():
        family = next(iter(rows.values())).task.family_id
        for action, record in rows.items():
            values[family][action].append(record.delta_utility)
    lookup = {
        family: max(action_values, key=lambda action: _mean(action_values[action]))
        for family, action_values in values.items()
    }

    def policy(rows: Mapping[str, ExperienceRecord]) -> str:
        family = next(iter(rows.values())).task.family_id
        return lookup.get(family, fallback_action)
    return policy


def oracle_capture(policy_utility: float, fixed_utility: float, oracle_utility: float) -> float:
    gap = oracle_utility - fixed_utility
    return 0.0 if math.isclose(gap, 0.0) else (policy_utility - fixed_utility) / gap
