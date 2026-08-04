"""Conservative value-of-computation policy with explicit cost and uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Protocol

from .schema import ALL_ACTIONS, NON_STOP_ACTIONS, Action, ReasoningState, StopReason
from .utility import UtilityConfig


class ValuePredictor(Protocol):
    def predict(self, state: ReasoningState) -> tuple[Dict[str, float], Dict[str, float]]: ...


@dataclass(frozen=True)
class PolicyConfig:
    uncertainty_beta: float = 1.0
    minimum_net_value: float = 0.0
    success_confidence: float = 0.90
    failure_confidence: float = 0.25


@dataclass(frozen=True)
class ControllerDecision:
    action: str
    mean_quality_gain: Mapping[str, float]
    uncertainty: Mapping[str, float]
    conservative_net_value: Mapping[str, float]
    stop_reason: str = ""
    current_correctness_estimate: float | None = None
    correctness_uncertainty: float | None = None


class ConservativeVOCPolicy:
    def __init__(
        self,
        predictor: ValuePredictor,
        utility: UtilityConfig = UtilityConfig(),
        config: PolicyConfig = PolicyConfig(),
    ) -> None:
        self.predictor = predictor
        self.utility = utility
        self.config = config

    def decide(
        self,
        state: ReasoningState,
        *,
        blocked_actions: Iterable[str] = (),
    ) -> ControllerDecision:
        correctness_estimate = correctness_uncertainty = None
        if hasattr(self.predictor, "predict_state"):
            means, uncertainty, correctness_estimate, correctness_uncertainty = self.predictor.predict_state(state)
        else:
            means, uncertainty = self.predictor.predict(state)
        blocked = set(blocked_actions)
        net: Dict[str, float] = {Action.STOP.value: 0.0}
        for action in NON_STOP_ACTIONS:
            name = action.value
            if name in blocked or self.utility.action_cost(action) > state.budget_remaining:
                net[name] = float("-inf")
            else:
                net[name] = (
                    float(means[name])
                    - self.utility.action_cost(action)
                    - self.config.uncertainty_beta * float(uncertainty[name])
                )
        best = max((action.value for action in ALL_ACTIONS), key=lambda name: net.get(name, float("-inf")))
        if best == Action.STOP.value or net[best] <= self.config.minimum_net_value:
            if all(name in blocked for name in (action.value for action in NON_STOP_ACTIONS)):
                reason = StopReason.LOOP_GUARD.value
            elif state.budget_remaining <= min(self.utility.action_cost(action) for action in NON_STOP_ACTIONS):
                reason = StopReason.BUDGET.value
            else:
                stop_confidence = (
                    state.answer_confidence
                    if correctness_estimate is None else correctness_estimate
                )
                if stop_confidence >= self.config.success_confidence:
                    reason = StopReason.SUCCESS.value
                elif stop_confidence <= self.config.failure_confidence:
                    reason = StopReason.FAILURE.value
                else:
                    reason = StopReason.NON_POSITIVE_VOC.value
            best = Action.STOP.value
        else:
            reason = ""
        return ControllerDecision(
            action=best,
            mean_quality_gain=means,
            uncertainty=uncertainty,
            conservative_net_value=net,
            stop_reason=reason,
            current_correctness_estimate=correctness_estimate,
            correctness_uncertainty=correctness_uncertainty,
        )


class FixedRuntimePolicy:
    """On-path fixed-depth control: execute one action N times, then stop."""

    def __init__(
        self, action: Action | str, *, max_actions: int = 1,
        utility: UtilityConfig = UtilityConfig(),
    ) -> None:
        self.action = Action(action)
        self.max_actions = max(0, int(max_actions))
        self.utility = utility

    def decide(self, state: ReasoningState, *, blocked_actions: Iterable[str] = ()) -> ControllerDecision:
        blocked = set(blocked_actions)
        should_stop = (
            self.action is Action.STOP
            or state.step >= self.max_actions
            or self.action.value in blocked
            or self.utility.action_cost(self.action) > state.budget_remaining
        )
        selected = Action.STOP.value if should_stop else self.action.value
        zeros = {action.value: 0.0 for action in ALL_ACTIONS}
        return ControllerDecision(
            action=selected,
            mean_quality_gain=zeros,
            uncertainty=zeros,
            conservative_net_value=zeros,
            stop_reason=StopReason.NON_POSITIVE_VOC.value if should_stop else "",
        )


class ThresholdRuntimePolicy:
    """Cheap confidence/entropy heuristic for real on-path controls."""

    def __init__(
        self, action: Action | str, *, feature: str, threshold: float,
        max_actions: int = 1, utility: UtilityConfig = UtilityConfig(),
    ) -> None:
        if feature not in {"confidence", "entropy"}:
            raise ValueError("ThresholdRuntimePolicy feature must be confidence or entropy")
        self.action = Action(action)
        self.feature = feature
        self.threshold = float(threshold)
        self.max_actions = max(0, int(max_actions))
        self.utility = utility

    def decide(self, state: ReasoningState, *, blocked_actions: Iterable[str] = ()) -> ControllerDecision:
        value = state.answer_confidence if self.feature == "confidence" else state.answer_entropy
        continue_warranted = (
            value < self.threshold if self.feature == "confidence" else value >= self.threshold
        )
        if not continue_warranted:
            return FixedRuntimePolicy(Action.STOP, utility=self.utility).decide(state)
        return FixedRuntimePolicy(
            self.action, max_actions=self.max_actions, utility=self.utility,
        ).decide(state, blocked_actions=blocked_actions)
