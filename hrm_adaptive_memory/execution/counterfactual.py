from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

from hrm_adaptive_memory.controller.actions import Action, ActionOutcome


@dataclass(frozen=True)
class DecisionState:
    task_id: str
    step: int
    hidden_summary: tuple[float, ...]
    current_answer: str
    evidence_ids: tuple[str, ...] = ()
    remaining_token_budget: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def state_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class CounterfactualRecord:
    task_id: str
    state_id: str
    step: int
    action: Action
    gross_quality: float
    retrieval_cost: float
    compute_cost: float
    latency_cost: float
    token_cost: float
    verification_cost: float
    utility: float
    reference_action: Action
    delta_utility_vs_reference: float
    outcome_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def quality(self) -> float:
        """Compatibility alias for older analysis code."""

        return self.gross_quality


class CounterfactualCollector:
    """Execute every action from an isolated copy of a reachable state."""

    def __init__(self, executors: Mapping[Action, Callable[[DecisionState], ActionOutcome]],
                 *, lambda_compute: float = 1.0, lambda_latency: float = 0.0,
                 lambda_tokens: float = 0.0, lambda_retrieval: float = 0.0,
                 lambda_verification: float = 0.0):
        if Action.STOP not in executors and Action.ANSWER not in executors:
            raise ValueError("Collector needs STOP or ANSWER as the reference action")
        self.executors = dict(executors)
        self.utility_weights = {
            "lambda_compute": float(lambda_compute),
            "lambda_latency": float(lambda_latency),
            "lambda_tokens": float(lambda_tokens),
            "lambda_retrieval": float(lambda_retrieval),
            "lambda_verification": float(lambda_verification),
        }

    def collect(self, state: DecisionState, actions: Sequence[Action] | None = None) -> list[CounterfactualRecord]:
        requested = tuple(actions or self.executors.keys())
        outcomes: dict[Action, ActionOutcome] = {}
        for action in requested:
            if action not in self.executors:
                raise KeyError(f"No executor for {action.value}")
            outcome = self.executors[action](copy.deepcopy(state))
            if outcome.action != action:
                raise ValueError("Executor returned the wrong action")
            outcomes[action] = outcome
        if Action.STOP in outcomes:
            reference_action = Action.STOP
        elif Action.ANSWER in outcomes:
            reference_action = Action.ANSWER
        else:
            raise ValueError("Requested actions must include STOP or ANSWER as reference")
        reference = outcomes[reference_action].utility(**self.utility_weights)
        return [CounterfactualRecord(
            task_id=state.task_id, state_id=state.state_id, step=state.step, action=action,
            gross_quality=outcome.quality,
            retrieval_cost=outcome.retrieval_cost,
            compute_cost=outcome.compute_cost,
            latency_cost=outcome.latency_cost,
            token_cost=outcome.token_cost,
            verification_cost=outcome.verification_cost,
            utility=outcome.utility(**self.utility_weights),
            reference_action=reference_action,
            delta_utility_vs_reference=outcome.utility(**self.utility_weights) - reference,
        ) for action, outcome in outcomes.items()]
