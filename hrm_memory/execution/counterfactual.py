from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

from hrm_memory.controller.actions import Action, ActionOutcome


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
    quality: float
    compute_cost: float
    utility: float
    reference_action: Action
    delta_utility_vs_reference: float
    outcome_metadata: Mapping[str, Any] = field(default_factory=dict)


class CounterfactualCollector:
    """Execute every action from an isolated copy of a reachable state."""

    def __init__(self, executors: Mapping[Action, Callable[[DecisionState], ActionOutcome]],
                 *, lambda_compute: float = 1.0):
        if Action.STOP not in executors and Action.ANSWER not in executors:
            raise ValueError("Collector needs STOP or ANSWER as the reference action")
        self.executors = dict(executors); self.lambda_compute = lambda_compute

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
        reference = outcomes[reference_action].utility(lambda_compute=self.lambda_compute)
        return [CounterfactualRecord(
            task_id=state.task_id, state_id=state.state_id, step=state.step, action=action,
            quality=outcome.quality, compute_cost=outcome.compute_cost,
            utility=outcome.utility(lambda_compute=self.lambda_compute),
            reference_action=reference_action,
            delta_utility_vs_reference=outcome.utility(lambda_compute=self.lambda_compute) - reference,
        ) for action, outcome in outcomes.items()]
