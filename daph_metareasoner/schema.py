"""Immutable schemas for marginal-utility (value-of-computation) research."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Tuple


class Action(str, Enum):
    STOP = "STOP"
    THINK = "THINK"
    VERIFY = "VERIFY"
    DECOMPOSE = "DECOMPOSE"


NON_STOP_ACTIONS: Tuple[Action, ...] = (
    Action.THINK,
    Action.VERIFY,
    Action.DECOMPOSE,
)
ALL_ACTIONS: Tuple[Action, ...] = (Action.STOP, *NON_STOP_ACTIONS)


class StopReason(str, Enum):
    SUCCESS = "STOP_SUCCESS"
    FAILURE = "STOP_FAILURE"
    BUDGET = "STOP_BUDGET"
    NON_POSITIVE_VOC = "STOP_NON_POSITIVE_VOC"
    LOOP_GUARD = "STOP_LOOP_GUARD"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    expected: str
    family_id: str
    split: str
    template_id: str = "unspecified"
    generator_seed: str = "unspecified"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class ReasoningState:
    state_id: str
    task_id: str
    step: int
    answer: str
    prompt: str
    evidence: Tuple[str, ...] = ()
    action_history: Tuple[str, ...] = ()
    hidden_by_depth: Mapping[str, Tuple[float, ...]] = field(default_factory=dict)
    hidden_final_token: Tuple[float, ...] = ()
    answer_entropy: float = 0.0
    answer_logprob: float = 0.0
    answer_confidence: float = 0.0
    token_count: int = 0
    compute_spent: float = 0.0
    budget_remaining: float = 1.0
    answer_changed: bool = False
    hidden_cosine_previous: float = 0.0
    confidence_delta: float = 0.0
    repeated_answer_count: int = 0
    initial_latency_ms: float = 0.0
    initial_input_tokens: int = 0
    initial_output_tokens: int = 0

    @classmethod
    def create(cls, **kwargs: Any) -> "ReasoningState":
        identity = {
            key: kwargs.get(key)
            for key in ("task_id", "step", "answer", "prompt", "evidence", "action_history")
        }
        return cls(state_id=canonical_digest(identity), **kwargs)

    def cheap_features(self) -> Tuple[float, ...]:
        prompt = self.prompt
        return (
            float(len(prompt)),
            float(sum(char.isdigit() for char in prompt)),
            float(sum(char in "+-*/=<>()" for char in prompt)),
            float(self.answer_entropy),
            float(self.answer_logprob),
            float(self.answer_confidence),
            float(self.token_count),
            float(self.step),
            float(self.compute_spent),
            float(self.budget_remaining),
            float(self.answer_changed),
            float(self.confidence_delta),
            float(self.repeated_answer_count),
        )

    def hidden_features(self, depths: Tuple[str, ...] = ("25", "50", "75", "100")) -> Tuple[float, ...]:
        values = []
        for depth in depths:
            values.extend(self.hidden_by_depth.get(depth, ()))
        return tuple(values)


@dataclass(frozen=True)
class ActionReceipt:
    action: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    normalized_compute: float
    model_digest: str
    environment_digest: str
    error: str = ""


@dataclass(frozen=True)
class BranchResult:
    action: Action
    state_before_id: str
    next_state: ReasoningState
    receipt: ActionReceipt


@dataclass(frozen=True)
class ExperienceRecord:
    task: Task
    state: ReasoningState
    action: str
    next_state_id: str
    answer_before: str
    answer_after: str
    verifier_status_before: str
    verifier_status_after: str
    quality_before: float
    quality_after: float
    delta_quality: float
    action_cost: float
    delta_utility: float
    receipt: ActionReceipt
    model_digest: str
    environment_digest: str
    dataset_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperienceRecord":
        data = dict(payload)
        data["task"] = Task(**data["task"])
        state = dict(data["state"])
        state["evidence"] = tuple(state.get("evidence", ()))
        state["action_history"] = tuple(state.get("action_history", ()))
        state["hidden_final_token"] = tuple(state.get("hidden_final_token", ()))
        state["hidden_by_depth"] = {
            str(key): tuple(value) for key, value in state.get("hidden_by_depth", {}).items()
        }
        data["state"] = ReasoningState(**state)
        data["receipt"] = ActionReceipt(**data["receipt"])
        return cls(**data)


def records_digest(records: Tuple[ExperienceRecord, ...] | list[ExperienceRecord]) -> str:
    ordered = sorted(records, key=lambda row: (row.task.task_id, row.state.state_id, row.action))
    return canonical_digest([record.to_dict() for record in ordered])
