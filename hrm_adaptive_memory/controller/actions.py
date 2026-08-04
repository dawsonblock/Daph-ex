from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Action(str, Enum):
    ANSWER = "ANSWER"
    RETRIEVE = "RETRIEVE"
    VERIFY = "VERIFY"
    CONTINUE = "CONTINUE"
    STOP = "STOP"


@dataclass(frozen=True)
class ActionOutcome:
    action: Action
    quality: float
    compute_cost: float = 0.0
    latency_cost: float = 0.0
    token_cost: float = 0.0

    def utility(self, *, lambda_compute: float = 1.0, lambda_latency: float = 0.0,
                lambda_tokens: float = 0.0) -> float:
        return self.quality - lambda_compute * self.compute_cost - lambda_latency * self.latency_cost - lambda_tokens * self.token_cost


def action_utilities(outcomes: Iterable[ActionOutcome], **weights: float) -> dict[Action, float]:
    values = list(outcomes)
    if len({outcome.action for outcome in values}) != len(values):
        raise ValueError("Action outcomes must be unique by action")
    return {outcome.action: outcome.utility(**weights) for outcome in values}
