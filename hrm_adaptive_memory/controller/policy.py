"""Utility-based decision rule. Training remains outside this foundation gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .actions import Action


@dataclass(frozen=True)
class ControllerDecision:
    action: Action
    expected_net_utility: float
    values: Mapping[Action, float]
    stopped: bool


class UtilityController:
    def __init__(self, *, allowed_actions: tuple[Action, ...] = (Action.ANSWER, Action.RETRIEVE),
                 verified_fit: bool = False):
        self.allowed_actions = allowed_actions; self.verified_fit = verified_fit

    def decide(self, predicted_marginal_benefit: Mapping[Action, float],
               action_costs: Mapping[Action, float] | None = None,
               *, research_override: bool = False) -> ControllerDecision:
        if not self.verified_fit and not research_override:
            raise RuntimeError("Adaptive execution requires a VERIFIED_FIT controller")
        costs = action_costs or {}
        net = {
            action: float(predicted_marginal_benefit.get(action, 0.0))
            - float(costs.get(action, 0.0))
            for action in self.allowed_actions
        }
        non_terminal = {action: value for action, value in net.items() if action not in (Action.ANSWER, Action.STOP)}
        best_action, best_value = max(net.items(), key=lambda item: (item[1], item[0].value))
        if not non_terminal or max(non_terminal.values()) <= 0:
            stop_action = Action.ANSWER if Action.ANSWER in self.allowed_actions else Action.STOP
            return ControllerDecision(stop_action, net.get(stop_action, 0.0), net, True)
        return ControllerDecision(best_action, best_value, net, False)
