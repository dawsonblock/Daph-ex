"""Observable utility and explicit value-of-computation accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .schema import Action, ActionReceipt


@dataclass(frozen=True)
class UtilityConfig:
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    abstention_reward: float = -0.1
    compute_weight: float = 1.0
    action_base_cost: Mapping[str, float] = field(default_factory=lambda: {
        Action.STOP.value: 0.0,
        Action.THINK.value: 0.02,
        Action.VERIFY.value: 0.04,
        Action.DECOMPOSE.value: 0.03,
    })
    normalized_compute_weight: float = 0.0

    def action_cost(self, action: Action | str, receipt: ActionReceipt | None = None) -> float:
        name = action.value if isinstance(action, Action) else str(action)
        base = float(self.action_base_cost.get(name, 0.0))
        measured = 0.0 if receipt is None else float(receipt.normalized_compute)
        return self.compute_weight * base + self.normalized_compute_weight * measured

    def quality(self, verified_quality: float, *, abstained: bool = False) -> float:
        if abstained:
            return float(self.abstention_reward)
        return self.correct_reward if float(verified_quality) > 0.0 else self.incorrect_reward

    def voc(
        self,
        *,
        quality_before: float,
        quality_after: float,
        action: Action | str,
        receipt: ActionReceipt | None = None,
    ) -> tuple[float, float, float]:
        """Return (gross quality delta, cost, net marginal utility/VOC)."""
        delta_quality = float(quality_after) - float(quality_before)
        cost = self.action_cost(action, receipt)
        return delta_quality, cost, delta_quality - cost
