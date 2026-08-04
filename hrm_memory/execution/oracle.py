from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from .counterfactual import CounterfactualRecord


def oracle_opportunity(records: Iterable[CounterfactualRecord], *, minimum_gap: float = 0.01) -> dict[str, object]:
    by_state: dict[str, list[CounterfactualRecord]] = defaultdict(list)
    for record in records: by_state[record.state_id].append(record)
    if not by_state: raise ValueError("Counterfactual records are required")
    actions = sorted({record.action for rows in by_state.values() for record in rows}, key=lambda action: action.value)
    fixed = {action: mean(next(row.utility for row in rows if row.action == action) for rows in by_state.values()) for action in actions if all(any(row.action == action for row in rows) for rows in by_state.values())}
    best_fixed_action, best_fixed = max(fixed.items(), key=lambda item: item[1])
    oracle = mean(max(row.utility for row in rows) for rows in by_state.values())
    gap = oracle - best_fixed
    return {"states": len(by_state), "best_fixed_action": best_fixed_action.value, "best_fixed_utility": best_fixed,
            "oracle_utility": oracle, "oracle_gap": gap, "passed": gap >= minimum_gap,
            "controller_training_allowed": gap >= minimum_gap}
