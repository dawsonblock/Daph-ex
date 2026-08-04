"""Task-level E2↔E3 comparisons for refinement qualification."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence


def e3_pair_metrics(pairs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize verified E2/E3 outcomes, including rescues and regressions."""
    rows = list(pairs)
    if not rows:
        raise ValueError("At least one verified E2/E3 pair is required")

    def summarize(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        n = len(items)
        e2 = sum(bool(item["e2_correct"]) for item in items)
        e3 = sum(bool(item["e3_correct"]) for item in items)
        rescues = sum(not bool(item["e2_correct"]) and bool(item["e3_correct"]) for item in items)
        regressions = sum(bool(item["e2_correct"]) and not bool(item["e3_correct"]) for item in items)
        return {
            "tasks": n, "e2_accuracy": e2 / n, "e3_accuracy": e3 / n,
            "rescues": rescues, "regressions": regressions,
            "rescue_rate": rescues / n, "regression_rate": regressions / n,
            "net_rescue_rate": (rescues - regressions) / n,
            "e3_correct_given_e2_wrong": rescues / max(n - e2, 1),
            "e3_wrong_given_e2_correct": regressions / max(e2, 1),
        }

    by_difficulty: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_family: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_difficulty[str(row.get("difficulty_bucket", "unspecified"))].append(row)
        by_family[str(row.get("task_family", "unspecified"))].append(row)
    return {
        **summarize(rows),
        "by_difficulty": {key: summarize(value) for key, value in sorted(by_difficulty.items())},
        "by_task_family": {key: summarize(value) for key, value in sorted(by_family.items())},
    }
