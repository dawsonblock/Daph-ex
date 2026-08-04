"""Task-level E2↔E3 comparisons for refinement qualification."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Any, Dict, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class E3QualificationConfig:
    """Predeclared task-level evidence threshold for E3 promotion."""

    bootstrap_samples: int = 2000
    confidence: float = 0.95
    min_verified_utility_delta: float = 0.0
    min_net_rescue_rate: float = 0.0
    seed: int = 42


def _bootstrap_lcb(values: Sequence[float], config: E3QualificationConfig) -> float:
    if not values:
        raise ValueError("Cannot bootstrap an empty sequence")
    rng = random.Random(config.seed)
    n = len(values)
    means = []
    for _ in range(max(1, config.bootstrap_samples)):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    tail = max(0.0, min(1.0, 1.0 - config.confidence))
    return means[min(len(means) - 1, int(tail * len(means)))]


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
            "rescue_count": rescues, "regression_count": regressions,
            "rescue_rate": rescues / n, "regression_rate": regressions / n,
            "net_rescue_rate": (rescues - regressions) / n,
            "e3_correct_given_e2_wrong": rescues / max(n - e2, 1),
            "e3_wrong_given_e2_correct": regressions / max(e2, 1),
        }

    by_difficulty: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_family: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_steps: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_region: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_difficulty[str(row.get("difficulty_bucket", "unspecified"))].append(row)
        by_family[str(row.get("task_family", "unspecified"))].append(row)
        by_steps[str(row.get("refinement_steps", "unspecified"))].append(row)
        by_region[str(row.get("profiled_region", "unspecified"))].append(row)
    return {
        **summarize(rows),
        "by_difficulty": {key: summarize(value) for key, value in sorted(by_difficulty.items())},
        "by_task_family": {key: summarize(value) for key, value in sorted(by_family.items())},
        "by_refinement_steps": {key: summarize(value) for key, value in sorted(by_steps.items())},
        "by_profiled_region": {key: summarize(value) for key, value in sorted(by_region.items())},
    }


def qualify_e3_pairs(
    pairs: Iterable[Mapping[str, Any]],
    config: E3QualificationConfig = E3QualificationConfig(),
) -> Dict[str, Any]:
    """Qualify E3 using paired verified outcomes and predeclared uncertainty.

    Optional ``verified_utility_e2/e3`` fields override binary correctness.
    Cross-entropy is deliberately not a qualification input.
    """
    rows = list(pairs)
    summary = e3_pair_metrics(rows)
    deltas = [
        float(row.get("verified_utility_e3", bool(row["e3_correct"])))
        - float(row.get("verified_utility_e2", bool(row["e2_correct"])))
        for row in rows
    ]
    mean_delta = sum(deltas) / len(deltas)
    lcb = _bootstrap_lcb(deltas, config)
    qualified = (
        mean_delta > config.min_verified_utility_delta
        and lcb > config.min_verified_utility_delta
        and summary["rescue_count"] > summary["regression_count"]
        and summary["net_rescue_rate"] > config.min_net_rescue_rate
    )
    return {
        **summary,
        "mean_verified_utility_delta": mean_delta,
        "quality_delta_lcb": lcb,
        "confidence": config.confidence,
        "qualified": qualified,
        "policy_training_allowed": qualified,
        "thresholds": {
            "min_verified_utility_delta": config.min_verified_utility_delta,
            "min_net_rescue_rate": config.min_net_rescue_rate,
            "bootstrap_samples": config.bootstrap_samples,
        },
    }
