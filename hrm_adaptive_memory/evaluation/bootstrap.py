"""Deterministic grouped bootstrap for correlated task templates."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Percentile requires values")
    index = max(0, min(len(values) - 1, int(math.floor(probability * len(values)))))
    return float(values[index])


def grouped_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, value_key: str, group_key: str,
    samples: int = 10_000, confidence: float = 0.95, seed: int = 42,
) -> dict[str, Any]:
    if not rows or samples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid grouped bootstrap input")
    groups: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        group = str(row.get(group_key, ""))
        if not group or value_key not in row:
            raise ValueError(f"Row {index} lacks {group_key} or {value_key}")
        groups[group].append(float(row[value_key]))
    names = sorted(groups)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled: list[float] = []
        for _ in names:
            sampled.extend(groups[names[rng.randrange(len(names))]])
        estimates.append(sum(sampled) / len(sampled))
    estimates.sort()
    alpha = 1.0 - confidence
    observed = [float(row[value_key]) for row in rows]
    return {
        "mean": sum(observed) / len(observed),
        "standard_error": statistics.stdev(estimates) if len(estimates) > 1 else 0.0,
        "ci95": [_percentile(estimates, alpha / 2), _percentile(estimates, 1 - alpha / 2)],
        "lcb95": _percentile(estimates, alpha),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "bootstrap_unit": group_key,
        "group_count": len(names),
        "task_count": len(rows),
        "seed": seed,
    }
