from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping


@dataclass(frozen=True)
class BaselineMetrics:
    accuracy: float
    exact_match: float
    verified_utility: float
    prompt_tokens: float
    completion_tokens: float
    latency_ms: float
    peak_memory_bytes: float


def aggregate_baseline_metrics(rows: Iterable[Mapping[str, float]]) -> BaselineMetrics:
    values = list(rows)
    if not values:
        raise ValueError("At least one baseline result is required")
    def avg(key: str) -> float:
        return mean(float(row.get(key, 0.0)) for row in values)
    return BaselineMetrics(
        accuracy=avg("accuracy"), exact_match=avg("exact_match"),
        verified_utility=avg("verified_utility"), prompt_tokens=avg("prompt_tokens"),
        completion_tokens=avg("completion_tokens"), latency_ms=avg("latency_ms"),
        peak_memory_bytes=avg("peak_memory_bytes"),
    )
