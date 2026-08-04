"""Baseline conditions and the mandatory oracle-context diagnostic gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean
from typing import Iterable


class BaselineCondition(str, Enum):
    NO_CONTEXT = "B0_NO_CONTEXT"
    RANDOM_CONTEXT = "B1_RANDOM_CONTEXT"
    NAIVE_RETRIEVAL = "B2_NAIVE_RETRIEVAL"
    ORACLE_EVIDENCE = "B3_ORACLE_EVIDENCE"


@dataclass(frozen=True)
class BaselineResult:
    task_id: str
    condition: BaselineCondition
    quality: float
    verified_utility: float
    exact_match: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    peak_memory_bytes: int = 0
    task_family: str = "unknown"
    difficulty: str = "unknown"


class OracleContextGate:
    """Determine whether HRM can exploit perfect evidence before RAG work."""

    def __init__(self, *, minimum_oracle_quality_gain: float = 0.05,
                 minimum_paired_tasks: int = 2):
        if minimum_oracle_quality_gain <= 0:
            raise ValueError("minimum_oracle_quality_gain must be positive")
        if minimum_paired_tasks < 1:
            raise ValueError("minimum_paired_tasks must be positive")
        self.minimum_gain = float(minimum_oracle_quality_gain)
        self.minimum_paired_tasks = int(minimum_paired_tasks)

    def evaluate(self, rows: Iterable[BaselineResult]) -> dict[str, object]:
        values = list(rows)
        by_condition: dict[BaselineCondition, list[BaselineResult]] = {}
        for row in values:
            by_condition.setdefault(row.condition, []).append(row)
        required = {BaselineCondition.NO_CONTEXT, BaselineCondition.ORACLE_EVIDENCE}
        missing = sorted(condition.value for condition in required - by_condition.keys())
        if missing:
            raise ValueError(f"Missing oracle-gate conditions: {missing}")
        for condition, condition_rows in by_condition.items():
            identifiers = [row.task_id for row in condition_rows]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"Duplicate task IDs in {condition.value}")
        base_ids = {row.task_id for row in by_condition[BaselineCondition.NO_CONTEXT]}
        oracle_ids = {row.task_id for row in by_condition[BaselineCondition.ORACLE_EVIDENCE]}
        if base_ids != oracle_ids:
            raise ValueError("Oracle gate requires paired task IDs")
        if len(base_ids) < self.minimum_paired_tasks:
            raise ValueError(
                f"Oracle gate requires at least {self.minimum_paired_tasks} paired tasks"
            )
        means = {
            condition.value: mean(row.quality for row in condition_rows)
            for condition, condition_rows in by_condition.items()
        }
        utility_means = {
            condition.value: mean(row.verified_utility for row in condition_rows)
            for condition, condition_rows in by_condition.items()
        }
        gain = means[BaselineCondition.ORACLE_EVIDENCE.value] - means[BaselineCondition.NO_CONTEXT.value]
        utility_gain = (
            utility_means[BaselineCondition.ORACLE_EVIDENCE.value]
            - utility_means[BaselineCondition.NO_CONTEXT.value]
        )
        passed = gain >= self.minimum_gain
        return {
            "status": "PASS_HRM_CAN_USE_EVIDENCE" if passed else "FAIL_HRM_EVIDENCE_USE",
            "passed": passed,
            "mean_quality_by_condition": means,
            "mean_verified_utility_by_condition": utility_means,
            "oracle_quality_gain": gain,
            "oracle_verified_utility_gain": utility_gain,
            "minimum_required_gain": self.minimum_gain,
            "minimum_paired_tasks": self.minimum_paired_tasks,
            "paired_tasks": len(base_ids),
            "next_stage": "STRUCTURED_RAG" if passed else "RETRIEVAL_CONDITIONED_ADAPTATION",
            "scientific_qualification": False,
            "retrieval_expansion_allowed": False,
            "controller_training_allowed": False,
        }
