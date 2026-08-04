"""Receipt-backed E2/E3 quality and cost-aware utility qualification.

Quality and utility are intentionally separate quantities.  A paired record is
invalid unless it supplies measured quality and normalized compute for both
arms; effort IDs are never used to infer cost.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import random
import statistics
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .e3_protocol import ExperimentScale


REQUIRED_PAIR_FIELDS = (
    "task_id", "quality_e2", "quality_e3", "compute_e2", "compute_e3",
    "e2_correct", "e3_correct", "task_family", "template_id",
)


@dataclass(frozen=True)
class E3QualificationConfig:
    """Predeclared paired quality and utility evidence thresholds."""

    lambda_compute: float = 1.0
    bootstrap_samples: int = 2000
    confidence: float = 0.95
    group_key: str = "template_id"
    min_quality_delta: float = 0.0
    min_utility_delta: float = 0.0
    min_net_rescue_rate: float = 0.0
    # A smoke study needs enough examples to be a meaningful smoke result. A
    # stronger tier raises this through ``experiment_scale``.
    min_tasks: int = 24
    min_groups: int = 2
    seed: int = 42
    experiment_scale: Optional[ExperimentScale] = None

    def validate(self) -> None:
        if self.lambda_compute < 0 or not math.isfinite(self.lambda_compute):
            raise ValueError("lambda_compute must be finite and non-negative")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if self.min_tasks < 1 or self.min_groups < 1:
            raise ValueError("min_tasks and min_groups must be positive")
        if not self.group_key:
            raise ValueError("group_key must be non-empty")
        if self.experiment_scale is not None and not isinstance(self.experiment_scale, ExperimentScale):
            raise TypeError("experiment_scale must be an ExperimentScale or None")


def _validate_pairs(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("At least one verified E2/E3 pair is required")
    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_PAIR_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"Pair {index} is missing required receipt-backed fields: {', '.join(missing)}. "
                "Quality may not be relabeled as utility and compute may not be inferred from effort IDs."
            )
        for field in ("quality_e2", "quality_e3", "compute_e2", "compute_e3"):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"Pair {index} has non-finite {field}")
        if float(row["compute_e2"]) < 0 or float(row["compute_e3"]) < 0:
            raise ValueError(f"Pair {index} has negative compute")


def materialize_utility_record(
    row: Mapping[str, Any], *, lambda_compute: float,
) -> Dict[str, Any]:
    """Return explicit Q, C, and U fields for one paired task record."""
    _validate_pairs([row])
    q2, q3 = float(row["quality_e2"]), float(row["quality_e3"])
    c2, c3 = float(row["compute_e2"]), float(row["compute_e3"])
    u2, u3 = q2 - lambda_compute * c2, q3 - lambda_compute * c3
    out = dict(row)
    out.update({
        "utility_e2": u2,
        "utility_e3": u3,
        "delta_quality": q3 - q2,
        "delta_compute": c3 - c2,
        "delta_utility": u3 - u2,
        "rescue": not bool(row["e2_correct"]) and bool(row["e3_correct"]),
        "regression": bool(row["e2_correct"]) and not bool(row["e3_correct"]),
        "lambda_compute": float(lambda_compute),
    })
    return out


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    index = max(0, min(len(sorted_values) - 1, int(math.floor(probability * len(sorted_values)))))
    return float(sorted_values[index])


def grouped_bootstrap(
    rows: Sequence[Mapping[str, Any]], value_key: str, *, group_key: str,
    samples: int, confidence: float, seed: int,
) -> Dict[str, Any]:
    """Cluster bootstrap whole template/family groups with replacement."""
    if not rows:
        raise ValueError("Cannot bootstrap an empty sequence")
    groups: Dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        if value_key not in row:
            raise ValueError(f"Row {index} is missing {value_key}")
        group = str(row.get(group_key, ""))
        if not group:
            raise ValueError(f"Row {index} is missing bootstrap group {group_key}")
        groups[group].append(float(row[value_key]))
    names = sorted(groups)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        sampled: list[float] = []
        for _ in names:
            sampled.extend(groups[names[rng.randrange(len(names))]])
        means.append(sum(sampled) / len(sampled))
    means.sort()
    alpha = 1.0 - confidence
    mean = sum(float(row[value_key]) for row in rows) / len(rows)
    se = statistics.stdev(means) if len(means) > 1 else 0.0
    return {
        "mean": mean,
        "standard_error": se,
        "ci95": [
            _percentile(means, alpha / 2.0),
            _percentile(means, 1.0 - alpha / 2.0),
        ],
        "lcb95": _percentile(means, alpha),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "bootstrap_unit": group_key,
        "group_count": len(names),
        "task_count": len(rows),
        "seed": seed,
    }


def e3_pair_metrics(pairs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize verified E2/E3 outcomes without calling quality utility."""
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

    dimensions = {
        "by_difficulty": "difficulty",
        "by_task_family": "task_family",
        "by_refinement_steps": "refinement_steps",
        "by_profiled_region": "profiled_region",
        "by_seed": "seed",
    }
    result = summarize(rows)
    for output_key, row_key in dimensions.items():
        grouped: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get(row_key, row.get("difficulty_bucket") if row_key == "difficulty" else "unspecified")
            grouped[str(value if value is not None else "unspecified")].append(row)
        result[output_key] = {key: summarize(value) for key, value in sorted(grouped.items())}
    return result


def qualify_e3_pairs(
    pairs: Iterable[Mapping[str, Any]],
    config: E3QualificationConfig = E3QualificationConfig(),
) -> Dict[str, Any]:
    """Run distinct capability (E3-Q) and cost-aware utility (E3-U) gates."""
    config.validate()
    source_rows = list(pairs)
    _validate_pairs(source_rows)
    rows = [materialize_utility_record(row, lambda_compute=config.lambda_compute) for row in source_rows]
    summary = e3_pair_metrics(rows)
    quality = grouped_bootstrap(
        rows, "delta_quality", group_key=config.group_key,
        samples=config.bootstrap_samples, confidence=config.confidence, seed=config.seed,
    )
    utility = grouped_bootstrap(
        rows, "delta_utility", group_key=config.group_key,
        samples=config.bootstrap_samples, confidence=config.confidence, seed=config.seed + 1,
    )
    unique_task_count = len({str(row["task_id"]) for row in rows})
    local_power = (
        unique_task_count >= config.min_tasks
        and quality["group_count"] >= config.min_groups
    )
    scale_report = None
    if config.experiment_scale is not None:
        # Validate the predeclared scale before a result can be promotable.  A
        # short or under-replicated result stays reportable as INSUFFICIENT_POWER
        # instead of being allowed to masquerade as a qualification pass.
        try:
            config.experiment_scale.validate()
        except ValueError:
            pass
        observed_seeds = sorted({
            int(row["training_seed"]) for row in rows
            if row.get("training_seed") is not None
        })
        scale_report = config.experiment_scale.validation_report(
            observed_tasks=unique_task_count, observed_groups=quality["group_count"],
            observed_training_seeds=observed_seeds,
        )
    sufficiently_powered = local_power and (scale_report is None or bool(scale_report["passed"]))
    rescue_condition = (
        summary["rescue_count"] > summary["regression_count"]
        and summary["net_rescue_rate"] > config.min_net_rescue_rate
    )
    quality_pass = bool(
        sufficiently_powered
        and quality["mean"] > config.min_quality_delta
        and quality["lcb95"] > config.min_quality_delta
        and rescue_condition
    )
    utility_pass = bool(
        sufficiently_powered
        and utility["mean"] > config.min_utility_delta
        and utility["lcb95"] > config.min_utility_delta
    )
    if not sufficiently_powered:
        status = "INSUFFICIENT_POWER"
    elif not quality_pass:
        status = "FAIL_QUALITY"
    elif not utility_pass:
        status = "PASS_QUALITY_FAIL_UTILITY"
    else:
        status = "PASS_QUALITY_AND_UTILITY"
    return {
        **summary,
        "mean_quality_delta": quality["mean"],
        "quality_standard_error": quality["standard_error"],
        "quality_ci95": quality["ci95"],
        "quality_lcb95": quality["lcb95"],
        "mean_compute_delta": sum(row["delta_compute"] for row in rows) / len(rows),
        "mean_utility_delta": utility["mean"],
        "utility_standard_error": utility["standard_error"],
        "utility_ci95": utility["ci95"],
        "utility_lcb95": utility["lcb95"],
        "lambda_compute": config.lambda_compute,
        "quality_gate": {"name": "E3-Q", "passed": quality_pass, "bootstrap": quality},
        "utility_gate": {"name": "E3-U", "passed": utility_pass, "bootstrap": utility},
        "qualification_status": status,
        "unique_task_count": unique_task_count,
        "sufficiently_powered": sufficiently_powered,
        "local_power_passed": local_power,
        "experiment_scale": scale_report,
        "qualified": status == "PASS_QUALITY_AND_UTILITY",
        "policy_training_allowed": False,
        "requires_oracle_opportunity_gate": True,
        "paired_records": rows,
        "thresholds": {
            "min_quality_delta": config.min_quality_delta,
            "min_utility_delta": config.min_utility_delta,
            "min_net_rescue_rate": config.min_net_rescue_rate,
            "min_tasks": config.min_tasks,
            "min_groups": config.min_groups,
            "bootstrap_samples": config.bootstrap_samples,
            "bootstrap_group_key": config.group_key,
        },
    }


def lambda_sweep(
    pairs: Iterable[Mapping[str, Any]], lambdas: Sequence[float], *,
    bootstrap_samples: int = 2000, confidence: float = 0.95,
    group_key: str = "template_id", seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate cost-aware utility over predeclared compute prices."""
    source_rows = list(pairs)
    _validate_pairs(source_rows)
    values = sorted(set(float(value) for value in lambdas))
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("Lambda sweep values must be finite and non-negative")
    dq = [float(row["quality_e3"]) - float(row["quality_e2"]) for row in source_rows]
    dc = [float(row["compute_e3"]) - float(row["compute_e2"]) for row in source_rows]
    mean_dq, mean_dc = sum(dq) / len(dq), sum(dc) / len(dc)
    aggregate_break_even = mean_dq / mean_dc if mean_dc > 0 else None
    per_example = [q / c for q, c in zip(dq, dc) if c > 0]
    results = []
    for offset, value in enumerate(values):
        rows = [materialize_utility_record(row, lambda_compute=value) for row in source_rows]
        boot = grouped_bootstrap(
            rows, "delta_utility", group_key=group_key, samples=bootstrap_samples,
            confidence=confidence, seed=seed + offset,
        )
        results.append({
            "lambda_compute": value,
            "mean_delta_utility": boot["mean"],
            "utility_lcb95": boot["lcb95"],
            "utility_ci95": boot["ci95"],
            "fraction_e3_higher_utility": sum(row["delta_utility"] > 0 for row in rows) / len(rows),
        })
    return {
        "lambda_values": values,
        "results": results,
        "mean_delta_quality": mean_dq,
        "mean_delta_compute": mean_dc,
        "aggregate_break_even_lambda": aggregate_break_even,
        "per_example_break_even_lambda": {
            "count": len(per_example),
            "mean": sum(per_example) / len(per_example) if per_example else None,
            "median": statistics.median(per_example) if per_example else None,
            "min": min(per_example) if per_example else None,
            "max": max(per_example) if per_example else None,
        },
        "bootstrap": {
            "samples": bootstrap_samples, "confidence": confidence,
            "group_key": group_key, "seed": seed,
        },
    }


def qualification_config_dict(config: E3QualificationConfig) -> Dict[str, Any]:
    return asdict(config)
