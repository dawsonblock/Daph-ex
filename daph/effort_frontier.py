"""Cost-aware E0-E3 frontier and oracle-opportunity qualification."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .e3_metrics import grouped_bootstrap


EFFORTS = ("E0", "E1", "E2", "E3")


def _validate_frontier_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError("At least one per-task effort record is required")
    for index, row in enumerate(records):
        for field in ("task_id", "task_family", "template_id", "efforts"):
            if field not in row:
                raise ValueError(f"Frontier row {index} is missing {field}")
        efforts = row["efforts"]
        for effort in EFFORTS:
            if effort not in efforts:
                raise ValueError(f"Frontier row {index} is missing {effort}")
            arm = efforts[effort]
            for field in ("quality", "compute"):
                if field not in arm or not math.isfinite(float(arm[field])):
                    raise ValueError(f"Frontier row {index} {effort} needs finite actual {field}")
            if float(arm["compute"]) < 0:
                raise ValueError("Compute cannot be negative")


def build_effort_frontier(
    records: Iterable[Mapping[str, Any]], *, lambdas: Sequence[float],
    qualified_arms: Sequence[str] = (),
) -> Dict[str, Any]:
    """Build means, utility sweeps, and Pareto labels from actual receipts."""
    rows = list(records)
    _validate_frontier_records(rows)
    lambda_values = sorted(set(float(value) for value in lambdas))
    if not lambda_values or any(value < 0 for value in lambda_values):
        raise ValueError("At least one non-negative lambda is required")
    qualified = set(qualified_arms)
    arms: Dict[str, Dict[str, Any]] = {}
    for effort in EFFORTS:
        values = [row["efforts"][effort] for row in rows]
        quality = sum(float(value["quality"]) for value in values) / len(values)
        compute = sum(float(value["compute"]) for value in values) / len(values)
        accuracy_values = [float(value.get("accuracy", value["quality"])) for value in values]
        ce_values = [float(value["ce"]) for value in values if value.get("ce") is not None]
        latency_values = [float(value["latency_ms"]) for value in values if value.get("latency_ms") is not None]
        arms[effort] = {
            "quality": quality,
            "compute": compute,
            "compute_source": "per_task_actual_receipts",
            "accuracy": sum(accuracy_values) / len(accuracy_values),
            "ce": sum(ce_values) / len(ce_values) if ce_values else None,
            "latency_ms": sum(latency_values) / len(latency_values) if latency_values else None,
            "utility_by_lambda": {
                str(value): quality - value * compute for value in lambda_values
            },
        }
    dominated: set[str] = set()
    for effort in EFFORTS:
        for other in EFFORTS:
            if effort == other:
                continue
            if (
                arms[other]["compute"] <= arms[effort]["compute"]
                and arms[other]["quality"] >= arms[effort]["quality"]
                and (
                    arms[other]["compute"] < arms[effort]["compute"]
                    or arms[other]["quality"] > arms[effort]["quality"]
                )
            ):
                dominated.add(effort)
                break
    for effort in EFFORTS:
        if effort == "E2":
            status = "ANCHOR"
        elif effort in dominated:
            status = "DOMINATED"
        elif effort in qualified:
            status = "PARETO"
        else:
            status = "UNQUALIFIED"
        arms[effort]["frontier_status"] = status
    return {
        "tasks": len(rows),
        "lambda_values": lambda_values,
        "arms": arms,
        "pareto_arms": [effort for effort in EFFORTS if effort not in dominated],
        "dominated_arms": sorted(dominated),
        "physical_compute_ordering": all(
            all(float(row["efforts"][EFFORTS[index]]["compute"]) < float(row["efforts"][EFFORTS[index + 1]]["compute"]) for index in range(3))
            for row in rows
        ),
    }


def qualify_oracle_opportunity(
    records: Iterable[Mapping[str, Any]], *, lambda_compute: float,
    qualified_non_e2_arms: Sequence[str], bootstrap_samples: int = 2000,
    confidence: float = 0.95, group_key: str = "template_id",
    min_oracle_gap: float = 0.0, seed: int = 42,
) -> Dict[str, Any]:
    """Require a receipt-backed oracle gap and a qualified non-E2 arm."""
    rows = list(records)
    _validate_frontier_records(rows)
    means = {}
    for effort in EFFORTS:
        means[effort] = sum(
            float(row["efforts"][effort]["quality"])
            - lambda_compute * float(row["efforts"][effort]["compute"])
            for row in rows
        ) / len(rows)
    best_fixed = max(EFFORTS, key=lambda effort: (means[effort], -int(effort[-1])))
    oracle_rows = []
    counts: Counter[str] = Counter()
    for row in rows:
        utilities = {
            effort: float(row["efforts"][effort]["quality"])
            - lambda_compute * float(row["efforts"][effort]["compute"])
            for effort in EFFORTS
        }
        oracle_effort = max(EFFORTS, key=lambda effort: (utilities[effort], -int(effort[-1])))
        counts[oracle_effort] += 1
        oracle_rows.append({
            "task_id": row["task_id"],
            "task_family": row["task_family"],
            "template_id": row["template_id"],
            "oracle_effort": oracle_effort,
            "oracle_utility": utilities[oracle_effort],
            "best_fixed_utility": utilities[best_fixed],
            "oracle_gap": utilities[oracle_effort] - utilities[best_fixed],
        })
    bootstrap = grouped_bootstrap(
        oracle_rows, "oracle_gap", group_key=group_key, samples=bootstrap_samples,
        confidence=confidence, seed=seed,
    )
    qualified_arms = sorted(set(qualified_non_e2_arms) - {"E2"})
    has_arm = bool(qualified_arms)
    passes_oracle = bootstrap["mean"] > min_oracle_gap and bootstrap["lcb95"] > min_oracle_gap
    allowed = bool(has_arm and passes_oracle)
    probabilities = [count / len(rows) for count in counts.values() if count]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return {
        "lambda_compute": lambda_compute,
        "best_fixed_effort": best_fixed,
        "mean_fixed_utility": means,
        "mean_oracle_gap": bootstrap["mean"],
        "oracle_gap_lcb95": bootstrap["lcb95"],
        "oracle_gap_ci95": bootstrap["ci95"],
        "oracle_effort_distribution": {
            effort: counts[effort] / len(rows) for effort in EFFORTS
        },
        "oracle_entropy": entropy,
        "qualified_non_e2_arms": qualified_arms,
        "non_e2_arm_gate_passed": has_arm,
        "oracle_gate_passed": passes_oracle,
        "has_routing_opportunity": allowed,
        "policy_training_allowed": allowed,
        "reason": (
            "PASS" if allowed else
            "NO_QUALIFIED_NON_E2_ARM" if not has_arm else
            "ORACLE_GAP_NOT_POSITIVE"
        ),
        "per_task": oracle_rows,
        "bootstrap": bootstrap,
    }


def write_effort_frontier(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "effort_frontier.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Effort frontier", "", "| Effort | Quality | Compute | Status |",
        "|---|---:|---:|---|",
    ]
    for effort in EFFORTS:
        arm = report["arms"][effort]
        lines.append(f"| {effort} | {arm['quality']:.6f} | {arm['compute']:.6f} | {arm['frontier_status']} |")
    lines.extend(("", "Compute values come from per-task physical execution receipts."))
    (output / "EFFORT_FRONTIER.md").write_text("\n".join(lines) + "\n")
