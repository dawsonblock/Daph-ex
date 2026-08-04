"""Receipt-backed Gate A qualification for oracle evidence use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..experiments.context_study import ExperimentTier, OracleTask, StudyCondition, TIER_REQUIREMENTS
from .bootstrap import grouped_bootstrap


@dataclass(frozen=True)
class GateAConfig:
    tier: ExperimentTier = ExperimentTier.QUALIFICATION
    minimum_mean_quality_gain: float = 0.05
    bootstrap_samples: int = 10_000
    confidence: float = 0.95
    seed: int = 42
    group_key: str = "template_id"


def _index_receipts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[StudyCondition, Mapping[str, Any]]]:
    indexed: dict[str, dict[StudyCondition, Mapping[str, Any]]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        condition = StudyCondition(row["condition"])
        if condition in indexed.setdefault(task_id, {}):
            raise ValueError(f"Duplicate receipt for {task_id}/{condition.value}")
        indexed[task_id][condition] = row
    return indexed


def qualify_gate_a(
    receipts: Sequence[Mapping[str, Any]], oracle_tasks: Sequence[OracleTask],
    config: GateAConfig = GateAConfig(),
) -> dict[str, Any]:
    if not receipts or not oracle_tasks:
        raise ValueError("Gate A requires receipts and independent oracle tasks")
    tasks = {task.task_id: task for task in oracle_tasks}
    indexed = _index_receipts(receipts)
    if set(indexed) != set(tasks):
        raise ValueError("Receipt and oracle task IDs differ")
    expected_conditions = set(StudyCondition)
    paired: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        arms = indexed[task_id]
        if set(arms) != expected_conditions:
            raise ValueError(f"Task {task_id} does not have all B0/B1/B2/B3 receipts")
        if not all(bool(row.get("scientific_eligible")) for row in arms.values()):
            raise ValueError("Engineering-fake model outputs cannot qualify Gate A")
        b0, b1 = arms[StudyCondition.B0_NO_CONTEXT], arms[StudyCondition.B1_RANDOM_CONTEXT]
        b2, b3 = arms[StudyCondition.B2_NAIVE_RETRIEVAL], arms[StudyCondition.B3_ORACLE_EVIDENCE]
        if b0["final_prompt_sha256"] == b3["final_prompt_sha256"]:
            raise ValueError("B0 and B3 prompt digests are identical")
        if tuple(b3["evidence_ids"]) != task.oracle_evidence_ids:
            raise ValueError("B3 evidence does not match the independent oracle label")
        b1_origins = {str(value).split("#b1:", 1)[0] for value in b1["evidence_ids"]}
        if b1_origins & (set(task.required_evidence_ids) | set(task.oracle_evidence_ids)):
            raise ValueError("B1 contains required/oracle evidence")
        if int(b1["evidence_tokens"]) != int(b3["evidence_tokens"]):
            raise ValueError("B1 is not token-matched to B3")
        identity = {
            (str(row["model_id"]), str(row["model_revision"]), str(row["corpus_digest"]))
            for row in arms.values()
        }
        if len(identity) != 1:
            raise ValueError("Paired arms must use one model revision and corpus")
        paired.append({
            "task_id": task_id,
            "template_id": task.template_id,
            "family": task.family,
            "quality_b0": float(b0["verified_quality"]),
            "quality_b1": float(b1["verified_quality"]),
            "quality_b2": float(b2["verified_quality"]),
            "quality_b3": float(b3["verified_quality"]),
            "utility_b0": float(b0["verified_utility"]),
            "utility_b3": float(b3["verified_utility"]),
            "delta_quality": float(b3["verified_quality"]) - float(b0["verified_quality"]),
            "delta_utility": float(b3["verified_utility"]) - float(b0["verified_utility"]),
            "random_context_delta": float(b1["verified_quality"]) - float(b0["verified_quality"]),
            "retrieval_delta": float(b2["verified_quality"]) - float(b0["verified_quality"]),
        })
    requirement = TIER_REQUIREMENTS[config.tier]
    unique_tasks = len(paired)
    groups = len({str(row[config.group_key]) for row in paired})
    powered = unique_tasks >= int(requirement["tasks"]) and groups >= int(requirement["groups"])
    quality = grouped_bootstrap(
        paired,
        value_key="delta_quality",
        group_key=config.group_key,
        samples=config.bootstrap_samples,
        confidence=config.confidence,
        seed=config.seed,
    )
    utility = grouped_bootstrap(
        paired,
        value_key="delta_utility",
        group_key=config.group_key,
        samples=config.bootstrap_samples,
        confidence=config.confidence,
        seed=config.seed + 1,
    )
    local_signal = quality["mean"] >= config.minimum_mean_quality_gain and quality["lcb95"] > 0
    passed = bool(powered and requirement["promotable"] and local_signal)
    if not powered or not requirement["promotable"]:
        status = "INSUFFICIENT_POWER"
    elif passed:
        status = "PASS_HRM_CAN_USE_ORACLE_EVIDENCE"
    else:
        status = "FAIL_HRM_EVIDENCE_USE"
    return {
        "gate": "A_HRM_CAN_USE_MEMORY",
        "status": status,
        "passed": passed,
        "tier": config.tier.value,
        "claim_strength": "STATISTICALLY_QUALIFIED" if passed else (
            "MECHANISM_SIGNAL" if local_signal else "ENGINEERING_PASS"
        ),
        "quality_bootstrap": quality,
        "utility_bootstrap_descriptive": utility,
        "minimum_mean_quality_gain": config.minimum_mean_quality_gain,
        "observed_tasks": unique_tasks,
        "observed_groups": groups,
        "requirements": requirement,
        "sufficiently_powered": powered,
        "retrieval_expansion_allowed": passed,
        "graphiti_integration_allowed": False,
        "controller_training_allowed": False,
        "next_stage": "SIMPLE_RETRIEVAL_CONTROLS" if passed else "EVIDENCE_USE_ADAPTATION",
        "paired_records": paired,
    }
