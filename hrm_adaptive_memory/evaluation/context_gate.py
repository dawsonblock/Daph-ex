"""Receipt-backed Gate A qualification for oracle evidence use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..experiments.context_study import (
    EvaluationMode,
    ExperimentTier,
    OracleTask,
    PRIMARY_STUDY_CONDITIONS,
    StudyCondition,
    TIER_REQUIREMENTS,
)
from .bootstrap import grouped_bootstrap


@dataclass(frozen=True)
class GateAConfig:
    tier: ExperimentTier = ExperimentTier.QUALIFICATION
    minimum_mean_quality_gain: float = 0.05
    bootstrap_samples: int = 10_000
    confidence: float = 0.95
    seed: int = 42
    evaluation_mode: EvaluationMode = EvaluationMode.CAPABILITY_USE
    group_keys: tuple[str, ...] = ("template_id", "family", "source_cluster_id")
    require_hard_distractor: bool = False


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
    expected_conditions = set(PRIMARY_STUDY_CONDITIONS)
    hard_distractor_presence = {
        StudyCondition.B1_HARD_DISTRACTOR in arms for arms in indexed.values()
    }
    if len(hard_distractor_presence) > 1:
        raise ValueError("Hard-distractor control must be present for every task or none")
    paired: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        arms = indexed[task_id]
        unexpected = set(arms) - (expected_conditions | {StudyCondition.B1_HARD_DISTRACTOR})
        if unexpected or not expected_conditions.issubset(arms):
            raise ValueError(f"Task {task_id} does not have all primary B0/B1/B2/B3 receipts")
        if not all(bool(row.get("scientific_eligible")) for row in arms.values()):
            raise ValueError("Engineering-fake model outputs cannot qualify Gate A")
        if not all(
            row.get("evaluation_mode") == config.evaluation_mode.value for row in arms.values()
        ):
            raise ValueError("Gate A receipts mix evaluation modes or omit the declared mode")
        b0, b1 = arms[StudyCondition.B0_NO_CONTEXT], arms[StudyCondition.B1_RANDOM_CONTEXT]
        b2, b3 = arms[StudyCondition.B2_NAIVE_RETRIEVAL], arms[StudyCondition.B3_ORACLE_EVIDENCE]
        if not all(isinstance(row.get("final_prompt"), str) for row in arms.values()):
            raise ValueError("Gate A receipts must retain the model-visible final prompt")
        if b0["final_prompt_sha256"] == b3["final_prompt_sha256"]:
            raise ValueError("B0 and B3 prompt digests are identical")
        if any(
            condition.value in str(row.get("final_prompt", ""))
            for condition, row in arms.items()
        ):
            raise ValueError("Study-condition labels must not be model-visible prompt content")
        if any(
            marker in str(row["final_prompt"])
            for row in arms.values()
            for marker in ("evidence_id=", "source_id=", "#b1:", "#b1b:")
        ):
            raise ValueError("Control/provenance identifiers must not be model-visible prompt content")
        if tuple(b3["evidence_ids"]) != task.oracle_evidence_ids:
            raise ValueError("B3 evidence does not match the independent oracle label")
        b1_origins = {str(value).split("#b1:", 1)[0] for value in b1["evidence_ids"]}
        if b1_origins & (set(task.required_evidence_ids) | set(task.oracle_evidence_ids)):
            raise ValueError("B1 contains required/oracle evidence")
        if int(b1["evidence_tokens"]) != int(b3["evidence_tokens"]):
            raise ValueError("B1 is not token-matched to B3")
        b1b = arms.get(StudyCondition.B1_HARD_DISTRACTOR)
        if config.require_hard_distractor and b1b is None:
            raise ValueError("Gate A requires a hard-distractor control receipt")
        if b1b is not None:
            b1b_origins = {str(value).split("#b1b:", 1)[0] for value in b1b["evidence_ids"]}
            if b1b_origins & (set(task.required_evidence_ids) | set(task.oracle_evidence_ids)):
                raise ValueError("B1b contains required/oracle evidence")
            if int(b1b["evidence_tokens"]) != int(b3["evidence_tokens"]):
                raise ValueError("B1b is not token-matched to B3")
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
            "source_cluster_id": task.source_cluster_id,
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
    if not config.group_keys:
        raise ValueError("Gate A requires at least one declared grouping key")
    if any(key not in paired[0] for key in config.group_keys):
        raise ValueError("Gate A grouping key is not available in paired records")
    group_counts = {
        key: len({str(row[key]) for row in paired})
        for key in config.group_keys
    }
    powered = (
        unique_tasks >= int(requirement["tasks"])
        and all(count >= int(requirement["groups"]) for count in group_counts.values())
    )
    quality_by_group = {
        key: grouped_bootstrap(
            paired,
            value_key="delta_quality",
            group_key=key,
            samples=config.bootstrap_samples,
            confidence=config.confidence,
            seed=config.seed + index,
        )
        for index, key in enumerate(config.group_keys)
    }
    utility_by_group = {
        key: grouped_bootstrap(
            paired,
            value_key="delta_utility",
            group_key=key,
            samples=config.bootstrap_samples,
            confidence=config.confidence,
            seed=config.seed + len(config.group_keys) + index,
        )
        for index, key in enumerate(config.group_keys)
    }
    conservative_group_key = min(
        config.group_keys, key=lambda key: float(quality_by_group[key]["lcb95"]),
    )
    local_signal = (
        quality_by_group[conservative_group_key]["mean"] >= config.minimum_mean_quality_gain
        and all(float(result["lcb95"]) > 0 for result in quality_by_group.values())
    )
    promotable_mode = config.evaluation_mode == EvaluationMode.CAPABILITY_USE
    passed = bool(powered and requirement["promotable"] and promotable_mode and local_signal)
    if not promotable_mode:
        status = "EVIDENCE_GROUNDED_NONPROMOTABLE"
    elif not powered or not requirement["promotable"]:
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
        "evaluation_mode": config.evaluation_mode.value,
        "claim_strength": "STATISTICALLY_QUALIFIED" if passed else (
            "MECHANISM_SIGNAL" if local_signal else "ENGINEERING_PASS"
        ),
        "quality_bootstrap_by_group": quality_by_group,
        "utility_bootstrap_descriptive_by_group": utility_by_group,
        "conservative_group_key": conservative_group_key,
        "conservative_quality_bootstrap": quality_by_group[conservative_group_key],
        "conservative_utility_bootstrap_descriptive": utility_by_group[conservative_group_key],
        "minimum_mean_quality_gain": config.minimum_mean_quality_gain,
        "observed_tasks": unique_tasks,
        "observed_groups_by_key": group_counts,
        "requirements": requirement,
        "sufficiently_powered": powered,
        "retrieval_expansion_allowed": passed,
        "graphiti_integration_allowed": False,
        "controller_training_allowed": False,
        "hard_distractor_required": config.require_hard_distractor,
        "hard_distractor_present": True in hard_distractor_presence,
        "next_stage": (
            "SIMPLE_RETRIEVAL_CONTROLS" if passed
            else "GROUNDING_DIAGNOSTIC_ONLY" if not promotable_mode
            else "EVIDENCE_USE_ADAPTATION"
        ),
        "paired_records": paired,
    }
