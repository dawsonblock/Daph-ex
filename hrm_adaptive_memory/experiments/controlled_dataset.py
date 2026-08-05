"""Deterministic, synthetic-but-structured oracle corpus for Gate A.

This is a controlled evidence-use benchmark, not a natural long-memory claim.
Every answer is generated after the model checkpoint's release and requires one
or more separately-addressable evidence records. The generator is retained so
the committed corpus can be independently rebuilt from its manifest.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping


GENERATOR_VERSION = "controlled-gate-a-v2"
FAMILIES = (
    "single_hop",
    "two_hop",
    "numeric_derivation",
    "temporal_update",
    "distractor_heavy",
)


@dataclass(frozen=True)
class ControlledCorpus:
    tasks: tuple[Mapping[str, Any], ...]
    evidence: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]


def _answer_leaks_into_question(question: str, answer: str) -> bool:
    """True when the normalized answer token sequence appears in the question.

    Entity names carry random numeric suffixes (e.g. Service-043-587), so an
    unguarded generator can hand B0 the answer inside the question itself.
    """

    question_terms = tuple(re.findall(r"\w+", question.lower()))
    answer_terms = tuple(re.findall(r"\w+", answer.lower()))
    width = len(answer_terms)
    return bool(width) and any(
        question_terms[index:index + width] == answer_terms
        for index in range(len(question_terms) - width + 1)
    )


def _digest(rows: tuple[Mapping[str, Any], ...]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def _row(
    *, evidence_id: str, source_id: str, content: str, family: str,
    source_cluster_id: str, kind: str,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "source_id": source_id,
        "source_type": "controlled_synthetic_source",
        "content": content,
        "metadata": {
            "generator_version": GENERATOR_VERSION,
            "family": family,
            "source_cluster_id": source_cluster_id,
            "record_kind": kind,
        },
    }


def _single_hop(
    rng: random.Random, task_id: str, family: str, cluster: str, ordinal: int,
) -> tuple[str, str, list[str], list[dict[str, Any]]]:
    project = f"Project-{ordinal:03d}-{rng.randrange(100, 999)}"
    answer = str(rng.randrange(10_000, 99_999))
    required = f"{task_id}/activation"
    source = f"{cluster}/registry"
    evidence = [_row(
        evidence_id=required,
        source_id=source,
        family=family,
        source_cluster_id=cluster,
        kind="required",
        content=f"The activation registry assigns {project} the access code {answer}.",
    )]
    question = f"What access code is assigned to {project}?"
    return question, answer, [required], evidence


def _two_hop(
    rng: random.Random, task_id: str, family: str, cluster: str, ordinal: int,
) -> tuple[str, str, list[str], list[dict[str, Any]]]:
    trial = f"Trial-{ordinal:03d}-{rng.randrange(100, 999)}"
    adapter = f"Adapter-{rng.randrange(10_000, 99_999)}"
    answer = str(rng.randrange(100, 999))
    first, second = f"{task_id}/assignment", f"{task_id}/category"
    evidence = [
        _row(
            evidence_id=first,
            source_id=f"{cluster}/deployment",
            family=family,
            source_cluster_id=cluster,
            kind="required",
            content=f"The deployment record for {trial} lists {adapter} as its adapter.",
        ),
        _row(
            evidence_id=second,
            source_id=f"{cluster}/classification",
            family=family,
            source_cluster_id=cluster,
            kind="required",
            content=f"The classification registry maps {adapter} to category code {answer}.",
        ),
    ]
    question = f"Which category code applies to {trial}?"
    return question, answer, [first, second], evidence


def _numeric_derivation(
    rng: random.Random, task_id: str, family: str, cluster: str, ordinal: int,
) -> tuple[str, str, list[str], list[dict[str, Any]]]:
    plan = f"Plan-{ordinal:03d}-{rng.randrange(100, 999)}"
    units, multiplier = rng.randrange(11, 89), rng.randrange(2, 9)
    answer = str(units * multiplier)
    first, second = f"{task_id}/units", f"{task_id}/multiplier"
    evidence = [
        _row(
            evidence_id=first,
            source_id=f"{cluster}/ledger",
            family=family,
            source_cluster_id=cluster,
            kind="required",
            content=f"The base ledger records {units} units for {plan}.",
        ),
        _row(
            evidence_id=second,
            source_id=f"{cluster}/rules",
            family=family,
            source_cluster_id=cluster,
            kind="required",
            content=f"The operating rule for {plan} multiplies its units by {multiplier}.",
        ),
    ]
    question = f"What output count results for {plan} after applying its operating rule?"
    return question, answer, [first, second], evidence


def _temporal_update(
    rng: random.Random, task_id: str, family: str, cluster: str, ordinal: int,
) -> tuple[str, str, list[str], list[dict[str, Any]]]:
    service = f"Service-{ordinal:03d}-{rng.randrange(100, 999)}"
    old_value, answer = rng.randrange(100, 499), str(rng.randrange(500, 999))
    current, stale = f"{task_id}/current", f"{task_id}/superseded"
    evidence = [
        _row(
            evidence_id=current,
            source_id=f"{cluster}/release-current",
            family=family,
            source_cluster_id=cluster,
            kind="required_current",
            content=f"Release 9 supersedes all earlier settings: {service} now uses setting {answer}.",
        ),
        _row(
            evidence_id=stale,
            source_id=f"{cluster}/release-old",
            family=family,
            source_cluster_id=cluster,
            kind="superseded",
            content=f"Release 8 recorded setting {old_value} for {service}; this setting was superseded.",
        ),
    ]
    question = f"According to Release 9, what setting does {service} use?"
    return question, answer, [current], evidence


def _distractor_heavy(
    rng: random.Random, task_id: str, family: str, cluster: str, ordinal: int,
) -> tuple[str, str, list[str], list[dict[str, Any]]]:
    station = f"Station-{ordinal:03d}-{rng.randrange(100, 999)}"
    answer = str(rng.randrange(1_000, 9_999))
    required = f"{task_id}/assignment"
    evidence = [_row(
        evidence_id=required,
        source_id=f"{cluster}/committee",
        family=family,
        source_cluster_id=cluster,
        kind="required",
        content=f"The allocation committee assigned {station} the routing code {answer}.",
    )]
    for decoy_index in range(4):
        decoy = str(rng.randrange(1_000, 9_999))
        while decoy == answer:
            decoy = str(rng.randrange(1_000, 9_999))
        evidence.append(_row(
            evidence_id=f"{task_id}/distractor-{decoy_index}",
            source_id=f"{cluster}/committee-decoy-{decoy_index}",
            family=family,
            source_cluster_id=cluster,
            kind="hard_distractor",
            content=(
                f"The allocation committee discussed {station} alongside a different routing code "
                f"{decoy}; that proposal was not adopted."
            ),
        ))
    question = f"Which routing code was assigned to {station}?"
    return question, answer, [required], evidence


_BUILDERS: dict[str, Callable[[random.Random, str, str, str, int], tuple[str, str, list[str], list[dict[str, Any]]]]] = {
    "single_hop": _single_hop,
    "two_hop": _two_hop,
    "numeric_derivation": _numeric_derivation,
    "temporal_update": _temporal_update,
    "distractor_heavy": _distractor_heavy,
}


def build_controlled_gate_a_corpus(
    *, seed: int = 3601, tasks_per_family: int = 100,
) -> ControlledCorpus:
    """Return a reproducible 500-task default capability-use corpus."""

    if tasks_per_family < 1:
        raise ValueError("tasks_per_family must be positive")
    rng = random.Random(seed)
    tasks: list[Mapping[str, Any]] = []
    evidence: list[Mapping[str, Any]] = []
    for family in FAMILIES:
        builder = _BUILDERS[family]
        for ordinal in range(tasks_per_family):
            task_id = f"{family}-{ordinal:03d}"
            template_id = f"{family}-template-{ordinal % 3}"
            source_cluster_id = f"{family}-source-cluster-{ordinal % 10}"
            for _attempt in range(20):
                question, answer, required, rows = builder(
                    rng, task_id, family, source_cluster_id, ordinal,
                )
                if not _answer_leaks_into_question(question, answer):
                    break
            else:
                raise RuntimeError(
                    f"Could not build an answer-free question for {task_id} in 20 attempts"
                )
            tasks.append({
                "task_id": task_id,
                "question": question,
                "answer": answer,
                "required_evidence_ids": required,
                "oracle_evidence_ids": required,
                "family": family,
                "template_id": template_id,
                "source_cluster_id": source_cluster_id,
                "split": "qualification",
                "verifier": "numeric",
                "metadata": {
                    "generator_version": GENERATOR_VERSION,
                    "generation_seed": seed,
                    "difficulty": "controlled_evidence_use",
                    "oracle_label_origin": "deterministic_generator",
                },
            })
            evidence.extend(rows)
    frozen_tasks, frozen_evidence = tuple(tasks), tuple(evidence)
    manifest = {
        "dataset_id": GENERATOR_VERSION,
        "generation_seed": seed,
        "tasks_per_family": tasks_per_family,
        "task_count": len(frozen_tasks),
        "evidence_count": len(frozen_evidence),
        "families": list(FAMILIES),
        "task_sha256": _digest(frozen_tasks),
        "evidence_sha256": _digest(frozen_evidence),
        "claim_strength": "CONTROLLED_SYNTHETIC_BENCHMARK_ONLY",
        "natural_memory_claim_allowed": False,
    }
    return ControlledCorpus(frozen_tasks, frozen_evidence, manifest)
