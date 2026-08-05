#!/usr/bin/env python3
"""Audit the controlled Gate A corpus for duplicates and answer leakage.

Section-4 audit: regenerates the corpus from its manifest seed, verifies the
committed JSONL digests, audits answer/entity duplication, and constructs the
B1/B1b control arms for every task to prove the leakage rate is zero after
context construction.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import LocalControlBackend, LocalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.experiments.context_study import (
    ContextConstructor,
    ContextStudyConfig,
    EvidenceCorpus,
    ExperimentTier,
    OracleTask,
    StudyCondition,
    _normalize,
)
from hrm_adaptive_memory.experiments.controlled_dataset import build_controlled_gate_a_corpus


def _terms(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", _normalize(text)))


def _contains_sequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    width = len(needle)
    return bool(width) and any(
        haystack[index:index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def _audit(dataset_dir: Path, output_path: Path, *, seed: int) -> dict:
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    task_rows = _load_jsonl(dataset_dir / "oracle_tasks.jsonl")
    evidence_rows = _load_jsonl(dataset_dir / "evidence.jsonl")

    regenerated = build_controlled_gate_a_corpus(
        seed=manifest["generation_seed"], tasks_per_family=manifest["tasks_per_family"],
    )
    generator_match = (
        regenerated.manifest["task_sha256"] == manifest["task_sha256"]
        and regenerated.manifest["evidence_sha256"] == manifest["evidence_sha256"]
    )

    tasks = [OracleTask.from_dict(row) for row in task_rows]
    records = [IndexRecord(
        evidence_id=str(row["evidence_id"]),
        source_id=str(row["source_id"]),
        content=str(row["content"]),
        token_count=max(1, len(str(row["content"]).split())),
        source_type=str(row.get("source_type", "source")),
        metadata=dict(row.get("metadata", {})),
    ) for row in evidence_rows]
    record_by_id = {row.evidence_id: row for row in records}

    families = Counter(task.family for task in tasks)
    answer_counts = Counter(task.answer for task in tasks)
    duplicate_answers = {value: count for value, count in answer_counts.items() if count > 1}

    entity_pattern = re.compile(r"(?:Project|Trial|Plan|Service|Station|Adapter)-[\w-]+")
    entity_counts = Counter(
        entity for task in tasks for entity in entity_pattern.findall(task.question)
    )
    duplicate_entities = {value: count for value, count in entity_counts.items() if count > 1}
    evidence_id_counts = Counter(row.evidence_id for row in records)
    duplicate_evidence_ids = {v: c for v, c in evidence_id_counts.items() if c > 1}

    # Pre-filter collisions: the gold answer token-sequence appearing in any
    # non-oracle evidence chunk.  Nonzero is expected (it is why the B1 filter
    # exists); the post-construction leak counts below must still be zero.
    prefilter_collision_tasks = 0
    answer_in_own_question = 0
    for task in tasks:
        answer_terms = _terms(task.answer)
        if _contains_sequence(_terms(task.question), answer_terms):
            answer_in_own_question += 1
        oracle = set(task.oracle_evidence_ids) | set(task.required_evidence_ids)
        for row in records:
            if row.evidence_id in oracle:
                continue
            if _contains_sequence(_terms(row.content), answer_terms):
                prefilter_collision_tasks += 1
                break

    constructor = ContextConstructor(
        EvidenceCorpus(records),
        LocalControlBackend(LocalRetrievalMode.BM25, records),
        config=ContextStudyConfig(tier=ExperimentTier.QUALIFICATION, seed=seed),
    )

    b1_answer_leaks = 0
    b1b_answer_leaks = 0
    required_evidence_leaks = 0
    token_mismatches = 0
    construction_failures: list[dict] = []
    for task in tasks:
        answer_terms = _terms(task.answer)
        oracle = set(task.oracle_evidence_ids) | set(task.required_evidence_ids)
        try:
            b3 = await constructor.construct(task, StudyCondition.B3_ORACLE_EVIDENCE)
            b1 = await constructor.construct(task, StudyCondition.B1_RANDOM_CONTEXT)
            b1b = await constructor.construct(task, StudyCondition.B1_HARD_DISTRACTOR)
        except ValueError as error:
            construction_failures.append({"task_id": task.task_id, "error": str(error)})
            continue
        for arm, suffix, leak_counter in ((b1, "#b1:", "b1"), (b1b, "#b1b:", "b1b")):
            evidence_terms = _terms(" \n ".join(row.content for row in arm.evidence))
            if _contains_sequence(evidence_terms, answer_terms):
                if leak_counter == "b1":
                    b1_answer_leaks += 1
                else:
                    b1b_answer_leaks += 1
            origins = {row.evidence_id.split(suffix, 1)[0] for row in arm.evidence}
            if origins & oracle:
                required_evidence_leaks += 1
            if arm.evidence_tokens != b3.evidence_tokens:
                token_mismatches += 1

    report = {
        "report_type": "controlled_corpus_audit",
        "dataset_dir": str(dataset_dir),
        "dataset_id": manifest["dataset_id"],
        "generation_seed": manifest["generation_seed"],
        "construction_seed": seed,
        "generator_reproduces_committed_digests": generator_match,
        "task_sha256": manifest["task_sha256"],
        "evidence_sha256": manifest["evidence_sha256"],
        "task_count": len(tasks),
        "evidence_count": len(records),
        "families": dict(sorted(families.items())),
        "unique_answers": len(answer_counts),
        "duplicate_answer_count": sum(duplicate_answers.values()) - len(duplicate_answers)
        if duplicate_answers else 0,
        "duplicate_answer_values": dict(sorted(duplicate_answers.items())),
        "duplicate_entity_count": len(duplicate_entities),
        "duplicate_evidence_id_count": len(duplicate_evidence_ids),
        "answer_in_own_question_count": answer_in_own_question,
        "prefilter_answer_collision_task_count": prefilter_collision_tasks,
        "b1_answer_leak_count": b1_answer_leaks,
        "b1b_answer_leak_count": b1b_answer_leaks,
        "required_evidence_leak_count": required_evidence_leaks,
        "b1_b3_token_mismatch_count": token_mismatches,
        "construction_failure_count": len(construction_failures),
        "construction_failures": construction_failures,
    }
    report["leakage_zero_after_construction"] = (
        b1_answer_leaks == 0
        and b1b_answer_leaks == 0
        and required_evidence_leaks == 0
        and token_mismatches == 0
        and not construction_failures
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/hrm/controlled_gate_a_v1")
    parser.add_argument("--output", default="evidence/controlled_corpus_audit_v1.json")
    parser.add_argument("--seed", type=int, default=42, help="Context-construction seed")
    args = parser.parse_args()
    report = asyncio.run(_audit(Path(args.dataset), Path(args.output), seed=args.seed))
    print(json.dumps(report, indent=2))
    if not report["leakage_zero_after_construction"]:
        raise SystemExit("AUDIT FAILED: leakage or construction failures detected")


if __name__ == "__main__":
    main()
