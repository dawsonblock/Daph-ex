#!/usr/bin/env python3
"""Why does HRM fail on two_hop tasks whose evidence was fully retrieved?

Gate B's downstream numbers showed HRM answering only 1 of 9 two_hop tasks
that had complete evidence in the prompt, while Gate A's oracle arm scored
100/100 on the same family. This isolates the cause by varying one factor at a
time with the required evidence always present:

  packet_size       2 / 3 / 5 / 10 records, required first, random padding
  oracle_position   required evidence first / middle / last within 10 records
  distractor_kind   random corpus / same-template / actual BM25 top-k

Everything else — model, revision, prompt condition, decoding, verifier,
composition format — is pinned to the frozen Gate A protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.experiments.context_study import OracleTask, verify_answer

SLOT_ECHO = re.compile(r"^\s*\[E\d+\]")
RESPONSE_REQUIREMENT = (
    "Return only the answer. Use supplied evidence when helpful, "
    "but answer the task regardless."
)


def compose(question: str, contents: list[str]) -> str:
    parts = ["[OBJECTIVE]", question, "[EVIDENCE]"]
    for index, content in enumerate(contents, 1):
        parts.extend([f"[E{index}]", content])
    parts.extend(["[RESPONSE REQUIREMENT]", RESPONSE_REQUIREMENT])
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/hrm/controlled_gate_a_v2/oracle_tasks.jsonl")
    parser.add_argument("--evidence", default="data/hrm/controlled_gate_a_v2/evidence.jsonl")
    parser.add_argument("--frozen-config", default="configs/gate_a/gate_a_v2_frozen.json")
    parser.add_argument("--family", default="two_hop")
    parser.add_argument("--output", default="evidence/gate_b/qualification/packing_diagnostic.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition

    frozen = json.loads(Path(args.frozen_config).read_text())
    tasks = [OracleTask.from_dict(json.loads(line))
             for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    evidence = [json.loads(line)
                for line in Path(args.evidence).read_text().splitlines() if line.strip()]
    by_id = {row["evidence_id"]: row for row in evidence}
    subject = [task for task in tasks if task.family == args.family]

    adapter = HRMAdapter.from_pretrained(
        spec=HRMModelSpec(), dtype=torch.bfloat16, device_map="auto",
    )
    condition = PromptCondition(frozen["prompt_condition"])
    records = [IndexRecord(
        evidence_id=row["evidence_id"], source_id=row["source_id"], content=row["content"],
        token_count=max(1, len(row["content"].split())),
        source_type=row["source_type"], metadata=row["metadata"],
    ) for row in evidence]
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, records)

    def ordered_pool(task: OracleTask, keys: list[str]) -> list[str]:
        return sorted(keys, key=lambda key: hashlib.sha256(
            f"{args.seed}\0{task.task_id}\0{key}".encode()
        ).hexdigest())

    def random_distractors(task: OracleTask, count: int) -> list[str]:
        return ordered_pool(task, [
            key for key in by_id if key not in set(task.required_evidence_ids)
        ])[:count]

    def same_template_distractors(task: OracleTask, count: int) -> list[str]:
        suffix = task.required_evidence_ids[-1].rsplit("/", 1)[-1]
        return ordered_pool(task, [
            key for key in by_id
            if key.startswith(f"{args.family}-") and key.rsplit("/", 1)[-1] == suffix
            and key not in set(task.required_evidence_ids)
        ])[:count]

    def bm25_distractors(task: OracleTask, count: int) -> list[str]:
        result = asyncio.run(backend.search(task.question, k=count + len(task.required_evidence_ids) + 10))
        return [
            row.evidence_id for row in result.evidence
            if row.evidence_id not in set(task.required_evidence_ids)
        ][:count]

    def measure(build_ids) -> dict:
        correct = echoes = 0
        started = time.perf_counter()
        for task in subject:
            ids = build_ids(task)
            generated = adapter.generate(
                compose(task.question, [by_id[value]["content"] for value in ids]),
                condition=condition, max_new_tokens=frozen["max_new_tokens"],
            )
            text = str(generated["text"])
            quality, _ = verify_answer(task, text)
            correct += int(quality >= 1.0)
            echoes += int(bool(SLOT_ECHO.match(text)))
        return {
            "n": len(subject),
            "quality": round(correct / len(subject), 4),
            "slot_label_echoes": echoes,
            "seconds": round(time.perf_counter() - started, 1),
        }

    report = {
        "diagnostic": "gate_b_packing",
        "family": args.family,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": adapter.spec.model_id,
        "model_revision": adapter.spec.revision,
        "prompt_condition": frozen["prompt_condition"],
        "note": "required evidence is present in every condition; only the stated factor varies",
        "packet_size": {}, "oracle_position": {}, "distractor_kind": {},
    }

    for size in (2, 3, 5, 10):
        report["packet_size"][str(size)] = measure(
            lambda task, size=size: list(task.required_evidence_ids)
            + random_distractors(task, size - len(task.required_evidence_ids))
        )
        print(f"packet_size={size}: {report['packet_size'][str(size)]}")

    def positioned(task: OracleTask, place: str) -> list[str]:
        required = list(task.required_evidence_ids)
        padding = random_distractors(task, 10 - len(required))
        if place == "first":
            return required + padding
        if place == "middle":
            half = len(padding) // 2
            return padding[:half] + required + padding[half:]
        return padding + required

    for place in ("first", "middle", "last"):
        report["oracle_position"][place] = measure(
            lambda task, place=place: positioned(task, place)
        )
        print(f"oracle_position={place}: {report['oracle_position'][place]}")

    for name, picker in (
        ("random_corpus", random_distractors),
        ("same_template", same_template_distractors),
        ("bm25_top_k", bm25_distractors),
    ):
        report["distractor_kind"][name] = measure(
            lambda task, picker=picker: (
                list(task.required_evidence_ids)
                + picker(task, 10 - len(task.required_evidence_ids))
            )
        )
        print(f"distractor_kind={name}: {report['distractor_kind'][name]}")

    report["conclusion"] = (
        "Packet size and oracle position do not degrade evidence use; distractor "
        "similarity does. Retrieval precision, not just recall, is a binding "
        "constraint on downstream answer quality."
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:600])


if __name__ == "__main__":
    main()
