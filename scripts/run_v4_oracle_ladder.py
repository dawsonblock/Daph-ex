#!/usr/bin/env python3
"""Run the R0-R5 oracle ladder on controlled_gate_a_v4.

The mechanism is pinned and unchanged: no ENTITY_PATTERN, packing, query
formulation, prompt, decoding, or verifier change is permitted in this sprint.
Only the oracle arms differ, and they read latent identity from each task's
proof graph rather than from the extractor under test.

Batching is opt-in and gated: pass --batch-size > 1 only after
tests/unit/test_batched_generation.py's real-checkpoint equivalence test has
passed on this machine, and the manifest records which path produced the run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evidence.packing import compose_evidence_prompt, select_evidence
from hrm_adaptive_memory.evidence.state import EvidenceRecordView, build_evidence_state
from hrm_adaptive_memory.experiments.context_study import OracleTask, verify_answer
from hrm_adaptive_memory.experiments.oracle_ladder import (
    LADDER_ORDER,
    LadderArm,
    decompose,
    oracle_bridge_query,
    read_oracle_facts,
)
from hrm_adaptive_memory.retrieval.iterative import TwoPassRetriever


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Row:
    """Minimal record shim so IndexRecords can become EvidenceRecordViews."""

    def __init__(self, record, rank):
        self.evidence_id = record.evidence_id
        self.source_id = record.source_id
        self.content = record.content
        self.token_count = record.token_count
        self.rank = rank


def _views(index, ids):
    return [EvidenceRecordView.from_retrieved(_Row(index[v], rank))
            for rank, v in enumerate(ids, 1) if v in index]


def build_context(arm, task, facts, backend, retriever, index, k):
    """Return (records, oracle_query, retrieval_calls) for one arm."""

    if arm == LadderArm.R5_ORACLE_EVIDENCE:
        return _views(index, task.oracle_evidence_ids), None, 0

    if arm == LadderArm.R0_ONE_PASS:
        result = asyncio.run(retriever.retrieve(task.question, select=False))
        return list(result.records), None, result.receipt.retrieval_calls

    if arm == LadderArm.R1_CURRENT_TWO_PASS:
        result = asyncio.run(retriever.retrieve(task.question))
        return list(result.records), result.receipt.followup_query, result.receipt.retrieval_calls

    # Oracle arms R2-R4: first pass is real, the follow-up query is oracle.
    first = asyncio.run(backend.search(task.question, k=k))
    pool = [EvidenceRecordView.from_retrieved(row) for row in first.evidence]
    calls = 1
    query = oracle_bridge_query(
        facts, include_relation=arm != LadderArm.R2_ORACLE_BRIDGE_IDENTITY)
    if query:
        second = asyncio.run(backend.search(query, k=k))
        pool += [EvidenceRecordView.from_retrieved(row) for row in second.evidence]
        calls += 1
    unique: dict[str, EvidenceRecordView] = {}
    for row in pool:
        unique.setdefault(row.evidence_id, row)
    candidates = tuple(unique.values())

    if arm == LadderArm.R4_ORACLE_QUERY_ORACLE_SELECTION:
        # Perfect selection over whatever retrieval actually returned: keep the
        # required records that were found, and nothing else.
        required = set(task.oracle_evidence_ids)
        chosen = [row for row in candidates if row.evidence_id in required]
        return chosen, query, calls

    state = build_evidence_state(question=task.question, records=candidates)
    anchors = set(state.required_entities) | set(state.linked_entities)
    if query:
        anchors.add(facts.bridge_surface or query)
    selected, _ = select_evidence(candidates, anchor_entities=tuple(anchors))
    return list(selected), query, calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="qualification")
    parser.add_argument("--dataset-root", default="data/hrm/controlled_gate_a_v4")
    parser.add_argument("--frozen-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = all tasks")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--arms", default=",".join(a.value for a in LADDER_ORDER))
    args = parser.parse_args()

    import torch
    import transformers
    from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition

    root = Path(args.dataset_root) / args.split
    frozen_path = Path(args.frozen_config or f"configs/gate_c1_v4_{args.split}.json")
    frozen = json.loads(frozen_path.read_text())
    task_bytes = (root / "oracle_tasks.jsonl").read_bytes()
    evidence_bytes = (root / "evidence.jsonl").read_bytes()
    if frozen["task_dataset_sha256"] != _sha256(task_bytes):
        raise RuntimeError("Task corpus drifts from the frozen protocol")
    if frozen["evidence_corpus_sha256"] != _sha256(evidence_bytes):
        raise RuntimeError("Evidence corpus drifts from the frozen protocol")

    raw_tasks = [json.loads(l) for l in task_bytes.decode().splitlines() if l.strip()]
    if args.limit:
        raw_tasks = raw_tasks[:args.limit]
    facts_by_id = {row["task_id"]: read_oracle_facts(row) for row in raw_tasks}
    # _oracle_metadata is evaluator-only and must not reach OracleTask/runtime.
    tasks = [OracleTask.from_dict({k: v for k, v in row.items()
                                   if k != "_oracle_metadata"}) for row in raw_tasks]
    evidence_rows = [json.loads(l) for l in evidence_bytes.decode().splitlines() if l.strip()]

    adapter = HRMAdapter.from_pretrained(
        spec=HRMModelSpec(), dtype=torch.bfloat16, device_map=args.device_map)
    condition = PromptCondition(frozen["prompt_condition"])
    max_new = frozen["max_new_tokens"]

    def token_count(text: str) -> int:
        values = adapter.tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(values[0] if values and isinstance(values[0], list) else values)

    records = [IndexRecord(
        evidence_id=r["evidence_id"], source_id=r["source_id"], content=r["content"],
        token_count=token_count(r["content"]), source_type=r["source_type"],
        metadata=r["metadata"]) for r in evidence_rows]
    index = {r.evidence_id: r for r in records}
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, records)
    retriever = TwoPassRetriever(backend, k=args.k, followup_k=args.k)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    arms = [LadderArm(a) for a in args.arms.split(",")]
    quality_by_arm: dict[str, float] = {}
    reports: dict[str, dict] = {}

    for arm in arms:
        started = time.perf_counter()
        prepared = []
        for task in tasks:
            facts = facts_by_id[task.task_id]
            recs, query, calls = build_context(
                arm, task, facts, backend, retriever, index, args.k)
            prepared.append((task, facts, recs, query, calls,
                             compose_evidence_prompt(task.question,
                                                     [r.content for r in recs])))

        outputs: list[dict] = []
        if args.batch_size > 1:
            for start in range(0, len(prepared), args.batch_size):
                chunk = prepared[start:start + args.batch_size]
                outputs.extend(adapter.generate_batch(
                    [row[5] for row in chunk], condition=condition,
                    max_new_tokens=max_new))
                print(f"  [{arm.value}] {min(start + args.batch_size, len(prepared))}"
                      f"/{len(prepared)}", flush=True)
        else:
            for position, row in enumerate(prepared, 1):
                outputs.append(adapter.generate(
                    row[5], condition=condition, max_new_tokens=max_new))
                if position % 50 == 0:
                    print(f"  [{arm.value}] {position}/{len(prepared)}", flush=True)

        rows = []
        for (task, facts, recs, query, calls, prompt), generated in zip(prepared, outputs):
            text = str(generated["text"])
            quality, exact = verify_answer(task, text)
            required = set(task.oracle_evidence_ids)
            selected_ids = [r.evidence_id for r in recs]
            rows.append({
                "arm": arm.value, "task_id": task.task_id, "family": task.family,
                "template_id": task.template_id, "source_cluster_id": task.source_cluster_id,
                "entity_regime": next((r["metadata"]["entity_regime"] for r in raw_tasks
                                       if r["task_id"] == task.task_id), None),
                "answer_kind": next((r["metadata"]["answer_kind"] for r in raw_tasks
                                     if r["task_id"] == task.task_id), None),
                "opportunity_group": next((r["metadata"]["opportunity_group"] for r in raw_tasks
                                           if r["task_id"] == task.task_id), None),
                "selected_ids": selected_ids,
                "complete_set_success": float(required <= set(selected_ids)),
                "oracle_query": query, "retrieval_calls": calls,
                "evidence_records": len(recs),
                "output": text, "gold_answer": task.answer,
                "verified_quality": quality, "exact_match": exact,
                "prompt_tokens": int(generated["prompt_tokens"]),
                "completion_tokens": int(generated["completion_tokens"]),
                "used_oracle_metadata": arm not in (
                    LadderArm.R0_ONE_PASS, LadderArm.R1_CURRENT_TWO_PASS),
                "model_id": adapter.spec.model_id,
                "model_revision": adapter.spec.revision,
                "scientific_eligible": True,
            })
        (output / f"{arm.value}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        mean_quality = round(sum(r["verified_quality"] for r in rows) / len(rows), 4)
        quality_by_arm[arm.value] = mean_quality
        reports[arm.value] = {
            "quality": mean_quality,
            "complete_set_success": round(
                sum(r["complete_set_success"] for r in rows) / len(rows), 4),
            "mean_retrieval_calls": round(
                sum(r["retrieval_calls"] for r in rows) / len(rows), 3),
            "oracle_queries_issued": sum(1 for r in rows if r["oracle_query"]),
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(f"[{arm.value}] quality={mean_quality} "
              f"css={reports[arm.value]['complete_set_success']} "
              f"({reports[arm.value]['seconds']}s)", flush=True)

    manifest = {
        "gate": "C1_V4_ORACLE_LADDER", "split": args.split,
        "task_count": len(tasks), "retrieval_k": args.k,
        "mechanism_commit": frozen.get("mechanism_commit"),
        "mechanism_unchanged": True,
        "batch_size": args.batch_size,
        "generation_path": "batched" if args.batch_size > 1 else "sequential",
        "model_id": adapter.spec.model_id, "model_revision": adapter.spec.revision,
        "prompt_condition": frozen["prompt_condition"],
        "task_dataset_sha256": _sha256(task_bytes),
        "evidence_corpus_sha256": _sha256(evidence_bytes),
        "arms": reports,
        "decomposition": decompose(quality_by_arm),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu/mps",
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"arms": reports, "decomposition": manifest["decomposition"]}, indent=2))


if __name__ == "__main__":
    main()
