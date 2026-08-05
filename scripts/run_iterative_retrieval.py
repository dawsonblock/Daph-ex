#!/usr/bin/env python3
"""Sprint 2: measure the marginal utility of RETRIEVE_FOLLOWUP and CALCULATE.

Fixed deterministic action policies, not a learned controller. Each arm adds
exactly one capability to the previous one, so each column of the result table
is the marginal contribution of that capability:

  A1  one_pass                 Gate B's best arm reproduced (BM25, k=10)
  A2  one_pass_selected        + entity anchoring and redundancy suppression
  A3  two_pass_selected        + bounded follow-up on the unresolved bridge
  A4  two_pass_calculate       + deterministic arithmetic when evidence states a rule

Model, revision, prompt condition, composition, decoding, and verifier are
pinned to the frozen Gate A protocol, so quality is comparable to the Gate A
and Gate B numbers by construction.
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

from hrm_adaptive_memory.actions.calculate import calculate_from_evidence
from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evidence.packing import compose_evidence_prompt
from hrm_adaptive_memory.evidence.sufficiency import SufficiencyVerdict
from hrm_adaptive_memory.experiments.context_study import OracleTask, verify_answer
from hrm_adaptive_memory.retrieval.iterative import TwoPassRetriever

ARMS = ("one_pass", "one_pass_selected", "two_pass_selected", "two_pass_calculate")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/hrm/controlled_gate_a_v2/oracle_tasks.jsonl")
    parser.add_argument("--evidence", default="data/hrm/controlled_gate_a_v2/evidence.jsonl")
    parser.add_argument("--frozen-config", default="configs/gate_a/gate_a_v2_frozen.json")
    parser.add_argument("--gate-b-verdict", default="evidence/gate_b/gate_b_verdict.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    verdict = json.loads(Path(args.gate_b_verdict).read_text())
    if not verdict.get("iterative_retrieval_authorized"):
        raise RuntimeError(
            "Bounded iterative retrieval is not authorized: Gate B verdict is "
            f"{verdict.get('verdict')!r}"
        )

    import torch
    import transformers
    from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition

    frozen = json.loads(Path(args.frozen_config).read_text())
    task_bytes = Path(args.tasks).read_bytes()
    evidence_bytes = Path(args.evidence).read_bytes()
    if frozen["task_dataset_sha256"] != _sha256(task_bytes):
        raise RuntimeError("Task corpus drifts from the frozen protocol")
    if frozen["evidence_corpus_sha256"] != _sha256(evidence_bytes):
        raise RuntimeError("Evidence corpus drifts from the frozen protocol")

    tasks = [OracleTask.from_dict(json.loads(line))
             for line in task_bytes.decode().splitlines() if line.strip()]
    evidence_rows = [json.loads(line)
                     for line in evidence_bytes.decode().splitlines() if line.strip()]

    adapter = HRMAdapter.from_pretrained(
        spec=HRMModelSpec(), dtype=torch.bfloat16, device_map=args.device_map,
    )
    condition = PromptCondition(frozen["prompt_condition"])

    def token_count(text: str) -> int:
        values = adapter.tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(values[0] if values and isinstance(values[0], list) else values)

    records = [IndexRecord(
        evidence_id=str(row["evidence_id"]), source_id=str(row["source_id"]),
        content=str(row["content"]), token_count=token_count(str(row["content"])),
        source_type=str(row.get("source_type", "source")),
        metadata=dict(row.get("metadata", {})),
    ) for row in evidence_rows]
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, records)

    arms = [value for value in args.arms.split(",") if value]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}

    for arm in arms:
        retriever = TwoPassRetriever(
            backend, k=args.k, followup_k=args.k,
            max_passes=1 if arm == "one_pass" else (1 if arm == "one_pass_selected" else 2),
            enforce_anchoring=arm != "one_pass",
        )
        rows = []
        started = time.perf_counter()
        for task in tasks:
            result = asyncio.run(retriever.retrieve(
                task.question, select=arm != "one_pass",
            ))
            calculation = None
            answered_by = "model"
            if arm == "two_pass_calculate" and result.report.verdict == SufficiencyVerdict.NEEDS_CALCULATION:
                calculation = calculate_from_evidence(result.records)
            if calculation is not None and calculation.verified:
                text = calculation.result
                answered_by = "calculator"
                prompt_tokens = completion_tokens = 0
                latency_ms = 0.0
            else:
                prompt = compose_evidence_prompt(
                    task.question, [row.content for row in result.records],
                )
                generation_started = time.perf_counter()
                generated = adapter.generate(
                    prompt, condition=condition, max_new_tokens=frozen["max_new_tokens"],
                )
                latency_ms = (time.perf_counter() - generation_started) * 1000
                text = str(generated["text"])
                prompt_tokens = int(generated["prompt_tokens"])
                completion_tokens = int(generated["completion_tokens"])
            quality, exact = verify_answer(task, text)
            required = set(task.required_evidence_ids)
            rows.append({
                "task_id": task.task_id, "family": task.family, "arm": arm,
                "answered_by": answered_by,
                "selected_ids": list(result.receipt.selected_ids),
                "complete_set_success": float(required <= set(result.receipt.selected_ids)),
                "sufficiency_verdict": result.report.verdict.value,
                "followup_query": result.receipt.followup_query,
                "retrieval_calls": result.receipt.retrieval_calls,
                "evidence_records": len(result.records),
                "evidence_tokens": sum(row.token_count for row in result.records),
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "output": text, "gold_answer": task.answer,
                "verified_quality": quality, "exact_match": exact,
                "latency_ms": latency_ms,
                "retrieval_latency_ms": result.receipt.latency_ms,
                "calculation": None if calculation is None else calculation.to_dict(),
                "model_id": adapter.spec.model_id, "model_revision": adapter.spec.revision,
                "scientific_eligible": True,
            })

        (output / f"{arm}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        by_family: dict[str, list[dict]] = {}
        for row in rows:
            by_family.setdefault(row["family"], []).append(row)
        reports[arm] = {
            "arm": arm,
            "task_count": len(rows),
            "mean_quality": round(sum(r["verified_quality"] for r in rows) / len(rows), 4),
            "complete_set_success": round(sum(r["complete_set_success"] for r in rows) / len(rows), 4),
            "mean_retrieval_calls": round(sum(r["retrieval_calls"] for r in rows) / len(rows), 3),
            "mean_evidence_records": round(sum(r["evidence_records"] for r in rows) / len(rows), 2),
            "mean_evidence_tokens": round(sum(r["evidence_tokens"] for r in rows) / len(rows), 1),
            "followups_fired": sum(1 for r in rows if r["followup_query"]),
            "calculator_answers": sum(1 for r in rows if r["answered_by"] == "calculator"),
            "wall_seconds": round(time.perf_counter() - started, 1),
            "per_family": {
                family: {
                    "quality": round(sum(r["verified_quality"] for r in items) / len(items), 4),
                    "complete_set_success": round(
                        sum(r["complete_set_success"] for r in items) / len(items), 4),
                    "followups_fired": sum(1 for r in items if r["followup_query"]),
                }
                for family, items in sorted(by_family.items())
            },
        }
        print(f"[{arm}] quality={reports[arm]['mean_quality']} "
              f"css={reports[arm]['complete_set_success']} "
              f"followups={reports[arm]['followups_fired']} "
              f"calc={reports[arm]['calculator_answers']}")

    marginal = {}
    ordered = [arm for arm in ARMS if arm in reports]
    for previous, current in zip(ordered, ordered[1:]):
        marginal[f"{current} - {previous}"] = {
            "delta_quality": round(
                reports[current]["mean_quality"] - reports[previous]["mean_quality"], 4),
            "delta_complete_set_success": round(
                reports[current]["complete_set_success"] - reports[previous]["complete_set_success"], 4),
            "delta_retrieval_calls": round(
                reports[current]["mean_retrieval_calls"] - reports[previous]["mean_retrieval_calls"], 3),
            "per_family_delta_quality": {
                family: round(
                    reports[current]["per_family"][family]["quality"]
                    - reports[previous]["per_family"][family]["quality"], 4)
                for family in reports[current]["per_family"]
            },
        }

    manifest = {
        "sprint": "2_bounded_iterative_retrieval",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authorized_by": {"gate": "B", "verdict": verdict["verdict"]},
        "task_count": len(tasks), "retrieval_k": args.k,
        "retriever": backend.backend_id,
        "reformulator": "deterministic-bridge-v1",
        "max_retrieval_depth": 2,
        "model_id": adapter.spec.model_id, "model_revision": adapter.spec.revision,
        "prompt_condition": frozen["prompt_condition"],
        "task_dataset_sha256": _sha256(task_bytes),
        "evidence_corpus_sha256": _sha256(evidence_bytes),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
        },
        "arms": reports,
        "marginal_utility": marginal,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps(marginal, indent=2))


if __name__ == "__main__":
    main()
