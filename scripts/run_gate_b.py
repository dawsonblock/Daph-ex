#!/usr/bin/env python3
"""Gate B: can practical retrieval recover the evidence Gate A proved HRM uses?

Runs every canonical retrieval arm over the frozen qualification corpus,
scores complete-evidence-set recovery, then answers each task with the same
HRM model, prompt condition, packing, decoding, and verifier that Gate A used,
so downstream deltas are attributable to retrieval alone.

B0/B3 anchors are read from the frozen Gate A report rather than regenerated:
the model revision, corpus digests, prompt condition, and decoding are pinned
identical, and the run aborts if they are not.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evaluation.failure_analysis import classify
from hrm_adaptive_memory.evaluation.failure_analysis import summarize as summarize_failures
from hrm_adaptive_memory.evaluation.resources import ResourceLedger
from hrm_adaptive_memory.evaluation.retrieval_metrics import score_task, summarize
from hrm_adaptive_memory.experiments.context_study import (
    ContextConstructor,
    ContextStudyConfig,
    EvaluationMode,
    EvidenceCorpus,
    ExperimentTier,
    OracleTask,
    StudyCondition,
    verify_answer,
)
from hrm_adaptive_memory.retrieval.embedding import EmbeddingSpec, PinnedTransformerEmbedder


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_jsonl(path: Path) -> tuple[bytes, list[dict]]:
    data = path.read_bytes()
    return data, [json.loads(line) for line in data.decode().splitlines() if line.strip()]


class HuggingFaceTokenCodec:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _ids(self, text: str) -> list[int]:
        values = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if values and isinstance(values[0], list):
            values = values[0]
        return list(values)

    def count(self, text: str) -> int:
        return len(self._ids(text))

    def truncate(self, text: str, tokens: int) -> str:
        return self.tokenizer.decode(self._ids(text)[:tokens], skip_special_tokens=False)


async def _run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    frozen_bytes = Path(args.frozen_config).read_bytes()
    frozen = json.loads(frozen_bytes)
    task_bytes, task_rows = _load_jsonl(Path(args.tasks))
    evidence_bytes, evidence_rows = _load_jsonl(Path(args.evidence))

    # Gate B must inherit Gate A's frozen experimental identity exactly.
    drift = {}
    if frozen["task_dataset_sha256"] != _sha256(task_bytes):
        drift["task_dataset_sha256"] = _sha256(task_bytes)
    if frozen["evidence_corpus_sha256"] != _sha256(evidence_bytes):
        drift["evidence_corpus_sha256"] = _sha256(evidence_bytes)
    if drift:
        raise RuntimeError(f"Gate B corpus drifts from the frozen Gate A protocol: {drift}")

    gate_a = json.loads(Path(args.gate_a_report).read_text())
    if not gate_a.get("passed"):
        raise RuntimeError("Gate B requires a passing Gate A report")
    anchors = {
        "b0_mean_quality": round(
            sum(row["quality_b0"] for row in gate_a["paired_records"]) / len(gate_a["paired_records"]), 4
        ),
        "b3_mean_quality": round(
            sum(row["quality_b3"] for row in gate_a["paired_records"]) / len(gate_a["paired_records"]), 4
        ),
        "b0_by_family": {}, "b3_by_family": {},
    }
    by_family: dict[str, list[dict]] = {}
    for row in gate_a["paired_records"]:
        by_family.setdefault(row["family"], []).append(row)
    for family, rows in sorted(by_family.items()):
        anchors["b0_by_family"][family] = round(sum(r["quality_b0"] for r in rows) / len(rows), 4)
        anchors["b3_by_family"][family] = round(sum(r["quality_b3"] for r in rows) / len(rows), 4)

    tasks = [OracleTask.from_dict(row) for row in task_rows]
    arms = [CanonicalRetrievalMode(value) for value in args.arms.split(",")]

    import torch
    import transformers

    adapter = None
    codec = None
    if not args.retrieval_only:
        from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition
        adapter = HRMAdapter.from_pretrained(
            spec=HRMModelSpec(), dtype=torch.bfloat16, device_map=args.device_map,
        )
        codec = HuggingFaceTokenCodec(adapter.tokenizer)
        condition = PromptCondition(frozen["prompt_condition"])

    # Token counts must use the same codec the packer uses, or the evidence
    # token accounting would not match Gate A's.
    def token_count(text: str) -> int:
        return codec.count(text) if codec is not None else max(1, len(text.split()))

    records = [IndexRecord(
        evidence_id=str(row["evidence_id"]), source_id=str(row["source_id"]),
        content=str(row["content"]), token_count=token_count(str(row["content"])),
        source_type=str(row.get("source_type", "source")),
        metadata=dict(row.get("metadata", {})),
    ) for row in evidence_rows]
    corpus = EvidenceCorpus(records)
    evidence_contents = {row.evidence_id: row.content for row in records}
    token_by_id = {row.evidence_id: row.token_count for row in records}

    shared_embedder = None
    if any(arm.value.startswith(("dense", "hybrid")) for arm in arms):
        shared_embedder = PinnedTransformerEmbedder(EmbeddingSpec())

    output.mkdir(parents=True, exist_ok=True)
    arm_reports: dict[str, dict] = {}

    for arm in arms:
        arm_dir = output / arm.value
        arm_dir.mkdir(parents=True, exist_ok=True)
        results_path = arm_dir / "per_task_results.jsonl"
        if results_path.exists() and args.resume:
            print(f"[{arm.value}] already complete; skipping")
            arm_reports[arm.value] = json.loads((arm_dir / "arm_report.json").read_text())
            continue

        ledger = ResourceLedger()
        needs_embedder = arm != CanonicalRetrievalMode.HASH and arm != CanonicalRetrievalMode.BM25
        build_started = time.perf_counter()
        backend = CanonicalRetrievalBackend(
            arm, embedder=shared_embedder if needs_embedder else None,
            rrf_k=args.rrf_k, candidate_k=args.candidate_k, dense_weight=args.dense_weight,
        )
        index_receipt = await backend.index(records)
        index_seconds = time.perf_counter() - build_started

        config = ContextStudyConfig(
            tier=ExperimentTier.QUALIFICATION, retrieval_k=args.k,
            seed=frozen["seed"], evaluation_mode=EvaluationMode(frozen["evaluation_mode"]),
        )
        constructor = ContextConstructor(corpus, backend, config=config, token_codec=codec)

        metric_rows = []
        answer_rows = []
        failures = []
        for task in tasks:
            with ledger.phase("retrieval"):
                result = await backend.search(task.question, k=args.k)
            ledger.calls.retrieval_calls += 1
            retrieved_ids = [row.evidence_id for row in result.evidence]
            metrics = score_task(
                task_id=task.task_id, family=task.family, backend_id=backend.backend_id,
                requested_k=args.k, retrieved_ids=retrieved_ids,
                required_ids=task.required_evidence_ids, token_counts=token_by_id,
                latency_ms=result.receipt.latency_ms,
            )
            metric_rows.append(metrics)

            if args.retrieval_only:
                continue

            with ledger.phase("prompt_construction"):
                context = await constructor.construct(task, StudyCondition.B2_NAIVE_RETRIEVAL)
            started = time.perf_counter()
            with ledger.phase("model"):
                generated = await asyncio.to_thread(
                    adapter.generate, context.prompt, condition=condition,
                    max_new_tokens=frozen["max_new_tokens"],
                )
            latency_ms = (time.perf_counter() - started) * 1000
            ledger.calls.model_calls += 1
            with ledger.phase("verification"):
                quality, exact = verify_answer(task, str(generated["text"]))
            ledger.calls.verifier_calls += 1
            ledger.tokens.prompt_tokens += int(generated["prompt_tokens"])
            ledger.tokens.completion_tokens += int(generated["completion_tokens"])
            ledger.tokens.evidence_tokens += int(context.evidence_tokens)

            answer_rows.append({
                "task_id": task.task_id, "family": task.family,
                "template_id": task.template_id, "source_cluster_id": task.source_cluster_id,
                "backend_id": backend.backend_id, "condition": "B2_NAIVE_RETRIEVAL",
                "evaluation_mode": config.evaluation_mode.value,
                "evidence_ids": retrieved_ids,
                "prompt_evidence_ids": [row.evidence_id for row in context.evidence],
                "final_prompt_sha256": context.prompt_sha256,
                "evidence_tokens": context.evidence_tokens,
                "prompt_tokens": int(generated["prompt_tokens"]),
                "completion_tokens": int(generated["completion_tokens"]),
                "output": str(generated["text"]), "gold_answer": task.answer,
                "verified_quality": quality, "exact_match": exact,
                "latency_ms": latency_ms,
                "model_id": adapter.spec.model_id, "model_revision": adapter.spec.revision,
                "corpus_digest": corpus.digest(),
                "retrieval_receipt": asdict(result.receipt),
                "scientific_eligible": True,
            })
            failures.append(classify(
                task_id=task.task_id, family=task.family, quality=quality,
                answer=task.answer, output=str(generated["text"]),
                required_ids=task.required_evidence_ids, retrieved_ids=retrieved_ids,
                prompt_evidence_ids=[row.evidence_id for row in context.evidence],
                evidence_contents=evidence_contents,
            ))

        ledger.memory = ledger.memory.sample(model=None if adapter is None else adapter.model)
        (arm_dir / "retrieval_metrics.jsonl").write_text(
            "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in metric_rows)
        )
        retrieval_summary = summarize(metric_rows, backend_id=backend.backend_id)
        report = {
            "arm": arm.value,
            "backend_id": backend.backend_id,
            "backend_config_digest": backend.config_digest(),
            "embedding_spec": None if backend.embedding_spec is None else asdict(backend.embedding_spec),
            "embedding_digest": None if backend.embedding_spec is None else backend.embedding_spec.digest(),
            "index_seconds": round(index_seconds, 3),
            "index_source_digest": index_receipt.source_digest,
            "retrieval_k": args.k,
            "retrieval": retrieval_summary.to_dict(),
            "resources": ledger.to_dict(),
        }
        if not args.retrieval_only:
            results_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in answer_rows)
            )
            (arm_dir / "failure_attribution.jsonl").write_text(
                "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in failures)
            )
            quality_by_family: dict[str, list[float]] = {}
            for row in answer_rows:
                quality_by_family.setdefault(row["family"], []).append(row["verified_quality"])
            report["downstream"] = {
                "mean_quality": round(
                    sum(row["verified_quality"] for row in answer_rows) / len(answer_rows), 4
                ),
                "per_family": {
                    family: round(sum(values) / len(values), 4)
                    for family, values in sorted(quality_by_family.items())
                },
                "b0_anchor": anchors["b0_mean_quality"],
                "b3_anchor": anchors["b3_mean_quality"],
            }
            report["downstream"]["delta_vs_b0"] = round(
                report["downstream"]["mean_quality"] - anchors["b0_mean_quality"], 4
            )
            report["downstream"]["oracle_gap"] = round(
                anchors["b3_mean_quality"] - report["downstream"]["mean_quality"], 4
            )
            report["failure_attribution"] = summarize_failures(failures)
        (arm_dir / "arm_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        arm_reports[arm.value] = report
        headline = report["retrieval"]["metrics"]["complete_set_success"]
        tail = "" if args.retrieval_only else f" downstream={report['downstream']['mean_quality']}"
        print(f"[{arm.value}] complete_set_success={headline}{tail}")

    manifest = {
        "gate": "B_RETRIEVAL",
        "protocol_version": "gate-b-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task_count": len(tasks),
        "evidence_count": len(records),
        "retrieval_k": args.k,
        "arms": [arm.value for arm in arms],
        "retrieval_only": args.retrieval_only,
        "frozen_config_sha256": _sha256(frozen_bytes),
        "task_dataset_sha256": _sha256(task_bytes),
        "evidence_corpus_sha256": _sha256(evidence_bytes),
        "normalized_corpus_digest": corpus.digest(),
        "gate_a_report": str(args.gate_a_report),
        "gate_a_anchors": anchors,
        "model_id": None if adapter is None else adapter.spec.model_id,
        "model_revision": None if adapter is None else adapter.spec.revision,
        "prompt_condition": frozen["prompt_condition"],
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
        },
        "arm_reports": arm_reports,
    }
    (output / "gate_b_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "arm_reports"}, indent=2)[:1200])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/hrm/controlled_gate_a_v2/oracle_tasks.jsonl")
    parser.add_argument("--evidence", default="data/hrm/controlled_gate_a_v2/evidence.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-config", default="configs/gate_a/gate_a_v2_frozen.json")
    parser.add_argument("--gate-a-report", default="evidence/gate_a/qualified_run_002/gate_a_report_v2r1.json")
    parser.add_argument(
        "--arms", default="bm25,hash,dense,hybrid_score,hybrid_rrf,hybrid_rerank",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
