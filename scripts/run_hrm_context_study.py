#!/usr/bin/env python3
"""Run the canonical paired B0/B1/B2/B3 study with native HRM."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import (
    LocalControlBackend,
    LocalRetrievalMode,
    RuVectorBackend,
    SidecarEndpoint,
)
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.experiments.context_study import (
    ContextStudyConfig,
    ContextStudyRunner,
    EvaluationMode,
    EvidenceCorpus,
    ExperimentTier,
    ModelOutput,
    OracleTask,
)
from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition
from hrm_adaptive_memory.source_lock import SourceLock


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


class NativeHRMExecutor:
    scientific_eligible = True

    def __init__(self, adapter: HRMAdapter, condition: PromptCondition, max_new_tokens: int):
        self.adapter = adapter
        self.condition = condition
        self.max_new_tokens = max_new_tokens
        self.model_id = adapter.spec.model_id
        self.model_revision = adapter.spec.revision

    async def generate(self, prompt: str) -> ModelOutput:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = await asyncio.to_thread(
            self.adapter.generate,
            prompt,
            condition=self.condition,
            max_new_tokens=self.max_new_tokens,
        )
        latency = (time.perf_counter() - started) * 1000
        if torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated())
        elif torch.backends.mps.is_available():
            # MPS has no peak-stats API; current allocation after generation is
            # the closest honest measurement.
            peak = int(torch.mps.current_allocated_memory())
        else:
            peak = 0
        return ModelOutput(
            text=str(result["text"]),
            prompt_tokens=int(result["prompt_tokens"]),
            completion_tokens=int(result["completion_tokens"]),
            latency_ms=latency,
            peak_memory_bytes=peak,
        )


def _load_jsonl(path: Path) -> tuple[bytes, list[dict]]:
    data = path.read_bytes()
    return data, [json.loads(line) for line in data.decode().splitlines() if line.strip()]


async def _run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    partial_path = output / "partial_results.jsonl"
    if output.exists():
        if not args.resume:
            raise FileExistsError(
                f"Evidence directory already exists and will not be overwritten: {output}"
            )
        if (output / "per_task_results.jsonl").exists():
            raise FileExistsError(
                f"Run already finalized; refusing to resume into: {output}"
            )

    task_bytes, task_rows = _load_jsonl(Path(args.tasks))
    evidence_bytes, evidence_rows = _load_jsonl(Path(args.evidence))
    frozen_config_digest = None
    if args.frozen_config:
        frozen_bytes = Path(args.frozen_config).read_bytes()
        frozen = json.loads(frozen_bytes)
        frozen_config_digest = _sha256(frozen_bytes)
        expected_args = {
            "prompt_condition": args.prompt_condition,
            "tier": args.tier,
            "retriever": args.retriever,
            "retrieval_k": args.retrieval_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "evaluation_mode": args.evaluation_mode,
            "include_hard_distractor": args.include_hard_distractor,
            "lambda_evidence_tokens": args.lambda_evidence_tokens,
        }
        drift = {
            key: (frozen[key], observed)
            for key, observed in expected_args.items()
            if frozen.get(key) != observed
        }
        if frozen.get("task_dataset_sha256") != _sha256(task_bytes):
            drift["task_dataset_sha256"] = (frozen.get("task_dataset_sha256"), _sha256(task_bytes))
        if frozen.get("evidence_corpus_sha256") != _sha256(evidence_bytes):
            drift["evidence_corpus_sha256"] = (frozen.get("evidence_corpus_sha256"), _sha256(evidence_bytes))
        if drift:
            raise RuntimeError(f"Run drifts from frozen protocol {args.frozen_config}: {drift}")
    tasks = [OracleTask.from_dict(row) for row in task_rows]
    config = ContextStudyConfig(
        tier=ExperimentTier(args.tier),
        retrieval_k=args.retrieval_k,
        seed=args.seed,
        lambda_evidence_tokens=args.lambda_evidence_tokens,
        evaluation_mode=EvaluationMode(args.evaluation_mode),
        include_hard_distractor=args.include_hard_distractor,
    )
    config.validate(tasks)

    ruvector_endpoint = None
    if args.retriever == "ruvector":
        if not args.gate_a_report:
            raise RuntimeError("RuVector is blocked until a passing Gate A report is supplied")
        gate = json.loads(Path(args.gate_a_report).read_text())
        if not gate.get("retrieval_expansion_allowed"):
            raise RuntimeError("Gate A does not allow RuVector integration")
        if not args.ruvector_url or not args.ruvector_version:
            raise RuntimeError("RuVector requires a local endpoint and pinned version")
        locked = SourceLock.load(args.source_lock).require_runtime("RuVector")
        if locked.observed_version != args.ruvector_version:
            raise RuntimeError("Requested RuVector version differs from the audited source lock")
        ruvector_endpoint = SidecarEndpoint(
            "ruvector", args.ruvector_url, args.sidecar_timeout, args.ruvector_version,
        )
        # Fail before loading HRM when the explicitly requested sidecar is unavailable.
        await RuVectorBackend(ruvector_endpoint).health()

    import torch
    import transformers

    adapter = HRMAdapter.from_pretrained(
        spec=HRMModelSpec(), dtype=torch.bfloat16, device_map=args.device_map,
    )
    codec = HuggingFaceTokenCodec(adapter.tokenizer)
    records = [IndexRecord(
        evidence_id=str(row["evidence_id"]),
        source_id=str(row["source_id"]),
        content=str(row["content"]),
        token_count=codec.count(str(row["content"])),
        source_type=str(row.get("source_type", "source")),
        metadata=dict(row.get("metadata", {})),
    ) for row in evidence_rows]
    corpus = EvidenceCorpus(records)

    if ruvector_endpoint is not None:
        backend = RuVectorBackend(ruvector_endpoint)
        await backend.index(records)
    else:
        backend = LocalControlBackend(LocalRetrievalMode(args.retriever), records)
    backend_health = await backend.health()
    backend_capabilities = await backend.capabilities()

    executor = NativeHRMExecutor(
        adapter,
        PromptCondition(args.prompt_condition),
        args.max_new_tokens,
    )
    runner = ContextStudyRunner(
        corpus=corpus,
        retrieval_backend=backend,
        executor=executor,
        config=config,
        token_codec=codec,
    )

    # Receipts persist per task so a long qualification run is restartable
    # (--resume) without duplicating completed tasks.
    output.mkdir(parents=True, exist_ok=True)
    expected_conditions = {condition.value for condition in config.conditions()}
    completed: dict[str, list[dict]] = {}
    if partial_path.exists():
        stale = [
            json.loads(line) for line in partial_path.read_text().splitlines() if line.strip()
        ]
        by_task: dict[str, dict[str, dict]] = {}
        for row in stale:
            by_task.setdefault(row["task_id"], {})[row["condition"]] = row
        completed = {
            task_id: [arms[value] for value in sorted(arms)]
            for task_id, arms in by_task.items()
            if set(arms) == expected_conditions
        }
        print(f"Resuming: {len(completed)} of {len(tasks)} tasks already complete")
    receipt_rows: list[dict] = []
    with partial_path.open("a") as handle:
        for task in tasks:
            if task.task_id in completed:
                receipt_rows.extend(completed[task.task_id])
                continue
            task_receipts = await runner.run_task(task)
            rows = [row.to_dict() for row in task_receipts]
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            receipt_rows.extend(rows)

    results_path = output / "per_task_results.jsonl"
    results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in receipt_rows))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        source_digest = hashlib.sha256()
        for path in sorted(ROOT.rglob("*.py")):
            if any(part in {"evidence", "__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            source_digest.update(str(path.relative_to(ROOT)).encode())
            source_digest.update(path.read_bytes())
        commit = f"non-git-tree:{source_digest.hexdigest()}"
    manifest = {
        "protocol_version": "hrm-context-study-v2",
        "tier": config.tier.value,
        "task_count": len(tasks),
        "receipt_count": len(receipt_rows),
        "conditions_per_task": len(config.conditions()),
        "evaluation_mode": config.evaluation_mode.value,
        "hard_distractor_control": config.include_hard_distractor,
        "model_id": executor.model_id,
        "model_revision": executor.model_revision,
        "prompt_condition": args.prompt_condition,
        "prefix_lm_masked": True,
        "retrieval_backend": backend.backend_id,
        "task_dataset_sha256": _sha256(task_bytes),
        "evidence_corpus_sha256": _sha256(evidence_bytes),
        "normalized_corpus_digest": corpus.digest(),
        "frozen_config_digest": frozen_config_digest,
        "source_commit": commit,
        "scientific_eligible": True,
        "retrieval_expansion_allowed": False,
        "graphiti_integration_allowed": False,
        "controller_training_allowed": False,
        "results_sha256": _sha256(results_path.read_bytes()),
    }
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "source_commit": commit,
    }
    backend_manifest = {
        "health": asdict(backend_health),
        "capabilities": asdict(backend_capabilities),
        "silent_fallback_allowed": False,
    }
    model_manifest = {
        "model_id": executor.model_id,
        "model_revision": executor.model_revision,
        "prompt_condition": args.prompt_condition,
        "prefix_lm_masked": True,
        "scientific_eligible": True,
    }
    dataset_manifest = {
        "task_count": len(tasks),
        "task_dataset_sha256": _sha256(task_bytes),
        "evidence_corpus_sha256": _sha256(evidence_bytes),
        "normalized_corpus_digest": corpus.digest(),
        "independent_oracle_labels": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps(environment, sort_keys=True, indent=2) + "\n")
    (output / "backend_manifest.json").write_text(json.dumps(backend_manifest, sort_keys=True, indent=2) + "\n")
    (output / "model_manifest.json").write_text(json.dumps(model_manifest, sort_keys=True, indent=2) + "\n")
    (output / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Independent oracle-task JSONL")
    parser.add_argument("--evidence", required=True, help="Immutable evidence-corpus JSONL")
    parser.add_argument("--output", required=True, help="New evidence directory")
    parser.add_argument("--tier", choices=[value.value for value in ExperimentTier], default="SMOKE")
    parser.add_argument("--retriever", choices=["bm25", "hash", "hybrid", "ruvector"], default="bm25")
    parser.add_argument("--retrieval-k", type=int, default=10)
    parser.add_argument("--prompt-condition", choices=[value.value for value in PromptCondition], default="synth,cot")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-evidence-tokens", type=float, default=0.0)
    parser.add_argument(
        "--evaluation-mode", choices=[value.value for value in EvaluationMode],
        default=EvaluationMode.CAPABILITY_USE.value,
        help="Capability-use and evidence-grounded studies are deliberately separate.",
    )
    parser.add_argument(
        "--include-hard-distractor", action="store_true",
        help="Add the optional answer-free B1b lexical hard-distractor control.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue an interrupted run, skipping tasks with complete receipts.",
    )
    parser.add_argument(
        "--frozen-config",
        help="Frozen protocol JSON; the run aborts on any drift from it.",
    )
    parser.add_argument("--gate-a-report")
    parser.add_argument("--ruvector-url")
    parser.add_argument("--ruvector-version")
    parser.add_argument("--source-lock", default="third_party/sources.lock.json")
    parser.add_argument("--sidecar-timeout", type=float, default=10.0)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
