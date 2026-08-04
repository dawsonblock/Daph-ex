#!/usr/bin/env python3
"""Run a fitted controller without access to unchosen counterfactual branches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.verifiers import NumericVerifier
from daph_metareasoner import (
    ActionValueEnsemble,
    ConservativeVOCPolicy,
    HFCausalLMAdapter,
    OnPathExecutor,
    PolicyConfig,
    RuntimeLimits,
    UtilityConfig,
    load_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--split", required=True, choices=("test", "ood"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-cost", type=float, default=0.20)
    parser.add_argument("--uncertainty-beta", type=float, default=1.0)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--fast-source-digest", action="store_true")
    args = parser.parse_args()
    ensemble, artifact = ActionValueEnsemble.load(args.controller)
    if artifact["training_status"] != "VERIFIED_FIT" and not args.allow_unverified:
        raise RuntimeError("On-path execution requires VERIFIED_FIT or --allow-unverified")
    tasks = load_tasks(args.tasks, default_split=args.split)
    adapter = HFCausalLMAdapter.from_pretrained(
        args.model, args.revision, device=args.device,
        max_new_tokens=args.max_new_tokens,
        full_model_digest=not args.fast_source_digest,
    )
    if artifact.get("base_model_digest") != adapter.model_digest:
        raise RuntimeError("Controller artifact is bound to a different base-model digest")
    policy = ConservativeVOCPolicy(
        ensemble, UtilityConfig(), PolicyConfig(uncertainty_beta=args.uncertainty_beta),
    )
    executor = OnPathExecutor(
        adapter, policy, RuntimeLimits(max_steps=args.max_steps, max_cost=args.max_cost),
    )
    verifier = NumericVerifier()
    rows = []
    for task in tasks:
        result = executor.run(task)
        quality, status = verifier(
            {"generated_text": result.answer}, {"expected": task.expected},
        )
        rows.append({**result.to_dict(), "verified_quality": quality, "verifier_status": status})
    report = {
        "policy_status": artifact["training_status"],
        "tasks": len(rows),
        "accuracy": sum(row["verified_quality"] for row in rows) / max(len(rows), 1),
        "mean_steps": sum(row["total_steps"] for row in rows) / max(len(rows), 1),
        "mean_tokens": sum(row["total_tokens"] for row in rows) / max(len(rows), 1),
        "mean_latency_ms": sum(row["total_latency_ms"] for row in rows) / max(len(rows), 1),
        "mean_cost": sum(row["total_cost"] for row in rows) / max(len(rows), 1),
        "outcomes": rows,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
