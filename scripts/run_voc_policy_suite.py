#!/usr/bin/env python3
"""Execute learned, sham, fixed, and heuristic policies on-path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.verifiers import NumericVerifier
from daph_metareasoner import (
    Action,
    ActionValueEnsemble,
    ConservativeVOCPolicy,
    FixedRuntimePolicy,
    HFCausalLMAdapter,
    OnPathExecutor,
    PolicyConfig,
    RuntimeLimits,
    ThresholdRuntimePolicy,
    UtilityConfig,
    bootstrap_lcb,
    load_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--hidden-controller", required=True)
    parser.add_argument("--sham-controller", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--split", required=True, choices=("test", "ood"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixed-action", default="STOP", choices=tuple(action.value for action in Action))
    parser.add_argument("--fixed-depth", type=int, default=1)
    parser.add_argument("--heuristic-feature", default="confidence", choices=("confidence", "entropy"))
    parser.add_argument("--heuristic-threshold", type=float, default=0.75)
    parser.add_argument("--heuristic-action", default="VERIFY", choices=("THINK", "VERIFY", "DECOMPOSE"))
    parser.add_argument("--uncertainty-beta", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-cost", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args()
    hidden, hidden_artifact = ActionValueEnsemble.load(args.hidden_controller)
    sham, sham_artifact = ActionValueEnsemble.load(args.sham_controller)
    if hidden_artifact["training_status"] != "VERIFIED_FIT" and not args.allow_unverified:
        raise RuntimeError("Policy suite requires a VERIFIED_FIT hidden controller")
    adapter = HFCausalLMAdapter.from_pretrained(
        args.model, args.revision, device=args.device,
        max_new_tokens=args.max_new_tokens, full_model_digest=True,
    )
    for artifact in (hidden_artifact, sham_artifact):
        if artifact.get("base_model_digest") != adapter.model_digest:
            raise RuntimeError("Controller artifact/base-model digest mismatch")
    utility = UtilityConfig()
    limits = RuntimeLimits(max_steps=args.max_steps, max_cost=args.max_cost)
    policies = {
        "immediate_stop": FixedRuntimePolicy(Action.STOP, utility=utility),
        "best_fixed": FixedRuntimePolicy(
            args.fixed_action, max_actions=args.fixed_depth, utility=utility,
        ),
        "heuristic": ThresholdRuntimePolicy(
            args.heuristic_action, feature=args.heuristic_feature,
            threshold=args.heuristic_threshold, utility=utility,
        ),
        "sham": ConservativeVOCPolicy(
            sham, utility, PolicyConfig(uncertainty_beta=args.uncertainty_beta),
        ),
        "learned": ConservativeVOCPolicy(
            hidden, utility, PolicyConfig(uncertainty_beta=args.uncertainty_beta),
        ),
    }
    tasks = load_tasks(args.tasks, default_split=args.split)
    verifier = NumericVerifier()
    outcomes = {}
    for policy_name, policy in policies.items():
        executor = OnPathExecutor(adapter, policy, limits)
        rows = []
        for task in tasks:
            result = executor.run(task)
            quality, status = verifier(
                {"generated_text": result.answer}, {"expected": task.expected},
            )
            initial_quality, initial_status = verifier(
                {"generated_text": result.initial_answer}, {"expected": task.expected},
            )
            rows.append({
                **result.to_dict(),
                "initial_verified_quality": initial_quality,
                "initial_verifier_status": initial_status,
                "verified_quality": quality,
                "verifier_status": status,
                "utility": float(quality) - result.total_cost,
            })
        outcomes[policy_name] = rows
    comparisons = {}
    learned_by_id = {row["task_id"]: row for row in outcomes["learned"]}
    for control in ("immediate_stop", "best_fixed", "heuristic", "sham"):
        control_by_id = {row["task_id"]: row for row in outcomes[control]}
        deltas = [
            learned_by_id[task.task_id]["utility"] - control_by_id[task.task_id]["utility"]
            for task in tasks
        ]
        lcb = bootstrap_lcb(
            deltas, confidence=args.confidence,
            samples=args.bootstrap_samples, seed=args.seed,
        )
        comparisons[f"learned_vs_{control}"] = {
            "mean_utility_delta": sum(deltas) / len(deltas),
            "utility_delta_lcb": lcb,
            "qualified": sum(deltas) / len(deltas) > 0.0 and lcb > 0.0,
        }
    summary = {
        name: {
            "accuracy": sum(row["verified_quality"] for row in rows) / len(rows),
            "mean_utility": sum(row["utility"] for row in rows) / len(rows),
            "mean_steps": sum(row["total_steps"] for row in rows) / len(rows),
            "mean_tokens": sum(row["total_tokens"] for row in rows) / len(rows),
            "mean_latency_ms": sum(row["total_latency_ms"] for row in rows) / len(rows),
            "harmful_continuation_rate": sum(
                row["initial_verified_quality"] > row["verified_quality"] for row in rows
            ) / len(rows),
            "waste_rate": sum(
                row["total_cost"] > 0.0
                and row["initial_verified_quality"] == row["verified_quality"]
                for row in rows
            ) / len(rows),
        }
        for name, rows in outcomes.items()
    }
    report = {
        "split": args.split,
        "tasks": len(tasks),
        "summary": summary,
        "comparisons": comparisons,
        "qualified": all(result["qualified"] for result in comparisons.values()),
        "outcomes": outcomes,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
