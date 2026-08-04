#!/usr/bin/env python3
"""Create the E0-E3 frontier and run the receipt-backed oracle gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.effort_frontier import build_effort_frontier, qualify_oracle_opportunity, write_effort_frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambda-compute", type=float, default=1.0)
    parser.add_argument("--lambda-sweep", default="0,0.1,0.25,0.5,1,2")
    parser.add_argument("--qualified-arms", default="")
    parser.add_argument("--group-key", default="template_id")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.per_task_results).read_text().splitlines() if line.strip()]
    lambdas = [float(value) for value in args.lambda_sweep.split(",") if value.strip()]
    qualified = [value.strip() for value in args.qualified_arms.split(",") if value.strip()]
    frontier = build_effort_frontier(rows, lambdas=lambdas, qualified_arms=qualified)
    write_effort_frontier(frontier, args.output)
    oracle = qualify_oracle_opportunity(
        rows, lambda_compute=args.lambda_compute, qualified_non_e2_arms=qualified,
        bootstrap_samples=args.bootstrap_samples, confidence=args.confidence,
        group_key=args.group_key, seed=args.seed,
    )
    output = Path(args.output)
    (output / "oracle_qualification.json").write_text(json.dumps(oracle, indent=2) + "\n")
    (output / "decision.json").write_text(json.dumps({
        "effort_frontier_physical": frontier["physical_compute_ordering"],
        "qualified_non_e2_arms": qualified,
        "oracle_gate_passed": oracle["oracle_gate_passed"],
        "policy_training_allowed": oracle["policy_training_allowed"],
        "reason": oracle["reason"],
    }, indent=2) + "\n")
    print(json.dumps({key: oracle[key] for key in ("mean_oracle_gap", "oracle_gap_lcb95", "policy_training_allowed", "reason")}, indent=2))


if __name__ == "__main__":
    main()
