#!/usr/bin/env python3
"""Qualify whether native HRM can use independently labeled oracle evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.evaluation import GateAConfig, qualify_gate_a
from hrm_adaptive_memory.experiments import EvaluationMode, ExperimentTier, OracleTask


def _jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--tasks", required=True, help="Independent oracle-task JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tier", choices=[value.value for value in ExperimentTier], default="QUALIFICATION")
    parser.add_argument("--minimum-mean-gain", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-mode", choices=[value.value for value in EvaluationMode],
        default=EvaluationMode.CAPABILITY_USE.value,
    )
    parser.add_argument("--require-hard-distractor", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable gate report: {output}")
    report = qualify_gate_a(
        _jsonl(args.results),
        [OracleTask.from_dict(row) for row in _jsonl(args.tasks)],
        GateAConfig(
            tier=ExperimentTier(args.tier),
            minimum_mean_quality_gain=args.minimum_mean_gain,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            evaluation_mode=EvaluationMode(args.evaluation_mode),
            require_hard_distractor=args.require_hard_distractor,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "paired_records"}, indent=2))


if __name__ == "__main__":
    main()
