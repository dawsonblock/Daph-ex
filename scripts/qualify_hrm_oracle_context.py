#!/usr/bin/env python3
"""Legacy descriptive oracle-context diagnostic; use Gate A for qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.baseline.evaluator import BaselineCondition, BaselineResult, OracleContextGate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.05)
    parser.add_argument("--minimum-tasks", type=int, default=500)
    args = parser.parse_args()
    rows = []
    for line in Path(args.results).read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        rows.append(BaselineResult(
            task_id=str(row["task_id"]), condition=BaselineCondition(row["condition"]),
            quality=float(row["quality"]), verified_utility=float(row.get("verified_utility", row["quality"])),
            exact_match=bool(row.get("exact_match", False)), prompt_tokens=int(row.get("prompt_tokens", 0)),
            completion_tokens=int(row.get("completion_tokens", 0)), latency_ms=float(row.get("latency_ms", 0)),
            peak_memory_bytes=int(row.get("peak_memory_bytes", 0)), task_family=str(row.get("task_family", "unknown")),
            difficulty=str(row.get("difficulty", "unknown")),
        ))
    report = OracleContextGate(
        minimum_oracle_quality_gain=args.minimum_gain,
        minimum_paired_tasks=args.minimum_tasks,
    ).evaluate(rows)
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
