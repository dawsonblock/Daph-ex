#!/usr/bin/env python3
"""Arm-level mean quality and paired deltas for a context-study run.

Development/pilot readout (Section 11): reports Q(B0..B3), B3-B0, B2-B0,
B1-B0, B1b-B0 overall and per family.  Interpretation only — the Gate A
decision comes from qualify_hrm_context_gate_a.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ARM_ORDER = (
    "B0_NO_CONTEXT", "B1_RANDOM_CONTEXT", "B1B_HARD_DISTRACTOR",
    "B2_NAIVE_RETRIEVAL", "B3_ORACLE_EVIDENCE",
)
SHORT = {
    "B0_NO_CONTEXT": "B0", "B1_RANDOM_CONTEXT": "B1",
    "B1B_HARD_DISTRACTOR": "B1b", "B2_NAIVE_RETRIEVAL": "B2",
    "B3_ORACLE_EVIDENCE": "B3",
}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(evidence_dir: Path) -> dict:
    rows = [json.loads(line) for line in (evidence_dir / "per_task_results.jsonl").read_text().splitlines() if line.strip()]
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_task[row["task_id"]][row["condition"]] = row

    def block(task_ids: list[str]) -> dict:
        quality = {
            SHORT[arm]: _mean([
                float(by_task[t][arm]["verified_quality"]) for t in task_ids if arm in by_task[t]
            ])
            for arm in ARM_ORDER
        }
        deltas = {}
        for arm in ("B1", "B1b", "B2", "B3"):
            if quality.get(arm) is not None and quality["B0"] is not None:
                deltas[f"{arm}-B0"] = round(quality[arm] - quality["B0"], 4)
        return {
            "n": len(task_ids),
            "mean_quality": {k: (round(v, 4) if v is not None else None) for k, v in quality.items()},
            "paired_deltas": deltas,
        }

    families: dict[str, list[str]] = defaultdict(list)
    for task_id, arms in by_task.items():
        families[next(iter(arms.values()))["family"]].append(task_id)

    latencies = [float(row["latency_ms"]) for row in rows]
    completions = [int(row["completion_tokens"]) for row in rows]
    return {
        "evidence_dir": str(evidence_dir),
        "prompt_condition": json.loads((evidence_dir / "manifest.json").read_text()).get("prompt_condition"),
        "overall": block(sorted(by_task)),
        "per_family": {family: block(sorted(ids)) for family, ids in sorted(families.items())},
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
        "mean_completion_tokens": round(sum(completions) / len(completions), 1),
        "peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dirs", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    reports = [summarize(Path(value)) for value in args.evidence_dirs]
    payload = reports[0] if len(reports) == 1 else reports
    if args.output:
        Path(args.output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
