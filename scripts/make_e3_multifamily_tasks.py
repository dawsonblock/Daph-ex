#!/usr/bin/env python3
"""Generate deterministic multi-family candidates and untouched natural test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.verified_tasks import generate_verified_tasks, natural_heldout_split


def _write(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--count-per-family", type=int, default=200)
    parser.add_argument("--natural-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = generate_verified_tasks(count_per_family=args.count_per_family, seed=args.seed)
    natural, manifest = natural_heldout_split(tasks, count=args.natural_count, seed=args.seed + 1)
    natural_ids = {row["task_id"] for row in natural}
    candidates = [row for row in tasks if row["task_id"] not in natural_ids]
    _write(output / "calibration_candidates.jsonl", candidates)
    _write(output / "natural_test.jsonl", natural)
    (output / "split_manifest.json").write_text(json.dumps({
        "natural": manifest,
        "calibration_candidates": {
            "count": len(candidates), "e2_outcomes_inspected": False,
            "e3_outcomes_inspected": False,
        },
        "disjoint": not bool(natural_ids & {row["task_id"] for row in candidates}),
    }, indent=2) + "\n")
    print(json.dumps({"candidates": len(candidates), "natural": len(natural)}, indent=2))


if __name__ == "__main__":
    main()
