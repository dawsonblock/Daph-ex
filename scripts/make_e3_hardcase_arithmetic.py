#!/usr/bin/env python3
"""Create disjoint deterministic arithmetic task splits for the E3 ablation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def make_tasks(count: int, *, seed: int, split: str) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    tasks: List[Dict[str, Any]] = []
    seen = set()
    while len(tasks) < count:
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        c, d = rng.randint(10, 99), rng.randint(10, 99)
        key = (a, b, c, d)
        if key in seen:
            continue
        seen.add(key)
        tasks.append({
            "task_id": f"{split}-{len(tasks):04d}",
            "prompt": f"Calculate exactly: ({a} * {b}) + ({c} * {d}). Answer with only the integer: ",
            "expected": str(a * b + c * d),
            "difficulty_bucket": "hard",
            "task_family": "synthetic_composed_arithmetic_v1",
        })
    return tasks


def write_jsonl(tasks: List[Dict[str, Any]], path: Path) -> None:
    path.write_text("".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--selection-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(make_tasks(args.train_count, seed=args.seed, split="train"), output / "train.jsonl")
    write_jsonl(make_tasks(args.selection_count, seed=args.seed + 1, split="selection"), output / "selection.jsonl")
    write_jsonl(make_tasks(args.test_count, seed=args.seed + 2, split="test"), output / "test.jsonl")


if __name__ == "__main__":
    main()
