#!/usr/bin/env python3
"""Create disjoint mixed-difficulty addition candidates for E2 calibration."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


TEMPLATES = (
    "{a} + {b} =",
    "Question: What is {a} + {b}? Answer:",
    "Calculate {a}+{b}. The answer is",
)


def make_candidates(
    count: int, *, seed: int, split: str, seen: Set[Tuple[int, int, int]],
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    tasks: List[Dict[str, Any]] = []
    while len(tasks) < count:
        if rng.random() < 0.70:
            a, b = rng.randint(0, 15), rng.randint(0, 15)
        else:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
        template_index = rng.randrange(len(TEMPLATES))
        key = (a, b, template_index)
        if key in seen:
            continue
        seen.add(key)
        tasks.append({
            "task_id": f"{split}-candidate-{len(tasks):04d}",
            "prompt": TEMPLATES[template_index].format(a=a, b=b),
            "expected": str(a + b),
            "difficulty_bucket": "easy" if max(a, b) <= 15 else "medium",
            "task_family": "single_addition_calibrated_v1",
        })
    return tasks


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-candidates", type=int, default=256)
    parser.add_argument("--selection-candidates", type=int, default=128)
    parser.add_argument("--test-candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seen: Set[Tuple[int, int, int]] = set()
    for offset, (split, count) in enumerate((
        ("train", args.train_candidates),
        ("selection", args.selection_candidates),
        ("test", args.test_candidates),
    )):
        write_jsonl(
            make_candidates(count, seed=args.seed + offset, split=split, seen=seen),
            output / f"{split}_candidates.jsonl",
        )


if __name__ == "__main__":
    main()
