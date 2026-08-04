#!/usr/bin/env python3
"""Create leakage-resistant Stage 1 metareasoning task splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph_metareasoner import Task, build_split_manifest


def _task(
    split: str, index: int, family: str, values: Tuple[int, ...], template: int,
    seed: int,
) -> Dict[str, Any]:
    a, b = values[:2]
    templates = {
        "addition": (
            "Compute {a} + {b}. Answer:",
            "What is the sum of {a} and {b}? Answer:",
            "Add {a} to {b}. Result:",
            "Evaluate {a}+{b}. Final answer:",
        ),
        "subtraction": (
            "Compute {a} - {b}. Answer:",
            "What is {a} minus {b}? Answer:",
            "Subtract {b} from {a}. Result:",
            "Evaluate {a}-{b}. Final answer:",
        ),
        "multiplication": (
            "Compute {a} × {b}. Answer:",
            "What is the product of {a} and {b}? Answer:",
            "Multiply {a} by {b}. Result:",
            "Evaluate {a}*{b}. Final answer:",
        ),
        "two_step_ood": (
            "Start with {a}, add {b}, then multiply by {c}. Answer:",
            "Compute ({a} + {b}) × {c}. Answer:",
            "Add {a} and {b}; multiply that sum by {c}. Result:",
            "Evaluate ({a}+{b})*{c}. Final answer:",
        ),
    }
    prompt = templates[family][template].format(
        a=a, b=b, c=values[2] if len(values) > 2 else 0,
    )
    if family == "addition":
        expected = a + b
    elif family == "subtraction":
        expected = a - b
    elif family == "multiplication":
        expected = a * b
    else:
        expected = (a + b) * values[2]
    return {
        "task_id": f"{split}-{family}-{index:06d}",
        "prompt": prompt,
        "expected": str(expected),
        "family_id": family,
        "split": split,
        "template_id": f"{split}-template-{template}",
        "generator_seed": str(seed),
        "metadata": {"operands": list(values)},
    }


def make_split(split: str, count: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    families = ("two_step_ood",) if split == "ood" else (
        "addition", "subtraction", "multiplication",
    )
    split_template = {"experience": 0, "validation": 1, "test": 2, "ood": 3}[split]
    seen: Set[Tuple[str, Tuple[int, ...]]] = set()
    rows = []
    while len(rows) < count:
        family = families[len(rows) % len(families)]
        if family == "multiplication":
            values = (rng.randint(2, 99), rng.randint(2, 99))
        elif family == "two_step_ood":
            values = (rng.randint(1, 30), rng.randint(1, 30), rng.randint(2, 9))
        else:
            values = (rng.randint(0, 99), rng.randint(0, 99))
            if family == "subtraction" and values[1] > values[0]:
                values = (values[1], values[0])
        key = (family, values)
        if key in seen:
            continue
        seen.add(key)
        rows.append(_task(split, len(rows), family, values, split_template, seed + len(rows)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--experience", type=int, default=5000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--ood", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for offset, split in enumerate(("experience", "validation", "test", "ood")):
        rows = make_split(split, int(getattr(args, split)), args.seed + offset * 100_000)
        all_rows.extend(rows)
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    tasks = [Task(**row) for row in all_rows]
    manifest = build_split_manifest(tasks)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value["tasks"] for key, value in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
