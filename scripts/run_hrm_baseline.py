#!/usr/bin/env python3
"""Run the untouched native HRM baseline on JSONL tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.baseline.evaluator import BaselineCondition
from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition


def normalize_answer(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="JSONL with task_id, prompt, expected")
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", choices=[item.value for item in PromptCondition], default="direct")
    parser.add_argument(
        "--baseline-condition", choices=[item.value for item in BaselineCondition],
        default=BaselineCondition.NO_CONTEXT.value,
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()
    import torch
    adapter = HRMAdapter.from_pretrained(
        spec=HRMModelSpec(), dtype=torch.bfloat16, device_map=args.device_map,
    )
    task_bytes = Path(args.tasks).read_bytes()
    tasks = [json.loads(line) for line in task_bytes.decode().splitlines() if line.strip()]
    if any("expected" not in task for task in tasks):
        raise ValueError("Every baseline task needs an expected value for verified scoring")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for task in tasks:
            started = time.perf_counter()
            result = adapter.generate(
                str(task["prompt"]), condition=PromptCondition(args.condition),
                max_new_tokens=args.max_new_tokens,
            )
            prompt_condition = result.pop("condition")
            exact_match = normalize_answer(result["text"]) == normalize_answer(task["expected"])
            result.update({
                "task_id": task["task_id"], "expected": task.get("expected"),
                "condition": args.baseline_condition,
                "prompt_condition": prompt_condition,
                "quality": float(exact_match), "accuracy": float(exact_match),
                "exact_match": exact_match, "verified_utility": float(exact_match),
                "task_family": task.get("task_family", "unknown"),
                "difficulty": task.get("difficulty", "unknown"),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "model_id": adapter.spec.model_id, "model_revision": adapter.spec.revision,
                "prefix_lm_masked": True,
            })
            handle.write(json.dumps(result, default=str) + "\n")
    manifest = {
        "tasks": len(tasks), "output": str(output), "model_id": adapter.spec.model_id,
        "model_revision": adapter.spec.revision, "prompt_condition": args.condition,
        "baseline_condition": args.baseline_condition,
        "dataset_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "prefix_lm_masked": True,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({**manifest, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
