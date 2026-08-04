#!/usr/bin/env python3
"""Select disjoint task splits with a predeclared mixed E2 success rate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.e3_experiment import (
    E2DifficultyBandConfig,
    numeric_answer_correct,
    select_mixed_success_tasks,
)


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No calibration candidates in {path}")
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def evaluate_e2(
    model: Any, tokenizer: Any, tasks: Sequence[Dict[str, Any]],
    *, device: torch.device, max_new_tokens: int,
) -> List[Dict[str, Any]]:
    outcomes = []
    for task in tasks:
        ids = tokenizer(
            str(task["prompt"]), add_special_tokens=False, return_tensors="pt",
        )["input_ids"].to(device)
        generated = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(generated[0, ids.size(1):], skip_special_tokens=True)
        outcomes.append({
            "task_id": str(task["task_id"]),
            "e2_correct": numeric_answer_correct(completion, task["expected"]),
            "e2_completion": completion,
        })
    return outcomes


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--selection-count", type=int, default=24)
    parser.add_argument("--test-count", type=int, default=24)
    parser.add_argument("--min-e2-accuracy", type=float, default=0.30)
    parser.add_argument("--max-e2-accuracy", type=float, default=0.70)
    parser.add_argument("--target-e2-accuracy", type=float, default=0.50)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32,
    ).to(device).eval()
    candidate_dir, output = Path(args.candidate_dir), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports: Dict[str, Any] = {}
    for offset, (split, count) in enumerate((
        ("train", args.train_count),
        ("selection", args.selection_count),
        ("test", args.test_count),
    )):
        source = candidate_dir / f"{split}_candidates.jsonl"
        tasks = load_tasks(source)
        outcomes = evaluate_e2(
            model, tokenizer, tasks, device=device, max_new_tokens=args.max_new_tokens,
        )
        config = E2DifficultyBandConfig(
            target_size=count,
            min_accuracy=args.min_e2_accuracy,
            max_accuracy=args.max_e2_accuracy,
            target_accuracy=args.target_e2_accuracy,
            seed=args.seed + offset,
        )
        selected, report = select_mixed_success_tasks(tasks, outcomes, config)
        destination = output / f"{split}.jsonl"
        destination.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
        reports[split] = {
            **report,
            "candidate_path": str(source),
            "candidate_sha256": sha256(source),
            "selected_path": str(destination),
            "selected_sha256": sha256(destination),
            "outcomes": outcomes,
        }
    manifest = {
        "experiment": "e2-mixed-success-task-calibration",
        "model": {"id": args.model, "revision": args.revision},
        "e2_runtime": "pinned_hf_source_with_exact_phase0a_parity",
        "environment": {"torch": torch.__version__, "platform": platform.platform(), "device": str(device)},
        "config": {
            "min_e2_accuracy": args.min_e2_accuracy,
            "max_e2_accuracy": args.max_e2_accuracy,
            "target_e2_accuracy": args.target_e2_accuracy,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "splits": reports,
    }
    (output / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({split: {key: value for key, value in report.items() if key.startswith("selected_")} for split, report in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
