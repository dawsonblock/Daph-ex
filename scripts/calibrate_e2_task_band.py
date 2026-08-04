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
from daph.verified_tasks import calibrated_sensitivity_split, choose_calibration_families


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
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--candidate-dir")
    source_group.add_argument("--candidates", help="Single multi-family pool split disjointly after E2 evaluation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--selection-count", type=int, default=24)
    parser.add_argument("--test-count", type=int, default=24)
    parser.add_argument("--min-e2-accuracy", type=float, default=0.30)
    parser.add_argument("--max-e2-accuracy", type=float, default=0.70)
    parser.add_argument("--target-e2-accuracy", type=float, default=0.50)
    parser.add_argument(
        "--family-stratified", action=argparse.BooleanOptionalAction, default=True,
        help="Balance E2 successes/failures within each task family (default: enabled).",
    )
    parser.add_argument("--min-calibrated-families", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Reuse a complete cached E2 outcome file")
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
    candidate_dir, output = Path(args.candidate_dir) if args.candidate_dir else None, Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports: Dict[str, Any] = {}
    shared_tasks = load_tasks(Path(args.candidates)) if args.candidates else None
    outcome_cache = output / "e2_outcomes.jsonl"
    if shared_tasks is not None and args.resume and outcome_cache.exists():
        shared_outcomes = load_tasks(outcome_cache)
        expected_ids = {str(task["task_id"]) for task in shared_tasks}
        observed_ids = {str(row["task_id"]) for row in shared_outcomes}
        if observed_ids != expected_ids:
            raise ValueError("Cached E2 outcomes do not match the current candidate task IDs")
    else:
        shared_outcomes = evaluate_e2(
            model, tokenizer, shared_tasks, device=device, max_new_tokens=args.max_new_tokens,
        ) if shared_tasks is not None else None
        if shared_outcomes is not None:
            outcome_cache.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in shared_outcomes))
    calibrated_families, family_selection = None, None
    if shared_tasks is not None and args.family_stratified:
        calibrated_families, family_selection = choose_calibration_families(
            shared_tasks, shared_outcomes,
            split_counts=(args.train_count, args.selection_count, args.test_count),
            target_e2_accuracy=args.target_e2_accuracy,
            minimum_families=args.min_calibrated_families,
        )
    remaining_ids = {str(task["task_id"]) for task in shared_tasks} if shared_tasks is not None else None
    for offset, (split, count) in enumerate((
        ("train", args.train_count),
        ("selection", args.selection_count),
        ("test", args.test_count),
    )):
        source = Path(args.candidates) if args.candidates else candidate_dir / f"{split}_candidates.jsonl"
        if shared_tasks is not None:
            tasks = [task for task in shared_tasks if str(task["task_id"]) in remaining_ids]
            outcomes = [row for row in shared_outcomes if str(row["task_id"]) in remaining_ids]
        else:
            tasks = load_tasks(source)
            outcomes = evaluate_e2(
                model, tokenizer, tasks, device=device, max_new_tokens=args.max_new_tokens,
            )
        if args.family_stratified:
            selected, split_manifest = calibrated_sensitivity_split(
                tasks, outcomes, count=count,
                target_e2_accuracy=args.target_e2_accuracy, seed=args.seed + offset,
                included_families=calibrated_families,
            )
            selected_accuracy = float(split_manifest["selected_e2_accuracy"])
            if not args.min_e2_accuracy <= selected_accuracy <= args.max_e2_accuracy:
                raise ValueError(
                    f"Family-stratified {split} accuracy {selected_accuracy:.4f} is outside "
                    f"[{args.min_e2_accuracy:.4f}, {args.max_e2_accuracy:.4f}]"
                )
            report = {
                "selected_count": len(selected),
                "selected_e2_accuracy": selected_accuracy,
                "selection_method": "family_stratified_mixed_success",
                "split_manifest": split_manifest,
            }
        else:
            config = E2DifficultyBandConfig(
                target_size=count,
                min_accuracy=args.min_e2_accuracy,
                max_accuracy=args.max_e2_accuracy,
                target_accuracy=args.target_e2_accuracy,
                seed=args.seed + offset,
            )
            selected, report = select_mixed_success_tasks(tasks, outcomes, config)
        if remaining_ids is not None:
            remaining_ids -= {str(task["task_id"]) for task in selected}
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
        "e2_runtime": "pinned_hf_source_proxy_for_imported_e2",
        "environment": {"torch": torch.__version__, "platform": platform.platform(), "device": str(device)},
        "config": {
            "min_e2_accuracy": args.min_e2_accuracy,
            "max_e2_accuracy": args.max_e2_accuracy,
            "target_e2_accuracy": args.target_e2_accuracy,
            "family_stratified": args.family_stratified,
            "minimum_calibrated_families": args.min_calibrated_families,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "splits": reports,
        "family_selection": family_selection,
    }
    (output / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({split: {key: value for key, value in report.items() if key.startswith("selected_")} for split, report in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
