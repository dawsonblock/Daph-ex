#!/usr/bin/env python3
"""Collect isolated STOP/THINK/VERIFY/DECOMPOSE outcomes from one frozen model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph.verifiers import NumericVerifier
from daph_metareasoner import (
    CollectionConfig,
    CounterfactualExperienceCollector,
    HFCausalLMAdapter,
    UtilityConfig,
    build_split_manifest,
    load_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--split", required=True, choices=("experience", "validation", "test", "ood"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-states-per-task", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fast-source-digest", action="store_true")
    parser.add_argument("--normalized-compute-weight", type=float, default=0.0)
    args = parser.parse_args()
    tasks = load_tasks(args.tasks, default_split=args.split)
    if any(task.split != args.split for task in tasks):
        raise ValueError("Every task must match --split")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter = HFCausalLMAdapter.from_pretrained(
        args.model, args.revision, device=args.device,
        max_new_tokens=args.max_new_tokens,
        full_model_digest=not args.fast_source_digest,
    )
    collector = CounterfactualExperienceCollector(
        adapter,
        NumericVerifier(),
        UtilityConfig(normalized_compute_weight=args.normalized_compute_weight),
        CollectionConfig(
            max_depth=args.max_depth,
            max_states_per_task=args.max_states_per_task,
        ),
    )
    records = collector.collect_many(tasks)
    receipt = collector.save(records, output)
    run_manifest = {
        "model": {"id": args.model, "revision": args.revision},
        "model_digest": adapter.model_digest,
        "environment_digest": adapter.environment_digest,
        "split_manifest": build_split_manifest(tasks),
        "collection": receipt,
        "config": vars(args),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(run_manifest, indent=2)
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
