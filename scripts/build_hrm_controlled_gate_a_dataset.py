#!/usr/bin/env python3
"""Build an immutable controlled capability-use corpus for HRM Gate A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.experiments.controlled_dataset import build_controlled_gate_a_corpus


def _write_jsonl(path: Path, rows: tuple[object, ...]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="New immutable dataset directory")
    parser.add_argument("--seed", type=int, default=3601)
    parser.add_argument("--tasks-per-family", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset directory: {output}")
    corpus = build_controlled_gate_a_corpus(
        seed=args.seed, tasks_per_family=args.tasks_per_family,
    )
    output.mkdir(parents=True)
    _write_jsonl(output / "oracle_tasks.jsonl", corpus.tasks)
    _write_jsonl(output / "evidence.jsonl", corpus.evidence)
    (output / "dataset_manifest.json").write_text(
        json.dumps(corpus.manifest, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(corpus.manifest, indent=2))


if __name__ == "__main__":
    main()
