#!/usr/bin/env python3
"""Freeze the Gate A qualification protocol into an immutable manifest.

Section 12: after pilot technical sanity, every experimental degree of freedom
is pinned — prompts, model revision, decoding, token budgets, verifier,
benchmark generator, grouping scheme, statistical thresholds — with SHA256
digests of the benchmark JSONL, evidence corpus, frozen config, and the
relevant source files.  The qualification run must then be started with
--frozen-config pointing at the config this script validates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.hrm.model import HRMModelSpec

PROTOCOL_SOURCE_FILES = (
    "hrm_adaptive_memory/experiments/context_study.py",
    "hrm_adaptive_memory/experiments/controlled_dataset.py",
    "hrm_adaptive_memory/evaluation/context_gate.py",
    "hrm_adaptive_memory/evaluation/bootstrap.py",
    "hrm_adaptive_memory/hrm/model.py",
    "hrm_adaptive_memory/context/packer.py",
    "hrm_adaptive_memory/backends/local.py",
    "scripts/run_hrm_context_study.py",
    "scripts/qualify_hrm_context_gate_a.py",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Frozen gate_a_qualification.json")
    parser.add_argument("--pilot-evidence-dir", required=True)
    parser.add_argument("--output", default="evidence/gate_a/protocol_manifest_v1_superseded.json")
    parser.add_argument("--source-lock", default="third_party/sources.lock.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Protocol manifest is immutable; refusing to overwrite {output}")

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    for key in (
        "prompt_condition", "tier", "retriever", "retrieval_k", "max_new_tokens",
        "seed", "evaluation_mode", "include_hard_distractor", "lambda_evidence_tokens",
        "tasks_path", "evidence_path", "task_dataset_sha256", "evidence_corpus_sha256",
        "condition_selection_rationale",
    ):
        if key not in config:
            raise ValueError(f"Frozen config is missing required key: {key}")
    tasks_path = Path(config["tasks_path"])
    evidence_path = Path(config["evidence_path"])
    if _sha256_file(tasks_path) != config["task_dataset_sha256"]:
        raise ValueError("Frozen config task digest does not match the benchmark JSONL")
    if _sha256_file(evidence_path) != config["evidence_corpus_sha256"]:
        raise ValueError("Frozen config evidence digest does not match the corpus JSONL")

    pilot_dir = Path(args.pilot_evidence_dir)
    pilot_manifest = json.loads((pilot_dir / "manifest.json").read_text())
    if pilot_manifest["prompt_condition"] != config["prompt_condition"]:
        raise ValueError("Pilot ran a different prompt condition than the frozen config")

    spec = HRMModelSpec()
    manifest = {
        "manifest_type": "gate_a_protocol_freeze",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_config_path": str(config_path),
        "frozen_config_sha256": _sha256_file(config_path),
        "frozen_config": config,
        "benchmark_tasks_sha256": config["task_dataset_sha256"],
        "evidence_corpus_sha256": config["evidence_corpus_sha256"],
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "model_architecture": spec.architecture,
        "source_lock_sha256": _sha256_file(Path(args.source_lock)),
        "pilot_evidence_dir": str(pilot_dir),
        "pilot_results_sha256": pilot_manifest["results_sha256"],
        "statistical_thresholds": {
            "minimum_mean_quality_gain": 0.05,
            "bootstrap_samples": 10_000,
            "confidence": 0.95,
            "group_keys": ["template_id", "family", "source_cluster_id"],
            "grouped_lcb_requirement": "LCB95 > 0 for every grouping key",
        },
        "source_files_sha256": {
            name: _sha256_file(ROOT / name) for name in PROTOCOL_SOURCE_FILES
        },
    }
    output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("frozen_config", "source_files_sha256")}, indent=2))


if __name__ == "__main__":
    main()
