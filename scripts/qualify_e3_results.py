#!/usr/bin/env python3
"""Create a versioned E3-Q/E3-U evidence package from paired raw results."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import daph
from daph.e3_metrics import E3QualificationConfig, lambda_sweep, qualify_e3_pairs
from daph.e3_protocol import ClaimStrength, EvidenceMetadata, digest_json, write_evidence_metadata


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No paired records in {path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _claim(report: Dict[str, Any]) -> ClaimStrength:
    if report["qualified"]:
        return ClaimStrength.STATISTICALLY_QUALIFIED
    if report["tasks"] >= 200:
        return ClaimStrength.PILOT_EVIDENCE
    if report["rescue_count"] or report["regression_count"]:
        return ClaimStrength.MECHANISM_SIGNAL
    return ClaimStrength.ENGINEERING_PASS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated-results", required=True)
    parser.add_argument("--natural-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambda-compute", type=float, default=1.0)
    parser.add_argument("--lambda-sweep", default="0,0.1,0.25,0.5,1,2")
    parser.add_argument("--group-key", default="template_id")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--pytest-output", required=True)
    parser.add_argument("--model-id", default="unspecified")
    parser.add_argument("--model-revision", default="unspecified")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Evidence output must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    calibrated_path, natural_path = Path(args.calibrated_results), Path(args.natural_results)
    calibrated, natural = _read_jsonl(calibrated_path), _read_jsonl(natural_path)
    config = E3QualificationConfig(
        lambda_compute=args.lambda_compute, bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence, group_key=args.group_key, seed=args.seed,
    )
    calibrated_report = qualify_e3_pairs(calibrated, config)
    natural_report = qualify_e3_pairs(natural, config)
    lambdas = [float(value) for value in args.lambda_sweep.split(",") if value.strip()]
    sweep = {
        "calibrated_sensitivity": lambda_sweep(
            calibrated, lambdas, bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence, group_key=args.group_key, seed=args.seed,
        ),
        "natural_heldout": lambda_sweep(
            natural, lambdas, bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence, group_key=args.group_key, seed=args.seed + 100,
        ),
    }
    overall_pass = calibrated_report["qualified"] and natural_report["qualified"]
    decision = {
        "calibrated_status": calibrated_report["qualification_status"],
        "natural_status": natural_report["qualification_status"],
        "canonical_e3_promoted": overall_pass,
        "policy_training_allowed": False,
        "reason": "ORACLE_GATE_REQUIRED" if overall_pass else "E3_QUALITY_OR_UTILITY_GATE_FAILED",
        "claim_strength": _claim(natural_report).value,
    }
    config_payload = vars(args)
    pytest_path = Path(args.pytest_output)
    metadata = EvidenceMetadata(
        artifact_commit=_git("rev-parse", "HEAD"), repository_version=daph.__version__,
        test_count_at_creation=args.test_count, pytest_digest=_sha256(pytest_path),
        config_digest=digest_json(config_payload),
        source_tree_digest=_git("rev-parse", "HEAD^{tree}"),
        claim_strength=_claim(natural_report),
    )
    write_evidence_metadata(output, metadata)
    (output / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
    }, indent=2) + "\n")
    (output / "source_model.json").write_text(json.dumps({
        "model_id": args.model_id, "revision": args.model_revision,
    }, indent=2) + "\n")
    (output / "source_revision.txt").write_text(args.model_revision + "\n")
    (output / "config.json").write_text(json.dumps(config_payload, indent=2, default=str) + "\n")
    (output / "dataset_manifest.json").write_text(json.dumps({
        "calibrated_sensitivity": {"path": str(calibrated_path), "sha256": _sha256(calibrated_path), "selection_may_use_e2": True, "selection_may_use_e3": False},
        "natural_heldout": {"path": str(natural_path), "sha256": _sha256(natural_path), "selection_may_use_e2": False, "selection_may_use_e3": False},
    }, indent=2) + "\n")
    with (output / "per_example_results.jsonl").open("w") as handle:
        for split, rows in (("calibrated_sensitivity", calibrated_report["paired_records"]), ("natural_heldout", natural_report["paired_records"])):
            for row in rows:
                handle.write(json.dumps({"evaluation_split": split, **row}, sort_keys=True) + "\n")
    (output / "quality_bootstrap.json").write_text(json.dumps({
        "calibrated_sensitivity": calibrated_report["quality_gate"],
        "natural_heldout": natural_report["quality_gate"],
    }, indent=2) + "\n")
    (output / "utility_bootstrap.json").write_text(json.dumps({
        "calibrated_sensitivity": calibrated_report["utility_gate"],
        "natural_heldout": natural_report["utility_gate"],
    }, indent=2) + "\n")
    (output / "lambda_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n")
    (output / "rescue_regression.json").write_text(json.dumps({
        split: {key: report[key] for key in ("tasks", "rescues", "regressions", "rescue_rate", "regression_rate", "net_rescue_rate", "by_task_family", "by_difficulty")}
        for split, report in (("calibrated_sensitivity", calibrated_report), ("natural_heldout", natural_report))
    }, indent=2) + "\n")
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    (output / "summary.md").write_text(
        "# E3 qualification summary\n\n"
        f"- Calibrated sensitivity: `{decision['calibrated_status']}`\n"
        f"- Natural held-out: `{decision['natural_status']}`\n"
        f"- Claim strength: `{decision['claim_strength']}`\n"
        "- Router training: blocked pending a qualified non-E2 arm and positive oracle gate.\n\n"
        "Quality is measured separately from cost-aware utility; all compute values are supplied by per-task execution receipts.\n"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
