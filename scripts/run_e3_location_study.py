#!/usr/bin/env python3
"""Run matched final, heuristic-middle, and profiled-middle E3 studies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--hard-train", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--natural-test", help="Untouched natural-distribution test JSONL")
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-step-counts", default="1,2,4")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--e3-scale", type=float, default=1e-3)
    parser.add_argument("--lr-refinement", type=float, default=1e-4)
    parser.add_argument("--lr-scale", type=float, default=1e-5)
    parser.add_argument("--regression-guard-weight", type=float, default=0.01)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-e2-accuracy", type=float, default=0.30)
    parser.add_argument("--max-e2-accuracy", type=float, default=0.70)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--lambda-compute", type=float, default=1.0)
    parser.add_argument("--lambda-sweep", default="0,0.1,0.25,0.5,1,2")
    parser.add_argument("--bootstrap-group-key", default="template_id")
    parser.add_argument("--experiment-tier", choices=("SMOKE", "PILOT", "QUALIFICATION", "FINAL"), default="SMOKE")
    parser.add_argument("--heldout-steps", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    modes = ("final_refine", "middle_recurrent", "profiled_middle_recurrent")
    reports: Dict[str, Any] = {}
    commands: Dict[str, List[str]] = {}
    for mode in modes:
        mode_output = output / mode
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_e3_hardcase_ablation.py"),
            "--model", args.model,
            "--revision", args.revision,
            "--hard-train", args.hard_train,
            "--selection", args.selection,
            "--test", args.test,
            "--output", str(mode_output),
            "--e3-mode", mode,
            "--latent-step-counts", args.latent_step_counts,
            "--steps", str(args.steps),
            "--e3-scale", str(args.e3_scale),
            "--lr-refinement", str(args.lr_refinement),
            "--lr-scale", str(args.lr_scale),
            "--regression-guard-weight", str(args.regression_guard_weight),
            "--seq-len", str(args.seq_len),
            "--max-new-tokens", str(args.max_new_tokens),
            "--latent-size", str(args.latent_size),
            "--seed", str(args.seed),
            "--device", args.device,
            "--min-e2-accuracy", str(args.min_e2_accuracy),
            "--max-e2-accuracy", str(args.max_e2_accuracy),
            "--bootstrap-samples", str(args.bootstrap_samples),
            "--confidence", str(args.confidence),
            "--lambda-compute", str(args.lambda_compute),
            "--lambda-sweep", args.lambda_sweep,
            "--bootstrap-group-key", args.bootstrap_group_key,
            "--experiment-tier", args.experiment_tier,
            "--heldout-steps", str(args.heldout_steps),
        ]
        if args.natural_test:
            command.extend(("--natural-test", args.natural_test))
        if mode == "profiled_middle_recurrent":
            command.extend(("--profile-dir", args.profile_dir))
        # Keep the evidence portable: record a repository-relative command while
        # executing with the current interpreter and resolved script path.
        commands[mode] = ["python", "scripts/run_e3_hardcase_ablation.py", *command[2:]]
        subprocess.run(command, cwd=ROOT, check=True)
        reports[mode] = json.loads((mode_output / "e3_hardcase_ablation_report.json").read_text())

    selected_steps = {mode: int(report["selected_latent_steps"]) for mode, report in reports.items()}
    heldout_rows = []
    for mode, report in reports.items():
        heldout = report["heldout"]
        heldout_rows.append({
            "mode": mode,
            "refinement_layer": report["architecture"]["refinement_layer"],
            "selected_steps": report["selected_latent_steps"],
            "e2_accuracy": heldout["e2_accuracy"],
            "e3_accuracy": heldout["e3_accuracy"],
            "rescues": heldout["rescue_count"],
            "regressions": heldout["regression_count"],
            "net_rescue_rate": heldout["net_rescue_rate"],
            "e3_ce_delta_vs_e2": heldout["e3_ce_delta_vs_e2"],
            "compute_overhead": heldout["e3_compute_overhead"],
            "quality_lcb95": report["qualification"]["quality_lcb95"],
            "utility_lcb95": report["qualification"]["utility_lcb95"],
            "qualification_status": report["qualification"]["qualification_status"],
            "qualified": report["qualification"]["qualified"],
        })
    best = max(
        heldout_rows,
        key=lambda row: (
            row["net_rescue_rate"], row["e3_accuracy"],
            -row["regressions"], -row["e3_ce_delta_vs_e2"],
        ),
    )
    matched_selected_dose = len(set(selected_steps.values())) == 1
    e3_arm_qualified = bool(matched_selected_dose and best["qualified"])
    policy_allowed = False
    study = {
        "experiment": "matched-e3-final-heuristic-profiled-location-study",
        "model": {"id": args.model, "revision": args.revision},
        "commands": commands,
        "selected_steps": selected_steps,
        "matched_selected_dose": matched_selected_dose,
        "heldout_results": heldout_rows,
        "best_mode": best["mode"],
        "qualification": {
            "qualified": e3_arm_qualified,
            "e3_arm_qualified": e3_arm_qualified,
            "policy_training_allowed": policy_allowed,
            "reason": (
                "E3_ARM_QUALIFIED_ORACLE_GATE_REQUIRED" if e3_arm_qualified
                else "NO_LOCATION_PASSED_QUALITY_AND_UTILITY_GATES"
            ),
        },
    }
    (output / "location_study_report.json").write_text(json.dumps(study, indent=2))
    print(json.dumps(study["qualification"], indent=2))


if __name__ == "__main__":
    main()
