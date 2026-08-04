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
sys.path.insert(0, str(ROOT))

from daph.e3_metrics import E3QualificationConfig, e3_pair_metrics, qualify_e3_pairs
from daph.e3_protocol import ExperimentScale, ExperimentTier, promote_e3_placement


def _parse_seeds(value: str, fallback: int) -> tuple[int, ...]:
    seeds = tuple(sorted(set(int(part) for part in value.split(",") if part.strip()))) if value.strip() else (int(fallback),)
    if not seeds:
        raise ValueError("--training-seeds must contain at least one integer seed")
    return seeds


def _jsonl_count(path: str) -> int:
    return sum(1 for line in Path(path).read_text().splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--hard-train", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--natural-test", help="Untouched natural-distribution test JSONL")
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--profile-stability-dir", help="Multi-seed profile stability evidence for profile-guided promotion")
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
    parser.add_argument("--training-seeds", help="Comma-separated independent E3 training seeds; defaults to --seed.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-e2-accuracy", type=float, default=0.30)
    parser.add_argument("--max-e2-accuracy", type=float, default=0.70)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--lambda-compute", type=float, default=1.0)
    parser.add_argument("--lambda-sweep", default="0,0.1,0.25,0.5,1,2")
    parser.add_argument("--bootstrap-group-key", default="template_id")
    parser.add_argument("--experiment-tier", choices=("SMOKE", "PILOT", "QUALIFICATION", "FINAL"), default="SMOKE")
    parser.add_argument("--predeclared-heldout-examples", type=int, help="Required for FINAL")
    parser.add_argument("--heldout-steps", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-mode/per-seed reports")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_seeds = _parse_seeds(args.training_seeds or "", args.seed)
    tier = ExperimentTier(args.experiment_tier)
    calibrated_count = _jsonl_count(args.test)
    natural_count = _jsonl_count(args.natural_test) if args.natural_test else 0
    calibrated_scale = ExperimentScale(
        tier=tier, heldout_examples=calibrated_count, training_seeds=training_seeds,
        evaluation_seed=args.seed, predeclared=True,
        predeclared_heldout_examples=args.predeclared_heldout_examples,
    )
    calibrated_scale.validate()
    if tier != ExperimentTier.SMOKE and not args.natural_test:
        raise ValueError(f"{tier.value} requires --natural-test")
    natural_scale = ExperimentScale(
        tier=tier, heldout_examples=natural_count, training_seeds=training_seeds,
        evaluation_seed=args.seed, predeclared=True,
        predeclared_heldout_examples=args.predeclared_heldout_examples,
    ) if args.natural_test else None
    if natural_scale is not None:
        natural_scale.validate()
    modes = ("final_refine", "middle_recurrent", "profiled_middle_recurrent")
    reports: Dict[str, Dict[int, Any]] = {mode: {} for mode in modes}
    commands: Dict[str, Dict[str, List[str]]] = {mode: {} for mode in modes}
    for training_seed in training_seeds:
        for mode in modes:
            mode_output = output / mode / f"seed_{training_seed}"
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
                "--seed", str(training_seed),
                "--training-seeds", ",".join(str(seed) for seed in training_seeds),
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
            if args.predeclared_heldout_examples is not None:
                command.extend(("--predeclared-heldout-examples", str(args.predeclared_heldout_examples)))
            if args.natural_test:
                command.extend(("--natural-test", args.natural_test))
            if mode == "profiled_middle_recurrent":
                command.extend(("--profile-dir", args.profile_dir))
                if args.profile_stability_dir:
                    command.extend(("--profile-stability-dir", args.profile_stability_dir))
            # Keep the evidence portable: record a repository-relative command while
            # executing with the current interpreter and resolved script path.
            commands[mode][str(training_seed)] = ["python", "scripts/run_e3_hardcase_ablation.py", *command[2:]]
            report_path = mode_output / "e3_hardcase_ablation_report.json"
            if not (args.resume and report_path.exists()):
                subprocess.run(command, cwd=ROOT, check=True)
            reports[mode][training_seed] = json.loads(report_path.read_text())

    selected_steps = {
        mode: {str(seed): int(report["selected_latent_steps"]) for seed, report in by_seed.items()}
        for mode, by_seed in reports.items()
    }
    heldout_rows = []
    for mode, by_seed in reports.items():
        all_pairs = [pair for report in by_seed.values() for pair in report["heldout"]["task_outcomes"]]
        natural_pairs = [
            pair for report in by_seed.values()
            for pair in (report.get("natural_heldout") or {}).get("task_outcomes", [])
        ]
        aggregate = qualify_e3_pairs(
            all_pairs,
            E3QualificationConfig(
                lambda_compute=args.lambda_compute, bootstrap_samples=args.bootstrap_samples,
                confidence=args.confidence, group_key=args.bootstrap_group_key,
                seed=args.seed, experiment_scale=calibrated_scale,
            ),
        )
        aggregate_natural = qualify_e3_pairs(
            natural_pairs,
            E3QualificationConfig(
                lambda_compute=args.lambda_compute, bootstrap_samples=args.bootstrap_samples,
                confidence=args.confidence, group_key=args.bootstrap_group_key,
                seed=args.seed + 1000, experiment_scale=natural_scale,
            ),
        ) if natural_pairs else None
        local_seed_reports = {
            str(seed): {key: value for key, value in qualify_e3_pairs(
                report["heldout"]["task_outcomes"],
                E3QualificationConfig(
                    lambda_compute=args.lambda_compute, bootstrap_samples=args.bootstrap_samples,
                    confidence=args.confidence, group_key=args.bootstrap_group_key, seed=seed,
                ),
            ).items() if key != "paired_records"} for seed, report in by_seed.items()
        }
        local_seed_natural_reports = {
            str(seed): {key: value for key, value in qualify_e3_pairs(
                report["natural_heldout"]["task_outcomes"],
                E3QualificationConfig(
                    lambda_compute=args.lambda_compute, bootstrap_samples=args.bootstrap_samples,
                    confidence=args.confidence, group_key=args.bootstrap_group_key, seed=seed + 1000,
                ),
            ).items() if key != "paired_records"} for seed, report in by_seed.items()
            if report.get("natural_heldout")
        }
        seed_pass_rate = sum(
            bool(local_seed_reports[str(seed)]["qualified"])
            and bool(local_seed_natural_reports.get(str(seed), {}).get("qualified", False))
            for seed in training_seeds
        ) / len(training_seeds)
        source_profile_tier = next(iter(by_seed.values()))["architecture"].get("source_profile_tier") or {}
        heldout = e3_pair_metrics(all_pairs)
        heldout_rows.append({
            "mode": mode,
            "refinement_layer": {str(seed): report["architecture"]["refinement_layer"] for seed, report in by_seed.items()},
            "selected_steps": selected_steps[mode],
            "e2_accuracy": heldout["e2_accuracy"],
            "e3_accuracy": heldout["e3_accuracy"],
            "rescues": heldout["rescue_count"],
            "regressions": heldout["regression_count"],
            "net_rescue_rate": heldout["net_rescue_rate"],
            "e3_ce_delta_vs_e2": sum(report["heldout"]["e3_ce_delta_vs_e2"] for report in by_seed.values()) / len(by_seed),
            "compute_overhead": sum(report["heldout"]["e3_compute_overhead"] for report in by_seed.values()) / len(by_seed),
            "quality_lcb95": aggregate["quality_lcb95"],
            "utility_lcb95": aggregate["utility_lcb95"],
            "compute_delta": aggregate["mean_compute_delta"],
            "qualification_status": aggregate["qualification_status"],
            "natural_qualification_status": aggregate_natural["qualification_status"] if aggregate_natural else "NO_NATURAL_TEST",
            "natural_test_passed": bool(aggregate_natural is not None and aggregate_natural["qualified"]),
            "qualified": bool(
                aggregate["qualified"] and aggregate_natural is not None
                and aggregate_natural["qualified"] and seed_pass_rate >= 2 / 3
            ),
            "seed_pass_rate": seed_pass_rate,
            "seed_replication_passed": seed_pass_rate >= 2 / 3,
            "per_seed": local_seed_reports,
            "natural_per_seed": local_seed_natural_reports,
            "profile_tier": source_profile_tier,
        })
    best = max(
        heldout_rows,
        key=lambda row: (
            row["net_rescue_rate"], row["e3_accuracy"],
            -row["regressions"], -row["e3_ce_delta_vs_e2"],
        ),
    )
    matched_selected_dose = len({step for values in selected_steps.values() for step in values.values()}) == 1
    scale_report = calibrated_scale.validation_report(
        observed_tasks=calibrated_count,
        observed_groups=max(
            int(row["per_seed"][str(training_seeds[0])]["quality_gate"]["bootstrap"]["group_count"])
            for row in heldout_rows
        ),
        observed_training_seeds=training_seeds,
    )
    profile_row = next(row for row in heldout_rows if row["mode"] == "profiled_middle_recurrent")
    profile_tier = profile_row["profile_tier"]
    placement = promote_e3_placement(
        heldout_rows,
        profile_stable=bool((profile_tier.get("profile_stability") or {}).get("stable_for_promotion", False)),
        profile_tier_passed=bool(profile_tier.get("promotion_passed", False)),
        experiment_scale_passed=bool(scale_report["passed"]),
        natural_test_passed=bool(all(
            row["natural_qualification_status"] != "NO_NATURAL_TEST" for row in heldout_rows
        )),
    )
    e3_arm_qualified = bool(matched_selected_dose and placement["promoted"])
    policy_allowed = False
    study = {
        "experiment": "matched-e3-final-heuristic-profiled-location-study",
        "model": {"id": args.model, "revision": args.revision},
        "commands": commands,
        "resumed": args.resume,
        "training_seeds": list(training_seeds),
        "experiment_scale": scale_report,
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
            "placement_promotion": placement,
        },
    }
    (output / "location_study_report.json").write_text(json.dumps(study, indent=2))
    print(json.dumps(study["qualification"], indent=2))


if __name__ == "__main__":
    main()
