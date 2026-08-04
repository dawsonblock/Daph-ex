#!/usr/bin/env python3
"""Train hidden and cheap-sham VOC controllers behind predeclared gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph_metareasoner import (
    ConservativeVOCPolicy,
    OracleGateConfig,
    PolicyConfig,
    ProbeTrainingConfig,
    StateVectorizer,
    UtilityConfig,
    ValueTrainingConfig,
    evaluate_offline_policy,
    entropy_threshold_policy,
    fit_family_lookup_policy,
    fixed_action_policy,
    frequency_matched_random_policy,
    load_records,
    oracle_capture,
    oracle_value_study,
    paired_policy_gate,
    predictor_policy,
    probe_signal_gate,
    confidence_threshold_policy,
    prompt_length_policy,
    records_digest,
    train_probe,
    train_value_ensemble,
    training_digest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--ood")
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-spec", default="hidden_runtime")
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--controller-epochs", type=int, default=150)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--uncertainty-beta", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--research-override", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = load_records(args.train)
    validation = load_records(args.validation)
    test = load_records(args.test)
    ood = load_records(args.ood) if args.ood else []
    named_splits = (("experience", train), ("validation", validation), ("test", test), ("ood", ood))
    seen_tasks = set()
    model_digests = set()
    for expected_split, split_records in named_splits:
        if not split_records:
            continue
        actual = {record.task.split for record in split_records}
        if actual != {expected_split}:
            raise ValueError(f"Expected {expected_split} records, found splits {sorted(actual)}")
        task_ids = {record.task.task_id for record in split_records}
        overlap = seen_tasks & task_ids
        if overlap:
            raise ValueError(f"Task leakage across controller splits: {sorted(overlap)[:5]}")
        seen_tasks.update(task_ids)
        model_digests.update(record.model_digest for record in split_records)
    if len(model_digests) != 1:
        raise ValueError(f"Controller records must share one base model digest: {sorted(model_digests)}")
    base_model_digest = next(iter(model_digests))
    train_families = {record.task.family_id for record in train + validation}
    ood_families = {record.task.family_id for record in ood}
    if train_families & ood_families:
        raise ValueError("OOD records overlap train/validation task families")
    oracle_train = oracle_value_study(train, OracleGateConfig(
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    ))
    probe_config = ProbeTrainingConfig(epochs=args.probe_epochs, seed=args.seed)
    hidden_vectorizer = StateVectorizer(args.feature_spec)
    cheap_vectorizer = StateVectorizer("cheap")
    _, _, hidden_probe_metrics = train_probe(train, validation, hidden_vectorizer, probe_config)
    _, _, cheap_probe_metrics = train_probe(train, validation, cheap_vectorizer, probe_config)
    probe_gate = probe_signal_gate(hidden_probe_metrics, cheap_probe_metrics)
    authorized = bool(
        oracle_train["controller_training_allowed"] and probe_gate["value_controller_training_allowed"]
    )
    if not authorized and not args.research_override:
        report = {
            "status": "BLOCKED_BEFORE_VALUE_CONTROLLER",
            "oracle_train": oracle_train,
            "hidden_probe": hidden_probe_metrics,
            "cheap_probe": cheap_probe_metrics,
            "probe_gate": probe_gate,
            "research_override": False,
        }
        (output / "controller_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    value_config = ValueTrainingConfig(
        epochs=args.controller_epochs,
        ensemble_size=args.ensemble_size,
        seed=args.seed,
    )
    hidden_ensemble = train_value_ensemble(train, hidden_vectorizer, value_config)
    sham_ensemble = train_value_ensemble(train, cheap_vectorizer, value_config)
    utility = UtilityConfig()
    policy_config = PolicyConfig(uncertainty_beta=args.uncertainty_beta)
    learned = predictor_policy(ConservativeVOCPolicy(hidden_ensemble, utility, policy_config))
    sham = predictor_policy(ConservativeVOCPolicy(sham_ensemble, utility, policy_config))
    validation_oracle = oracle_value_study(validation)
    fixed = fixed_action_policy(validation_oracle["best_fixed_action"])
    family_lookup = fit_family_lookup_policy(
        train, fallback_action=validation_oracle["best_fixed_action"],
    )
    heuristic_candidates = {}
    for action in ("THINK", "VERIFY", "DECOMPOSE"):
        for threshold in (0.25, 0.50, 0.75, 0.90):
            heuristic_candidates[f"confidence_{threshold}_{action}"] = confidence_threshold_policy(threshold, action)
        for threshold in (0.5, 1.0, 2.0, 4.0):
            heuristic_candidates[f"entropy_{threshold}_{action}"] = entropy_threshold_policy(threshold, action)
        for threshold in (20, 40, 80, 160):
            heuristic_candidates[f"length_{threshold}_{action}"] = prompt_length_policy(threshold, action)
    heuristic_name, heuristic = max(
        heuristic_candidates.items(),
        key=lambda item: evaluate_offline_policy(validation, item[1])["mean_utility"],
    )

    def evaluate_split(records, seed_offset):
        learned_metrics = evaluate_offline_policy(records, learned)
        sham_metrics = evaluate_offline_policy(records, sham)
        fixed_metrics = evaluate_offline_policy(records, fixed)
        heuristic_metrics = evaluate_offline_policy(records, heuristic)
        family_metrics = evaluate_offline_policy(records, family_lookup)
        oracle = oracle_value_study(records)
        learned_vs_fixed = paired_policy_gate(
            records, learned, fixed, confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed + seed_offset + 1,
        )
        learned_frequency = learned_vs_fixed["learned_action_frequency"]
        random_policy = frequency_matched_random_policy(
            learned_frequency, seed=args.seed + seed_offset + 2,
        )
        return {
            "learned": learned_metrics,
            "sham": sham_metrics,
            "fixed": fixed_metrics,
            "heuristic": heuristic_metrics,
            "family_lookup": family_metrics,
            "frequency_matched_random": evaluate_offline_policy(records, random_policy),
            "oracle": oracle,
            "learned_vs_sham": paired_policy_gate(
                records, learned, sham, confidence=args.confidence,
                bootstrap_samples=args.bootstrap_samples, seed=args.seed + seed_offset,
            ),
            "learned_vs_fixed": learned_vs_fixed,
            "learned_vs_heuristic": paired_policy_gate(
                records, learned, heuristic, confidence=args.confidence,
                bootstrap_samples=args.bootstrap_samples, seed=args.seed + seed_offset + 3,
            ),
            "learned_vs_family_lookup": paired_policy_gate(
                records, learned, family_lookup, confidence=args.confidence,
                bootstrap_samples=args.bootstrap_samples, seed=args.seed + seed_offset + 4,
            ),
            "learned_vs_frequency_random": paired_policy_gate(
                records, learned,
                frequency_matched_random_policy(learned_frequency, seed=args.seed + seed_offset + 5),
                confidence=args.confidence,
                bootstrap_samples=args.bootstrap_samples, seed=args.seed + seed_offset + 6,
            ),
            "oracle_capture": oracle_capture(
                learned_metrics["mean_utility"], fixed_metrics["mean_utility"],
                oracle["oracle_utility"],
            ),
        }

    test_report = evaluate_split(test, 100)
    ood_report = evaluate_split(ood, 200) if ood else None
    iid_pass = (
        test_report["learned_vs_sham"]["qualified"]
        and test_report["learned_vs_fixed"]["qualified"]
        and test_report["learned_vs_heuristic"]["qualified"]
        and test_report["learned_vs_family_lookup"]["qualified"]
        and test_report["learned_vs_frequency_random"]["qualified"]
        and test_report["oracle_capture"] > 0.25
    )
    ood_pass = bool(
        ood_report is None
        or (
            ood_report["learned_vs_sham"]["qualified"]
            and ood_report["learned_vs_fixed"]["qualified"]
            and ood_report["learned_vs_heuristic"]["qualified"]
            and ood_report["learned_vs_family_lookup"]["qualified"]
            and ood_report["learned_vs_frequency_random"]["qualified"]
        )
    )
    verified = bool(authorized and iid_pass and ood_pass and not args.research_override)
    status = "VERIFIED_FIT" if verified else "UNVERIFIED_FIT"
    report = {
        "status": status,
        "oracle_train": oracle_train,
        "hidden_probe": hidden_probe_metrics,
        "cheap_probe": cheap_probe_metrics,
        "probe_gate": probe_gate,
        "validation_fixed_action": validation_oracle["best_fixed_action"],
        "validation_heuristic": heuristic_name,
        "test": test_report,
        "ood": ood_report,
        "iid_gate_passed": iid_pass,
        "ood_gate_passed": ood_pass,
        "research_override": args.research_override,
        "digests": {
            "train": records_digest(train),
            "validation": records_digest(validation),
            "test": records_digest(test),
            "ood": records_digest(ood) if ood else "",
        },
    }
    report_text = json.dumps(report, indent=2)
    report_path = output / "controller_report.json"
    report_path.write_text(report_text)
    evaluation_digest = records_digest(test + ood)
    hidden_ensemble.save(
        output / "hidden_controller.pt",
        training_digest=training_digest(train), config=value_config,
        training_status=status, evaluation_digest=evaluation_digest,
        base_model_digest=base_model_digest,
    )
    sham_ensemble.save(
        output / "sham_controller.pt",
        training_digest=training_digest(train), config=value_config,
        training_status="CONTROL_FIT", evaluation_digest=evaluation_digest,
        base_model_digest=base_model_digest,
    )
    print(report_text)


if __name__ == "__main__":
    main()
