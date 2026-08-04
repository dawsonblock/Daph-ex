#!/usr/bin/env python3
"""Run the mandatory oracle-value study before controller training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daph_metareasoner import (
    Action,
    OracleGateConfig,
    confidence_threshold_policy,
    entropy_threshold_policy,
    evaluate_offline_policy,
    fixed_action_policy,
    load_records,
    oracle_value_study,
    prompt_length_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experience", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-oracle-gain", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    records = load_records(args.experience)
    oracle = oracle_value_study(records, OracleGateConfig(
        min_oracle_gain_over_fixed=args.minimum_oracle_gain,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    ))
    baselines = {}
    for action in Action:
        baselines[f"fixed_{action.value.lower()}"] = evaluate_offline_policy(
            records, fixed_action_policy(action.value),
        )
    for action in (Action.THINK, Action.VERIFY, Action.DECOMPOSE):
        for threshold in (0.25, 0.50, 0.75, 0.90):
            baselines[f"confidence_{threshold}_{action.value.lower()}"] = evaluate_offline_policy(
                records, confidence_threshold_policy(threshold, action.value),
            )
        for threshold in (0.5, 1.0, 2.0, 4.0):
            baselines[f"entropy_{threshold}_{action.value.lower()}"] = evaluate_offline_policy(
                records, entropy_threshold_policy(threshold, action.value),
            )
        for threshold in (20, 40, 80, 160):
            baselines[f"length_{threshold}_{action.value.lower()}"] = evaluate_offline_policy(
                records, prompt_length_policy(threshold, action.value),
            )
    report = {
        "oracle": oracle,
        "baselines": baselines,
        "controller_training_allowed": oracle["controller_training_allowed"],
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(oracle, indent=2))


if __name__ == "__main__":
    main()
