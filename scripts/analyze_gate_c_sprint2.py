#!/usr/bin/env python3
"""Analyse the frozen Sprint 2 receipts against the pre-declared Gate C bar.

Reads only committed receipts; it never re-runs the model, so analysis can
never be confused with a change to the measurement. Produces per-task paired
marginal utilities (the eventual executive training set), per-family blocks,
the Gate C failure taxonomy, precision/echo counts, and a verdict computed
from thresholds frozen before the numbers were read.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from collections import Counter

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.evaluation.gate_c_taxonomy import classify_gate_c, summarize_gate_c

SLOT_ECHO = re.compile(r"^\s*\[E\d+\]")
FAMILIES = ("single_hop", "temporal_update", "distractor_heavy", "two_hop", "numeric_derivation")

# Frozen before reading results (roadmap item 17).
THRESHOLDS = {
    "min_overall_quality": 0.90,
    "min_two_hop_complete_set_success": 0.70,
    "max_family_regression": 0.03,
    "require_quality_gain_lcb_positive_under_every_grouping": True,
    "max_mean_retrieval_calls": 2.0,
}


def load(directory: Path, arm: str) -> dict[str, dict]:
    path = directory / f"{arm}.jsonl"
    return {
        row["task_id"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def paired_bootstrap_lcb(
    deltas: list[float], groups: list[str], *, samples: int = 10000, seed: int = 42,
) -> float:
    """Grouped bootstrap lower confidence bound on the mean paired delta."""

    by_group: dict[str, list[float]] = {}
    for value, group in zip(deltas, groups):
        by_group.setdefault(group, []).append(value)
    names = sorted(by_group)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        pooled: list[float] = []
        for _ in names:
            pooled.extend(by_group[names[rng.randrange(len(names))]])
        estimates.append(sum(pooled) / len(pooled))
    estimates.sort()
    return estimates[int(0.05 * len(estimates))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", default="evidence/gate_c/sprint2")
    parser.add_argument("--tasks", default="data/hrm/controlled_gate_a_v2/oracle_tasks.jsonl")
    parser.add_argument("--output", default="evidence/gate_c/sprint2/analysis.json")
    args = parser.parse_args()

    directory = Path(args.receipts)
    manifest = json.loads((directory / "manifest.json").read_text())
    arms = {name: load(directory, name) for name in manifest["arms"]}
    tasks = {
        row["task_id"]: row
        for row in (json.loads(line) for line in Path(args.tasks).read_text().splitlines() if line.strip())
    }

    single, packed = arms["one_pass"], arms["one_pass_selected"]
    two_pass = arms["two_pass_selected"]
    calculated = arms.get("two_pass_calculate", two_pass)

    # ---- per-task paired marginal utilities (future executive training data)
    per_task = []
    for task_id in sorted(single):
        u_single = single[task_id]["verified_quality"]
        u_packed = packed[task_id]["verified_quality"]
        u_two = two_pass[task_id]["verified_quality"]
        u_calc = calculated[task_id]["verified_quality"]
        per_task.append({
            "task_id": task_id,
            "family": single[task_id]["family"],
            "u_single": u_single, "u_packed": u_packed,
            "u_two_pass": u_two, "u_two_pass_calc": u_calc,
            "delta_packing": round(u_packed - u_single, 4),
            "delta_followup": round(u_two - u_packed, 4),
            "delta_calculate": round(u_calc - u_two, 4),
            "delta_combined": round(u_two - u_single, 4),
            "followup_fired": bool(two_pass[task_id]["followup_query"]),
            "retrieval_calls": two_pass[task_id]["retrieval_calls"],
            "answered_by": calculated[task_id]["answered_by"],
        })
    (directory / "per_task_marginal_utility.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_task)
    )

    # ---- conditional-opportunity partition (roadmap item 18)
    fired = [row for row in per_task if row["followup_fired"]]
    partition = Counter(
        "positive" if row["delta_followup"] > 0 else
        "negative" if row["delta_followup"] < 0 else "neutral"
        for row in fired
    )

    # ---- family blocks
    def block(rows: dict[str, dict], family: str | None = None) -> dict:
        chosen = [r for r in rows.values() if family is None or r["family"] == family]
        echoes = sum(1 for r in chosen if SLOT_ECHO.match(r["output"]))
        return {
            "n": len(chosen),
            "quality": round(sum(r["verified_quality"] for r in chosen) / len(chosen), 4),
            "complete_set_success": round(
                sum(r["complete_set_success"] for r in chosen) / len(chosen), 4),
            "mean_records": round(sum(r["evidence_records"] for r in chosen) / len(chosen), 2),
            "mean_evidence_tokens": round(
                sum(r["evidence_tokens"] for r in chosen) / len(chosen), 1),
            "mean_retrieval_calls": round(
                sum(r["retrieval_calls"] for r in chosen) / len(chosen), 3),
            "slot_label_echoes": echoes,
        }

    arm_blocks = {
        name: {"overall": block(rows),
               "per_family": {f: block(rows, f) for f in FAMILIES}}
        for name, rows in arms.items()
    }

    # ---- Gate C taxonomy on the strongest arm
    attributions = []
    for task_id, row in sorted(calculated.items()):
        task = tasks[task_id]
        source = two_pass[task_id]
        attributions.append(classify_gate_c(
            task_id=task_id, family=row["family"], arm="two_pass_calculate",
            quality=row["verified_quality"], answer=row["gold_answer"], output=row["output"],
            required_ids=task["required_evidence_ids"],
            first_pass_ids=source.get("first_pass_ids", source.get("selected_ids", [])),
            second_pass_ids=source.get("second_pass_ids", []),
            merged_ids=source.get("merged_ids", source.get("selected_ids", [])),
            selected_ids=row.get("selected_ids", []),
            followup_query=source["followup_query"],
            bridge_entities=source.get(
                "bridge_entities",
                [source["followup_query"]] if source["followup_query"] else [],
            ),
            calculation=row.get("calculation"),
        ))
    taxonomy = summarize_gate_c(attributions)
    (directory / "gate_c_attribution.jsonl").write_text(
        "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in attributions)
    )

    # ---- verdict against pre-declared thresholds
    # Gate A's established rule: LCB95 > 0 under *every* declared grouping key,
    # and the most conservative view decides. Applying the same rule here.
    deltas = [row["delta_combined"] for row in per_task]
    lcb_by_group = {}
    for key in ("family", "template_id", "source_cluster_id"):
        groups = [tasks[row["task_id"]][key] for row in per_task]
        distinct = sorted(set(groups))
        with_effect = {group for delta, group in zip(deltas, groups) if delta > 0}
        lcb_by_group[key] = {
            "lcb95": round(paired_bootstrap_lcb(deltas, groups), 4),
            "group_count": len(distinct),
            "groups_with_effect": len(with_effect),
        }
    conservative = min(lcb_by_group, key=lambda key: lcb_by_group[key]["lcb95"])
    lcb = lcb_by_group[conservative]["lcb95"]
    best = arm_blocks["two_pass_calculate"]["overall"]
    baseline = arm_blocks["one_pass"]["overall"]
    regressions = {
        family: round(
            arm_blocks["two_pass_calculate"]["per_family"][family]["quality"]
            - arm_blocks["one_pass"]["per_family"][family]["quality"], 4)
        for family in FAMILIES
    }
    worst_regression = min(regressions.values())
    two_hop_css = arm_blocks["two_pass_calculate"]["per_family"]["two_hop"]["complete_set_success"]
    echo_increase = (
        arm_blocks["two_pass_calculate"]["overall"]["slot_label_echoes"]
        - baseline["slot_label_echoes"]
    )

    checks = {
        "overall_quality_at_least_0.90": best["quality"] >= THRESHOLDS["min_overall_quality"],
        "two_hop_complete_set_at_least_0.70": two_hop_css >= THRESHOLDS["min_two_hop_complete_set_success"],
        "no_family_regression_beyond_0.03": worst_regression >= -THRESHOLDS["max_family_regression"],
        "paired_gain_lcb95_positive_under_every_grouping": all(
            row["lcb95"] > 0 for row in lcb_by_group.values()
        ),
        "near_duplicate_confusion_not_increased": echo_increase <= 0,
        "retrieval_cost_bounded": best["mean_retrieval_calls"] <= THRESHOLDS["max_mean_retrieval_calls"],
    }
    passed = all(checks.values())

    report = {
        "gate": "C_ITERATIVE_RETRIEVAL",
        "verdict": "PASS_ITERATIVE_RETRIEVAL" if passed else "FAIL_ITERATIVE_RETRIEVAL",
        "thresholds_frozen_before_reading_results": THRESHOLDS,
        "checks": checks,
        "arms": arm_blocks,
        "marginal_utility": {
            "packing (one_pass_selected - one_pass)": round(
                arm_blocks["one_pass_selected"]["overall"]["quality"] - baseline["quality"], 4),
            "followup (two_pass_selected - one_pass_selected)": round(
                arm_blocks["two_pass_selected"]["overall"]["quality"]
                - arm_blocks["one_pass_selected"]["overall"]["quality"], 4),
            "calculate (two_pass_calculate - two_pass_selected)": round(
                arm_blocks["two_pass_calculate"]["overall"]["quality"]
                - arm_blocks["two_pass_selected"]["overall"]["quality"], 4),
            "combined (two_pass_calculate - one_pass)": round(
                best["quality"] - baseline["quality"], 4),
            "paired_grouped_bootstrap_lcb95_by_group": lcb_by_group,
            "conservative_group_key": conservative,
            "paired_grouped_bootstrap_lcb95": round(lcb, 4),
        },
        "per_family_quality_delta_vs_one_pass": regressions,
        "conditional_opportunity": {
            "followups_fired": len(fired),
            "positive": partition["positive"],
            "neutral": partition["neutral"],
            "negative": partition["negative"],
            "fraction_positive": round(partition["positive"] / max(1, len(fired)), 4),
            "interpretation": (
                "near-universal benefit; a fixed two-pass policy suffices"
                if partition["positive"] / max(1, len(fired)) >= 0.95
                else "heterogeneous benefit; a learned trigger may be justified"
            ),
        },
        "failure_taxonomy": taxonomy,
        "slot_label_echoes_by_arm": {
            name: arm_blocks[name]["overall"]["slot_label_echoes"] for name in arms
        },
        "calculator": {
            "answers_produced": sum(1 for r in per_task if r["answered_by"] == "calculator"),
            "marginal_quality": round(
                arm_blocks["two_pass_calculate"]["overall"]["quality"]
                - arm_blocks["two_pass_selected"]["overall"]["quality"], 4),
        },
    }
    Path(args.output).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("arms",)}, indent=2))


if __name__ == "__main__":
    main()
