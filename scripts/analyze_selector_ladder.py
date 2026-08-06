#!/usr/bin/env python3
"""Paired task-level analysis of the Gate C2-S selector ladder.

Comparing arm means hides the shape of an effect: "+0.01 mean" can be 20 tasks
improved and 18 harmed, or 5 improved and 3 harmed. Every arm is therefore
compared to S0 on the SAME tasks, and the uncertainty comes from a grouped
bootstrap over template_id / family / source_cluster_id, never an IID bootstrap
over tasks, because tasks within a template or source cluster are not
independent.

Also answers the mechanism question directly: on tasks where S0 answered
correctly and a reranker did not, which record role did the reranker discard?
If identity and bridge loss dominates, selector-type-versus-accuracy stops being
a correlation and becomes a mechanism.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPING_KEYS = ("source_cluster_id", "template_id", "family")
BOOTSTRAP = 10000
SEED = 20260806


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def grouped_bootstrap_ci(values: dict[str, list[float]], *, resamples: int = BOOTSTRAP,
                         seed: int = SEED) -> tuple[float, float]:
    """Percentile CI resampling whole GROUPS, preserving within-group correlation."""
    keys = sorted(values)
    if not keys:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        drawn: list[float] = []
        for _ in range(len(keys)):
            drawn.extend(values[keys[rng.randrange(len(keys))]])
        if drawn:
            means.append(sum(drawn) / len(drawn))
    means.sort()
    if not means:
        return (float("nan"), float("nan"))
    return (round(means[int(0.025 * len(means))], 4),
            round(means[min(len(means) - 1, int(0.975 * len(means)))], 4))


def paired_delta(arm_rows: dict[str, dict], base_rows: dict[str, dict], field: str,
                 eligible=lambda row: True) -> dict:
    """Paired per-task deltas on tasks eligible under BOTH arms."""
    per_group: dict[str, dict[str, list[float]]] = {k: defaultdict(list) for k in GROUPING_KEYS}
    deltas: list[float] = []
    positive = negative = neutral = 0
    for task_id, arm_row in arm_rows.items():
        base = base_rows.get(task_id)
        if base is None or not eligible(arm_row) or not eligible(base):
            continue
        delta = float(arm_row[field]) - float(base[field])
        deltas.append(delta)
        if delta > 0:
            positive += 1
        elif delta < 0:
            negative += 1
        else:
            neutral += 1
        for key in GROUPING_KEYS:
            per_group[key][str(arm_row.get(key))].append(delta)
    if not deltas:
        return {"paired_tasks": 0}
    cis = {key: grouped_bootstrap_ci(per_group[key]) for key in GROUPING_KEYS}
    # The most conservative grouping governs the verdict.
    widest = max(cis, key=lambda k: cis[k][1] - cis[k][0])
    return {
        "paired_tasks": len(deltas),
        "mean_delta": round(sum(deltas) / len(deltas), 4),
        "positive_tasks": positive, "negative_tasks": negative, "neutral_tasks": neutral,
        "ci95_by_grouping": {k: list(v) for k, v in cis.items()},
        "groups_by_grouping": {k: len(per_group[k]) for k in GROUPING_KEYS},
        "governing_grouping": widest,
        "governing_groups": len(per_group[widest]),
        "ci95": list(cis[widest]),
        "excludes_zero": bool(cis[widest][0] > 0 or cis[widest][1] < 0),
    }


def discard_analysis(arm_rows: dict[str, dict], base_rows: dict[str, dict]) -> dict:
    """On tasks S0 got right and this arm got wrong, what did the arm discard?"""
    dropped = defaultdict(int)
    regressions = 0
    also_dropped_nothing = 0
    for task_id, arm_row in arm_rows.items():
        base = base_rows.get(task_id)
        if base is None or not (base["quality"] > arm_row["quality"]):
            continue
        regressions += 1
        # Roles the arm lost that S0 had kept: the causal candidates.
        base_dropped = set(base["roles_dropped"])
        lost = [r for r in arm_row["roles_dropped"] if r not in base_dropped]
        if not lost:
            also_dropped_nothing += 1
        for role in lost:
            dropped[role] += 1
    return {
        "regressions_vs_s0": regressions,
        "role_lost_that_s0_kept": dict(sorted(dropped.items(), key=lambda kv: -kv[1])),
        "regressions_with_no_role_loss": also_dropped_nothing,
        "note": ("A regression with no role loss cannot be explained by discarding a "
                 "required record; it is packing order or distractor composition."),
    }


def flip_analysis(arm_rows: dict[str, dict], base_rows: dict[str, dict]) -> dict:
    """2x2 contingency against S0, plus what was lost on each correct->wrong flip.

    Verified quality is strictly binary on this corpus (0.0/1.0, agreeing with
    exact_match on every row), so "correct" is unambiguous.
    """
    both_correct = both_wrong = s0_only = arm_only = 0
    attribution: dict[str, int] = defaultdict(int)
    for task_id, arm_row in arm_rows.items():
        base = base_rows.get(task_id)
        if base is None:
            continue
        base_ok, arm_ok = base["quality"] == 1.0, arm_row["quality"] == 1.0
        if base_ok and arm_ok:
            both_correct += 1
        elif not base_ok and not arm_ok:
            both_wrong += 1
        elif base_ok and not arm_ok:
            s0_only += 1
            # Roles this arm lost that S0 had retained.
            lost = [role for role in ("identity", "bridge", "answer")
                    if base["role_retained"].get(role) and not arm_row["role_retained"].get(role)]
            for role in lost:
                attribution[f"{role}_lost"] += 1
            if not lost:
                if arm_row["distractors"] > base["distractors"]:
                    attribution["extra_distractor_introduced"] += 1
                else:
                    attribution["no_role_loss_no_extra_distractor"] += 1
        else:
            arm_only += 1
    return {
        "both_correct": both_correct, "both_wrong": both_wrong,
        "s0_correct_arm_wrong": s0_only, "s0_wrong_arm_correct": arm_only,
        "net": arm_only - s0_only,
        "loss_attribution_on_correct_to_wrong": dict(
            sorted(attribution.items(), key=lambda kv: -kv[1])),
    }


def gain_attribution(arm_rows: dict[str, dict], base_rows: dict[str, dict]) -> dict:
    """On tasks the arm FIXED, what did it newly retain?

    This is the test of whether an arm wins for the mechanism it claims. The
    intended pattern for a structural selector is bridge and answer recovered
    jointly. A win concentrated in "no new role retained" would mean the arm
    improved by some other route -- packet composition or ordering -- and the
    structural story would not be established.
    """
    pattern: dict[str, int] = defaultdict(int)
    fixed = 0
    for task_id, arm_row in arm_rows.items():
        base = base_rows.get(task_id)
        if base is None or not (arm_row["quality"] == 1.0 and base["quality"] == 0.0):
            continue
        fixed += 1
        gained = [role for role in ("identity", "bridge", "answer")
                  if arm_row["role_retained"].get(role) and not base["role_retained"].get(role)]
        for role in gained:
            pattern[f"{role}_newly_retained"] += 1
        if {"bridge", "answer"} <= set(gained):
            pattern["bridge_and_answer_jointly_recovered"] += 1
        if arm_row["distractors"] < base["distractors"]:
            pattern["distractors_reduced"] += 1
        if not gained:
            pattern["no_new_role_retained"] += 1
    return {"tasks_fixed_by_arm": fixed,
            "gain_pattern": dict(sorted(pattern.items(), key=lambda kv: -kv[1]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evidence/gate_c2/selector_ladder")
    parser.add_argument("--budget", type=int, default=6)
    args = parser.parse_args()

    in_dir = ROOT / args.input
    rows = load_rows(in_dir / "tasks.jsonl")
    cells = load_rows(in_dir / "cells.jsonl")

    # ConnectedProofRetention is recomputed here from each row's stored selection
    # rather than read from the cell, so arms measured before the metric existed
    # are covered without re-running the reader.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "_runner", ROOT / "scripts/run_selector_ladder.py")
    _runner = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_runner)
    corpus_root = ROOT / "data/hrm/controlled_gate_c2_description_valid_v4"
    pool_root = ROOT / "evidence/gate_c2/candidate_pools/v1"
    meta_by_task: dict[str, dict] = {}
    pool_by_task: dict[str, set[str]] = {}
    for part in sorted({r["partition"] for r in rows}):
        task_file = corpus_root / part / "oracle_tasks.jsonl"
        if task_file.exists():
            for row in load_rows(task_file):
                meta_by_task[row["task_id"]] = row["_oracle_metadata"]
        pool_file = pool_root / f"{part}.jsonl"
        if pool_file.exists():
            for row in load_rows(pool_file):
                pool_by_task[row["task_id"]] = {c["document_id"] for c in row["candidates"]}
    for row in rows:
        meta = meta_by_task.get(row["task_id"])
        pool = pool_by_task.get(row["task_id"])
        row["connected_proof_retained"] = (
            _runner.connected_proof_retained(meta, set(row["selected"]), pool)
            if meta and pool else None)

    for row in rows:
        # Numeric views of the role flags so retention can be paired like quality.
        for role in ("answer", "bridge", "identity"):
            flag = row["role_retained"].get(role)
            row[f"_{role[:3]}"] = None if flag is None else float(bool(flag))
    report: dict = {"budget": args.budget, "bootstrap_resamples": BOOTSTRAP, "seed": SEED,
                    "grouping_keys": list(GROUPING_KEYS), "partitions": {}}

    for part in sorted({r["partition"] for r in rows}):
        by_arm: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            if row["partition"] == part and row["budget"] == args.budget:
                by_arm[row["arm"]][row["task_id"]] = row
        if "S0_raw" not in by_arm:
            continue
        base = by_arm["S0_raw"]
        cell_by_arm = {c["arm"]: c for c in cells
                       if c["partition"] == part and c["budget"] == args.budget}
        arms: dict = {}
        for arm in sorted(by_arm):
            cpr_vals = [r["connected_proof_retained"] for r in by_arm[arm].values()
                        if r["connected_proof_retained"] is not None]
            entry: dict = {"connected_proof_retention_recomputed": (
                round(sum(cpr_vals) / len(cpr_vals), 4) if cpr_vals else None),
                "cpr_eligible_tasks": len(cpr_vals), "aggregate": {
                k: cell_by_arm.get(arm, {}).get(k) for k in
                ("quality", "CSR_given_complete_set_available", "IdentityRetention",
                 "BridgeRetention", "AnswerRetention", "GoldDensity", "DistractorCount",
                 "SelectedTokens", "differs_from_s0")}}
            if arm != "S0_raw":
                entry["paired_quality"] = paired_delta(by_arm[arm], base, "quality")
                entry["paired_csr"] = paired_delta(
                    by_arm[arm], base, "csr_ok", eligible=lambda r: r["csr_eligible"])
                for role, short in (("answer", "ans"), ("bridge", "bri"), ("identity", "ide")):
                    entry[f"paired_{role}_retention"] = paired_delta(
                        by_arm[arm], base, f"_{short}",
                        eligible=lambda r, s=short: r[f"_{s}"] is not None)
                entry["discards"] = discard_analysis(by_arm[arm], base)
                entry["flips"] = flip_analysis(by_arm[arm], base)
                entry["gains"] = gain_attribution(by_arm[arm], base)
            arms[arm] = entry
        # S5 headroom is what says whether selection is the bottleneck at all.
        s0_q = cell_by_arm.get("S0_raw", {}).get("quality")
        s5_q = cell_by_arm.get("S5_oracle", {}).get("quality")
        headroom = None
        if s0_q is not None and s5_q is not None:
            headroom = {
                "s0_quality": s0_q, "s5_quality": s5_q,
                "absolute_opportunity": round(s5_q - s0_q, 4),
                "recovery_fraction": {
                    a: (round((arms[a]["aggregate"]["quality"] - s0_q) / (s5_q - s0_q), 4)
                        if s5_q > s0_q and arms[a]["aggregate"]["quality"] is not None else None)
                    for a in arms if a not in ("S0_raw", "S5_oracle")},
            }
        ceilings = None
        if s0_q is not None and s5_q is not None:
            ceilings = {
                "selector_ceiling_within_retrieved_pool": s5_q,
                "baseline": s0_q,
                "selector_opportunity": round(s5_q - s0_q, 4),
                "reader_interface_residual_under_perfect_selection": round(1.0 - s5_q, 4),
                "note": ("S5 is the ceiling for SELECTION over the frozen pool. It is not the "
                         "pipeline ceiling: the residual above it is reader/interface/task "
                         "difficulty, and a separate candidate-pool ceiling sits upstream. "
                         "R5-style oracle-evidence arms bypass more of the pipeline than S5 "
                         "and must not be substituted for this number."),
            }
        report["partitions"][part] = {"arms": arms, "s5_headroom": headroom,
                                      "ceilings": ceilings}

    for part, payload in report["partitions"].items():
        print(f"\n=== MAIN SELECTOR TABLE  {part}  budget={args.budget}")
        print(f"{'Arm':26} {'Ident':>7} {'Bridge':>7} {'Answer':>7} {'CPR':>7} "
              f"{'CSR':>7} {'Q':>7}")
        for arm, entry in payload["arms"].items():
            agg = entry["aggregate"]
            def fmt(key):
                value = agg.get(key)
                return f"{value:7.4f}" if isinstance(value, (int, float)) else f"{'n/a':>7}"
            print(f"{arm:26} {fmt('IdentityRetention')} {fmt('BridgeRetention')} "
                  f"{fmt('AnswerRetention')} "
                  f"{entry['connected_proof_retention_recomputed']:7.4f} "
                  f"{fmt('CSR_given_complete_set_available')} {fmt('quality')}")

    out = in_dir / "paired_analysis.json"
    out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"wrote {out}")
    for part, payload in report["partitions"].items():
        print(f"\n=== {part} budget={args.budget}")
        head = payload["s5_headroom"]
        if head:
            print(f"  S5 opportunity: {head['s0_quality']:.4f} -> {head['s5_quality']:.4f} "
                  f"(+{head['absolute_opportunity']:.4f})")
        for arm, entry in payload["arms"].items():
            if arm == "S0_raw":
                continue
            pq = entry.get("paired_quality", {})
            if not pq.get("paired_tasks"):
                continue
            print(f"  {arm:24} dQ={pq['mean_delta']:+.4f} CI{pq['ci95']} "
                  f"(+{pq['positive_tasks']}/-{pq['negative_tasks']}/={pq['neutral_tasks']}) "
                  f"{'SIGNIFICANT' if pq['excludes_zero'] else 'indistinguishable'}")
            flips = entry.get("flips", {})
            if flips:
                print(f"      flips: both_ok={flips['both_correct']} "
                      f"S0_only={flips['s0_correct_arm_wrong']} "
                      f"arm_only={flips['s0_wrong_arm_correct']} "
                      f"both_wrong={flips['both_wrong']} net={flips['net']:+d}")
                if flips["loss_attribution_on_correct_to_wrong"]:
                    print(f"      losses: {flips['loss_attribution_on_correct_to_wrong']}")
            gains = entry.get("gains", {})
            if gains.get("tasks_fixed_by_arm"):
                print(f"      fixed={gains['tasks_fixed_by_arm']} "
                      f"gains={gains['gain_pattern']}")
            head = payload.get("s5_headroom") or {}
            frac = (head.get("recovery_fraction") or {}).get(arm)
            if frac is not None:
                print(f"      SGC (share of the S5 opportunity captured): {frac:+.1%}")
            disc = entry.get("discards", {})
            if disc.get("regressions_vs_s0"):
                print(f"      regressions={disc['regressions_vs_s0']} "
                      f"roles_lost={disc['role_lost_that_s0_kept']} "
                      f"no_role_loss={disc['regressions_with_no_role_loss']}")


if __name__ == "__main__":
    main()
