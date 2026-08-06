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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evidence/gate_c2/selector_ladder")
    parser.add_argument("--budget", type=int, default=6)
    args = parser.parse_args()

    in_dir = ROOT / args.input
    rows = load_rows(in_dir / "tasks.jsonl")
    cells = load_rows(in_dir / "cells.jsonl")
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
            entry: dict = {"aggregate": {
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
        report["partitions"][part] = {"arms": arms, "s5_headroom": headroom}

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
            disc = entry.get("discards", {})
            if disc.get("regressions_vs_s0"):
                print(f"      regressions={disc['regressions_vs_s0']} "
                      f"roles_lost={disc['role_lost_that_s0_kept']} "
                      f"no_role_loss={disc['regressions_with_no_role_loss']}")


if __name__ == "__main__":
    main()
