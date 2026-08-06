#!/usr/bin/env python3
"""Gate C1 analysis: decompose the v3 failure into independent subsystem ceilings.

Answers four questions before any new implementation is attempted:

  1. How much of the failure is query/bridge inference?
     delta_bridge = Q(oracle_bridge) - Q(two_pass_selected)
  2. How much is retrieval after the correct query is known?
     delta_retrieval = Q(oracle_evidence) - Q(oracle_bridge)
  3. How much is evidence selection / context presentation?
     delta_selection = Q(one_pass_selected) - Q(one_pass)
  4. How much remains when HRM receives perfect evidence?
     reader_error = 1 - Q(oracle_evidence)

Note on terminology: `reader_error` is error relative to perfect task accuracy
under the current model and prompt. It is *not* headroom belonging to any
retrieval component, and must not be added to the retrieval deltas.

Every delta is also cut by family, source style, entity regime, opportunity
group, and answer kind, because the aggregate is no longer sufficient.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SLOT_ECHO = re.compile(r"^\s*\[E\d+\]")
ARMS = ("one_pass", "one_pass_selected", "two_pass_selected",
        "two_pass_calculate", "oracle_bridge", "oracle_evidence")
AXES = ("family", "source_style", "entity_regime", "opportunity_group", "answer_kind")


def load(directory: Path, arm: str) -> dict[str, dict]:
    path = directory / f"{arm}.jsonl"
    if not path.exists():
        return {}
    return {row["task_id"]: row for row in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def grouped_lcb(deltas: list[float], groups: list[str], *, samples: int = 10000,
                seed: int = 42) -> float:
    by_group: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(deltas, groups):
        by_group[group].append(value)
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


def analyse(directory: Path, tasks: dict[str, dict]) -> dict:
    arms = {name: load(directory, name) for name in ARMS}
    arms = {name: rows for name, rows in arms.items() if rows}
    ids = sorted(next(iter(arms.values())))

    def axis_of(task_id: str, axis: str) -> str:
        task = tasks[task_id]
        return task["family"] if axis == "family" else str(task["metadata"][axis])

    def quality(arm: str, subset: list[str] | None = None) -> float | None:
        rows = arms.get(arm)
        if not rows:
            return None
        chosen = subset if subset is not None else ids
        return round(sum(rows[t]["verified_quality"] for t in chosen) / len(chosen), 4)

    def css(arm: str, subset: list[str] | None = None) -> float | None:
        rows = arms.get(arm)
        if not rows:
            return None
        chosen = subset if subset is not None else ids
        return round(sum(rows[t]["complete_set_success"] for t in chosen) / len(chosen), 4)

    def deltas(subset: list[str] | None = None) -> dict:
        q = {name: quality(name, subset) for name in arms}
        out = {}
        if q.get("one_pass") is not None and q.get("one_pass_selected") is not None:
            out["selection"] = round(q["one_pass_selected"] - q["one_pass"], 4)
        if q.get("two_pass_selected") is not None and q.get("one_pass_selected") is not None:
            out["iteration"] = round(q["two_pass_selected"] - q["one_pass_selected"], 4)
        if q.get("oracle_bridge") is not None and q.get("two_pass_selected") is not None:
            out["bridge"] = round(q["oracle_bridge"] - q["two_pass_selected"], 4)
        if q.get("oracle_evidence") is not None and q.get("oracle_bridge") is not None:
            out["retrieval"] = round(q["oracle_evidence"] - q["oracle_bridge"], 4)
        if q.get("oracle_evidence") is not None:
            out["reader_error"] = round(1.0 - q["oracle_evidence"], 4)
        return out

    report = {
        "receipts": str(directory),
        "task_count": len(ids),
        "arms": {name: {"quality": quality(name), "complete_set_success": css(name),
                        "slot_label_echoes": sum(
                            1 for t in ids if SLOT_ECHO.match(arms[name][t]["output"])),
                        "mean_retrieval_calls": round(
                            sum(arms[name][t]["retrieval_calls"] for t in ids) / len(ids), 3)}
                 for name in arms},
        "decomposition": deltas(),
    }

    # Per-axis cuts.
    by_axis: dict[str, dict] = {}
    for axis in AXES:
        buckets: dict[str, list[str]] = defaultdict(list)
        for task_id in ids:
            buckets[axis_of(task_id, axis)].append(task_id)
        by_axis[axis] = {
            name: {"n": len(subset),
                   **{f"q_{arm}": quality(arm, subset) for arm in arms},
                   "delta": deltas(subset)}
            for name, subset in sorted(buckets.items())
        }
    report["by_axis"] = by_axis

    # Statistical view of the headline iteration effect.
    if "two_pass_selected" in arms and "one_pass_selected" in arms:
        paired = [arms["two_pass_selected"][t]["verified_quality"]
                  - arms["one_pass_selected"][t]["verified_quality"] for t in ids]
        report["iteration_lcb95_by_group"] = {
            axis: {
                "lcb95": round(grouped_lcb(paired, [axis_of(t, axis) for t in ids]), 4),
                "group_count": len({axis_of(t, axis) for t in ids}),
                "groups_with_effect": len({
                    axis_of(t, axis) for t, d in zip(ids, paired) if d > 0}),
            }
            for axis in ("family", "source_style", "entity_regime")
        }

    # ---- Gate D: does a per-task oracle trigger beat the best fixed policy? --
    if "two_pass_selected" in arms and "one_pass_selected" in arms:
        one = arms["one_pass_selected"]
        two = arms["two_pass_selected"]
        fixed_one = sum(one[t]["verified_quality"] for t in ids) / len(ids)
        fixed_two = sum(two[t]["verified_quality"] for t in ids) / len(ids)
        oracle = sum(max(one[t]["verified_quality"], two[t]["verified_quality"])
                     for t in ids) / len(ids)
        partition = Counter(
            "positive" if two[t]["verified_quality"] > one[t]["verified_quality"] else
            "negative" if two[t]["verified_quality"] < one[t]["verified_quality"] else
            "neutral" for t in ids)
        by_group: dict[str, Counter] = defaultdict(Counter)
        for task_id in ids:
            delta = two[task_id]["verified_quality"] - one[task_id]["verified_quality"]
            label = "positive" if delta > 0 else "negative" if delta < 0 else "neutral"
            by_group[axis_of(task_id, "opportunity_group")][label] += 1
        best_fixed = max(fixed_one, fixed_two)
        report["gate_d"] = {
            "always_one_pass": round(fixed_one, 4),
            "always_two_pass": round(fixed_two, 4),
            "best_fixed_policy": round(best_fixed, 4),
            "oracle_trigger": round(oracle, 4),
            "opportunity": round(oracle - best_fixed, 4),
            "followup_delta_partition": dict(partition),
            "partition_by_opportunity_group": {k: dict(v) for k, v in sorted(by_group.items())},
            "heterogeneous": partition["positive"] > 0 and partition["negative"] > 0,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", default="evidence/gate_c/v3_qualification")
    parser.add_argument("--ood", default="evidence/gate_c/v3_ood")
    parser.add_argument("--tasks-root", default="data/hrm/controlled_gate_a_v3")
    parser.add_argument("--output", default="evidence/gate_c/v3_analysis.json")
    args = parser.parse_args()

    report = {"gate": "C1_STRUCTURAL_GENERALIZATION", "splits": {}}
    for name, directory in (("qualification", Path(args.qualification)), ("ood", Path(args.ood))):
        if not directory.exists():
            continue
        tasks = {
            row["task_id"]: row for row in (
                json.loads(line) for line in
                (Path(args.tasks_root) / name / "oracle_tasks.jsonl").read_text().splitlines()
                if line.strip())
        }
        report["splits"][name] = analyse(directory, tasks)

    Path(args.output).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    for name, block in report["splits"].items():
        print(f"\n=== {name} (n={block['task_count']})")
        for arm, row in block["arms"].items():
            print(f"  {arm:20} q={row['quality']:.4f} css={row['complete_set_success']:.4f} "
                  f"echoes={row['slot_label_echoes']:3}")
        print(f"  decomposition: {block['decomposition']}")
        if "gate_d" in block:
            print(f"  gate_d: {block['gate_d']['best_fixed_policy']:.4f} fixed vs "
                  f"{block['gate_d']['oracle_trigger']:.4f} oracle "
                  f"-> opportunity {block['gate_d']['opportunity']:+.4f} "
                  f"(heterogeneous={block['gate_d']['heterogeneous']})")


if __name__ == "__main__":
    main()
