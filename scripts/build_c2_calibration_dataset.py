#!/usr/bin/env python3
"""Build, audit, and freeze controlled_gate_c2_calibration_v1. Fail-closed.

generate -> V4 separation audit -> inferability -> leakage -> oracle isolation
-> regime balance -> pytest -> hashes -> freeze. Nothing is written unless every
stage passes.
"""
from __future__ import annotations
import hashlib, json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from hrm_adaptive_memory.experiments.c2_calibration_dataset import (
    ANSWER_BEARING, HEADS, PARTITIONS, build_calibration, leaks)
from hrm_adaptive_memory.experiments.generalization_dataset_v4 import (
    _HEADS as V4_HEADS, verify_inferable)

SPLITS = {"c2_cal_id": (51001, 50), "c2_cal_surface": (51002, 50), "c2_cal_holdout": (51003, 25)}
FORBIDDEN = re.compile(r"#(s|b|v|d\d|n)\b|latent_|entity_\d+")


def main() -> None:
    out = Path("data/hrm/controlled_gate_c2_calibration_v1")
    if out.exists():
        raise FileExistsError(f"frozen dataset exists: {out}")

    print("[1/6] generating")
    built = {}
    for name, (seed, per) in SPLITS.items():
        built[name] = build_calibration(seed=seed, partition=name, per_regime=per)
        m = built[name]["manifest"]
        print(f"      {name}: {m['task_count']} tasks, {m['evidence_count']} records, "
              f"regimes={m['regimes']}")

    print("[2/6] auditing")
    problems: list[str] = []

    # V4 separation — the whole point of a separate calibration corpus.
    overlap = set(HEADS) & set(V4_HEADS)
    if overlap:
        problems.append(f"entity heads overlap V4: {sorted(overlap)}")
    v4_ids = set()
    for split in ("development", "qualification", "ood"):
        p = ROOT / "data/hrm/controlled_gate_a_v4" / split / "evidence.jsonl"
        if p.exists():
            v4_ids |= {json.loads(l)["evidence_id"] for l in p.read_text().splitlines() if l.strip()}
    for name, c in built.items():
        ids = {r["evidence_id"] for r in c["evidence"]}
        if ids & v4_ids:
            problems.append(f"{name}: evidence ids collide with V4")
        if not all(t["task_id"].startswith("c2cal-") for t in c["tasks"]):
            problems.append(f"{name}: task ids not namespaced")

    audit: dict = {}
    for name, c in built.items():
        by_id = {r["evidence_id"]: r for r in c["evidence"]}
        bad = {t["task_id"]: verify_inferable(t, by_id) for t in c["tasks"]}
        bad = {k: v for k, v in bad.items() if v}
        qleak = sum(1 for t in c["tasks"] if leaks(t["question"], t["answer"]))
        dleak = sum(1 for t in c["tasks"] for r in c["evidence"]
                    if r["evidence_id"].startswith(t["task_id"] + "/")
                    and r["metadata"]["record_kind"] not in ANSWER_BEARING
                    and leaks(r["content"], t["answer"]))
        latent = sum(1 for r in c["evidence"] if FORBIDDEN.search(r["content"]))
        latent += sum(1 for t in c["tasks"] if FORBIDDEN.search(t["question"]))
        regimes = {}
        for t in c["tasks"]:
            regimes[t["metadata"]["entity_regime"]] = regimes.get(t["metadata"]["entity_regime"], 0) + 1
        audit[name] = {"not_inferable": len(bad), "question_leaks": qleak,
                       "distractor_leaks": dleak, "latent_id_leaks": latent,
                       "regime_counts": regimes,
                       "proof_graphs": all("_oracle_metadata" in t for t in c["tasks"])}
        for key in ("not_inferable", "question_leaks", "distractor_leaks", "latent_id_leaks"):
            if audit[name][key]:
                problems.append(f"{name}: {key}={audit[name][key]}")
        if not audit[name]["proof_graphs"]:
            problems.append(f"{name}: missing proof graphs")
        if len(set(regimes.values())) > 1:
            problems.append(f"{name}: regimes unbalanced {regimes}")

    for line in problems:
        print(f"      FAIL {line}")
    if problems:
        raise SystemExit("VALID_C2_CAL false; nothing written")
    print("      VALID_C2_CAL = true")

    print("[3/6] pytest")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-1500:]); raise SystemExit("pytest failed; nothing written")
    print(f"      {r.stdout.strip().splitlines()[-1]}")

    print("[4/6] writing")
    out.mkdir(parents=True)
    for name, c in built.items():
        d = out / name; d.mkdir()
        (d / "oracle_tasks.jsonl").write_text(
            "".join(json.dumps(t, sort_keys=True) + "\n" for t in c["tasks"]))
        (d / "evidence.jsonl").write_text(
            "".join(json.dumps(e, sort_keys=True) + "\n" for e in c["evidence"]))
        (d / "dataset_manifest.json").write_text(
            json.dumps(c["manifest"], sort_keys=True, indent=2) + "\n")

    print("[5/6] hashing")
    digests = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "AUDIT.json").write_text(json.dumps({
        "generator": "controlled-gate-c2-calibration-v1",
        "purpose": "Gate C2 component selection. Does NOT replace V4, which stays immutable.",
        "partitions": {k: built[k]["manifest"] for k in built},
        "audit": audit, "problems": problems, "VALID_C2_CAL": True,
        "v4_separation": {"entity_head_overlap": sorted(set(HEADS) & set(V4_HEADS)),
                          "evidence_id_collisions": 0,
                          "task_id_prefix": "c2cal-"},
        "holdout_policy": ("c2_cal_holdout is RESERVED. Selecting on it would burn it exactly as "
                           "using V4 OOD for architecture selection burned that split."),
        "frozen_before_evaluation": True,
    }, sort_keys=True, indent=2) + "\n")
    (out / "RECEIPTS.sha256").write_text(
        "".join(f"{v}  {k}\n" for k, v in sorted(digests.items())))
    print("[6/6] frozen")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
