#!/usr/bin/env python3
"""Gate C2 candidate-generation policy arms on frozen calibration. No HRM.

Verdicts come from configs/gate_c2_protocol.json, which was frozen before this
script could produce a number. The reference arm, thresholds, and the
alias-report-only rule are read, never chosen here.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evaluation.retrieval_coverage import (
    RetrievalGroundTruth, score_coverage, summarize_coverage)

CAL = Path("data/hrm/controlled_gate_c2_calibration_v1")
PARTITIONS = ("c2_cal_id", "c2_cal_surface")   # holdout deliberately excluded


def rrf(rankings, k, limit):
    s = {}
    for r in rankings:
        for rank, v in enumerate(r, 1):
            s[v] = s.get(v, 0.0) + 1.0 / (k + rank)
    return [v for v, _ in sorted(s.items(), key=lambda i: (-i[1], i[0]))][:limit]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="evidence/gate_c2/policy_arms")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--rrf-k", type=int, default=10)  # frozen negative result
    a = p.parse_args()
    protocol = json.loads((ROOT / "configs" / "gate_c2_protocol.json").read_text())
    assert protocol["frozen_before_any_calibration_run"] is True
    assert "c2_cal_holdout" not in PARTITIONS, "holdout must not be evaluated here"

    report = {"protocol": protocol["protocol"], "reference_arm": protocol["reference_arm"]["name"],
              "rrf_k": a.rrf_k, "k": a.k, "partitions": {}}

    for part in PARTITIONS:
        raw = [json.loads(l) for l in (CAL / part / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
        ev = [json.loads(l) for l in (CAL / part / "evidence.jsonl").read_text().splitlines() if l.strip()]
        truths = {r["task_id"]: RetrievalGroundTruth.from_task(r) for r in raw}
        recs = [IndexRecord(evidence_id=r["evidence_id"], source_id=r["source_id"],
                            content=r["content"], token_count=max(1, len(r["content"].split())),
                            source_type=r["source_type"], metadata=r["metadata"]) for r in ev]
        print(f"\n=== {part} ({len(raw)} tasks) — indexing", flush=True)
        B = {"bm25": CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, recs),
             "bge": CanonicalRetrievalBackend(CanonicalRetrievalMode.DENSE_BGE, recs)}
        hits = {n: {t["task_id"]: [e.evidence_id for e in
                    asyncio.run(B[n].search(t["question"], k=a.k)).evidence] for t in raw}
                for n in B}

        arms = {}
        for name, build in (("P0_bm25", lambda i: hits["bm25"][i]),
                            ("P1_bge", lambda i: hits["bge"][i]),
                            ("P2_rrf_bm25_bge",
                             lambda i: rrf([hits["bm25"][i], hits["bge"][i]], a.rrf_k, a.k))):
            rows = [score_coverage(truths[t["task_id"]], build(t["task_id"]), retriever=name)
                    for t in raw]
            arms[name] = summarize_coverage(rows, truths, retriever=name)
            o = arms[name]["overall"]
            print(f"  {name:18} CES@50={o['complete_set@50']:.3f} PPC@50={o['partial_proof@50']:.3f} "
                  f"identity={o['identity_record_recall_among_identity_tasks']} "
                  f"target|identity={o['target_recall_given_identity_found']}")
        report["partitions"][part] = arms

    # --- verdicts, computed from the frozen protocol -----------------------
    ref = protocol["reference_arm"]["name"]
    rules = protocol["per_regime_rules"]
    verdicts = {}
    for arm in ("P1_bge", "P2_rrf_bm25_bge"):
        checks, deltas = {}, {}
        for part in PARTITIONS:
            for regime, block in report["partitions"][part][arm]["by_axis"]["entity_regime"].items():
                base = report["partitions"][part][ref]["by_axis"]["entity_regime"][regime]
                d = round(block["complete_set@50"] - base["complete_set@50"], 4)
                deltas[regime] = d
                rule = rules[regime]
                if rule["threshold"] is None:
                    checks[regime] = "REPORT_ONLY"
                else:
                    ok = d >= rule["threshold"] if rule["direction"] != "material_gain" else d >= rule["threshold"]
                    checks[regime] = "PASS" if ok else "FAIL"
        binding = {k: v for k, v in checks.items() if v != "REPORT_ONLY"}
        verdicts[arm] = {
            "deltas_vs_reference": deltas, "per_regime_checks": checks,
            "verdict": ("PASS_CANDIDATE_COVERAGE" if all(v == "PASS" for v in binding.values())
                        else "FAIL_CANDIDATE_COVERAGE"),
            "alias_reported_independently": deltas.get("alias"),
        }
    report["verdicts"] = verdicts

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / "policy_arms.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (out / "RESULTS.sha256").write_text(
        f"{hashlib.sha256((out / 'policy_arms.json').read_bytes()).hexdigest()}  policy_arms.json\n")
    print("\n=== verdicts (from the frozen protocol)")
    for arm, v in verdicts.items():
        print(f"  {arm:18} {v['verdict']}")
        print(f"    deltas: {v['deltas_vs_reference']}")
        print(f"    checks: {v['per_regime_checks']}")


if __name__ == "__main__":
    main()
