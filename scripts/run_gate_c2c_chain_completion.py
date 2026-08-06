#!/usr/bin/env python3
"""Gate C2-C chain-completion ladder on frozen calibration. No HRM.

C1-C4 canonicalize by PARSING runtime-visible candidates. Only C5/C6 read
_oracle_metadata, and they are ceilings.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evaluation.retrieval_coverage import (
    RetrievalGroundTruth, score_coverage, summarize_coverage)
from hrm_adaptive_memory.retrieval.canonicalization import resolve_canonical
from hrm_adaptive_memory.retrieval.information_state import InformationState, formulate_followup

CAL = Path("data/hrm/controlled_gate_c2_calibration_v1")
PARTS = ("c2_cal_id", "c2_cal_surface")
K = 50


def rrf(rankings, k, limit):
    s = {}
    for r in rankings:
        for rank, v in enumerate(r, 1):
            s[v] = s.get(v, 0.0) + 1.0 / (k + rank)
    return [v for v, _ in sorted(s.items(), key=lambda i: (-i[1], i[0]))][:limit]


def interleave(rankings, limit):
    out, seen = [], set()
    for pos in range(max(len(r) for r in rankings)):
        for r in rankings:
            if pos < len(r) and r[pos] not in seen:
                seen.add(r[pos]); out.append(r[pos]); 
                if len(out) >= limit: return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="evidence/gate_c2/chain_completion")
    a = ap.parse_args()
    proto = json.loads((ROOT / "configs" / "gate_c2c_protocol.json").read_text())
    assert proto["frozen_before_any_run"] is True

    report = {"protocol": proto["protocol"], "reference": proto["reference_arm"]["name"],
              "partitions": {}}
    for part in PARTS:
        raw = [json.loads(l) for l in (CAL / part / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
        ev = [json.loads(l) for l in (CAL / part / "evidence.jsonl").read_text().splitlines() if l.strip()]
        truths = {r["task_id"]: RetrievalGroundTruth.from_task(r) for r in raw}
        recs = [IndexRecord(evidence_id=r["evidence_id"], source_id=r["source_id"],
                            content=r["content"], token_count=max(1, len(r["content"].split())),
                            source_type=r["source_type"], metadata=r["metadata"]) for r in ev]
        index = {r.evidence_id: r for r in recs}
        print(f"\n=== {part} ({len(raw)}) indexing", flush=True)
        bm, bg = (CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, recs),
                  CanonicalRetrievalBackend(CanonicalRetrievalMode.DENSE_BGE, recs))
        def fuse(q):
            return rrf([[e.evidence_id for e in asyncio.run(bm.search(q, k=K)).evidence],
                        [e.evidence_id for e in asyncio.run(bg.search(q, k=K)).evidence]], 10, K)

        arms, depths = {}, {}
        base = {t["task_id"]: fuse(t["question"]) for t in raw}

        def run(name, build):
            rows, d = [], []
            for t in raw:
                ids, hops = build(t)
                rows.append(score_coverage(truths[t["task_id"]], ids, retriever=name))
                d.append(hops)
            arms[name] = summarize_coverage(rows, truths, retriever=name)
            depths[name] = round(sum(d) / len(d), 3)
            o = arms[name]["overall"]
            print(f"  {name:30} CES@50={o['complete_set@50']:.3f} "
                  f"tgt|id={o['target_recall_given_identity_found']} depth={depths[name]}")

        def state_of(t, pool):
            meta = t["_oracle_metadata"]
            subj = meta["surfaces"]["subject"]
            st = InformationState(subject=subj, target_relation=meta["target_relation"])
            link = resolve_canonical(subj, [index[i] for i in pool if i in index])
            if link:
                st = st.with_identity(link.surface, link.canonical, record_id=link.record_id)
            return st, link

        run("C0_p2_fusion", lambda t: (base[t["task_id"]], 1))

        def c1(t):
            pool = base[t["task_id"]]; st, link = state_of(t, pool)
            if not link: return pool, 1
            q = f"{st.canonical_subject} {st.target_relation}"
            return interleave([pool, fuse(q)], K), 2
        run("C1_canonical_relation", c1)

        def c2(t):
            pool = base[t["task_id"]]; st, link = state_of(t, pool)
            if not link: return pool, 1
            q = formulate_followup(st.with_bridge(st.canonical_subject))
            return interleave([pool, fuse(q)], K), 2
        run("C2_subject_canonical_relation", c2)

        def c3(t):
            pool = base[t["task_id"]]; st, link = state_of(t, pool)
            if not link: return pool, 1
            q1 = f"{st.canonical_subject} {st.target_relation}"
            q2 = f"{st.subject} {st.canonical_subject} {st.target_relation}"
            return interleave([pool, fuse(q1), fuse(q2)], K), 2
        run("C3_two_query_union", c3)

        def c4(t):
            pool, hops = base[t["task_id"]], 1
            st, link = state_of(t, pool)
            if not link: return pool, hops
            merged = interleave([pool, fuse(f"{st.subject} {st.canonical_subject} {st.target_relation}")], K)
            hops = 2
            required = set(t["required_evidence_ids"])
            if not required <= set(merged):
                st2, link2 = state_of(t, merged)
                q = f"{st2.canonical_subject} {st2.target_relation}"
                merged = interleave([merged, fuse(q)], K)
            return merged, hops
        run("C4_bounded_two_step", c4)

        def c5(t):  # CEILING: oracle next query
            meta = t["_oracle_metadata"]; s = meta["surfaces"]
            q = " ".join(filter(None, [s.get("canonical"), s.get("bridge"), meta["target_relation"]]))
            return interleave([base[t["task_id"]], fuse(q)], K), 2
        run("C5_oracle_next_query", c5)
        run("C6_oracle_evidence", lambda t: (list(t["oracle_evidence_ids"]), 0))

        report["partitions"][part] = {"arms": arms, "mean_followup_depth": depths}

    # verdicts from the frozen protocol
    rules = proto["promotion_rules_all_required"]
    ref = proto["reference_arm"]["name"]
    def reg(part, arm, r, key="complete_set@50"):
        return report["partitions"][part]["arms"][arm]["by_axis"]["entity_regime"].get(r, {}).get(key)
    def cond(part, arm):
        return report["partitions"][part]["arms"][arm]["overall"]["target_recall_given_identity_found"]
    verdicts = {}
    for arm in ("C1_canonical_relation","C2_subject_canonical_relation","C3_two_query_union","C4_bounded_two_step"):
        checks, info = {}, {}
        alias_d = round((reg("c2_cal_surface", arm, "alias") or 0) - (reg("c2_cal_surface", ref, "alias") or 0), 4)
        cond_d = round((cond("c2_cal_surface", arm) or 0) - (cond("c2_cal_surface", ref) or 0), 4)
        can_d = round((reg("c2_cal_id", arm, "canonical") or 0) - (reg("c2_cal_id", ref, "canonical") or 0), 4)
        abb_d = round((reg("c2_cal_id", arm, "abbreviation") or 0) - (reg("c2_cal_id", ref, "abbreviation") or 0), 4)
        des_d = round((reg("c2_cal_surface", arm, "description") or 0) - (reg("c2_cal_surface", ref, "description") or 0), 4)
        depth = max(report["partitions"][p]["mean_followup_depth"][arm] for p in PARTS)
        info = {"alias_delta": alias_d, "target_given_identity_delta": cond_d,
                "canonical_delta": can_d, "abbreviation_delta": abb_d,
                "description_delta": des_d, "max_mean_depth": depth}
        checks["alias_complete_set_delta"] = alias_d >= rules["alias_complete_set_delta"]["threshold"]
        checks["target_recall_given_identity_delta"] = cond_d >= rules["target_recall_given_identity_delta"]["threshold"]
        checks["canonical_regression"] = can_d >= rules["canonical_regression"]["threshold"]
        checks["abbreviation_regression"] = abb_d >= rules["abbreviation_regression"]["threshold"]
        checks["description_regression"] = des_d >= rules["description_regression"]["threshold"]
        checks["max_followup_depth"] = depth <= rules["max_followup_depth"]["threshold"]
        verdicts[arm] = {"deltas": info, "checks": checks,
                         "verdict": "PASS_CHAIN_COMPLETION" if all(checks.values()) else "FAIL_CHAIN_COMPLETION"}
    report["verdicts"] = verdicts
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / "chain_completion.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    (out / "RESULTS.sha256").write_text(
        f"{hashlib.sha256((out/'chain_completion.json').read_bytes()).hexdigest()}  chain_completion.json\n")
    print("\n=== verdicts (frozen protocol)")
    for arm, v in verdicts.items():
        print(f"  {arm:32} {v['verdict']}")
        print(f"    {v['deltas']}")
        print(f"    failed: {[k for k,ok in v['checks'].items() if not ok] or 'none'}")


if __name__ == "__main__":
    main()
