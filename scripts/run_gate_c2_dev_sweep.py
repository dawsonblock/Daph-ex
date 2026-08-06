#!/usr/bin/env python3
"""Gate C2 Phases 3-5, development split ONLY. Retrieval metrics, no HRM.

Phase 3  Q3 formulation sweep (Q3a-Q3e)
Phase 4  fusion arms + unique-gold contribution and backend overlap
Phase 5  RRF k sweep over {10,30,60,100}

Everything here is chosen on development and then frozen. Qualification and OOD
are not touched.
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evaluation.retrieval_coverage import (
    RetrievalGroundTruth, score_coverage, summarize_coverage)
from hrm_adaptive_memory.experiments.oracle_ladder import read_oracle_facts

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
                seen.add(r[pos]); out.append(r[pos])
                if len(out) >= limit: return out
    return out

def q3_variants(task, f):
    b, rel, subj = f.bridge_surface, f.target_relation, f.subject_surface
    if not b: b = subj
    return {"Q3a_bridge": b,
            "Q3b_bridge_relation": f"{b} {rel}",
            "Q3c_subject_bridge_relation": f"{subj} {b} {rel}",
            "Q3d_relation_bridge": f"{rel} {b}",
            "Q3e_two_query_union": None}  # handled specially

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="development")
    p.add_argument("--output", default="evidence/gate_c2/dev_sweep")
    p.add_argument("--k", type=int, default=50)
    a = p.parse_args()
    d = Path("data/hrm/controlled_gate_a_v4") / a.split
    raw = [json.loads(l) for l in (d / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    ev = [json.loads(l) for l in (d / "evidence.jsonl").read_text().splitlines() if l.strip()]
    truths = {r["task_id"]: RetrievalGroundTruth.from_task(r) for r in raw}
    recs = [IndexRecord(evidence_id=r["evidence_id"], source_id=r["source_id"], content=r["content"],
                        token_count=max(1, len(r["content"].split())), source_type=r["source_type"],
                        metadata=r["metadata"]) for r in ev]
    print("indexing...", flush=True)
    B = {"bm25": CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, recs),
         "minilm": CanonicalRetrievalBackend(CanonicalRetrievalMode.DENSE, recs),
         "bge": CanonicalRetrievalBackend(CanonicalRetrievalMode.DENSE_BGE, recs)}
    def hits(name, q): return [e.evidence_id for e in asyncio.run(B[name].search(q, k=a.k)).evidence]

    report = {"split": a.split, "k": a.k}

    # ---- Phase 3: Q3 formulation sweep (single backend pair, fused) --------
    print("\n[Phase 3] Q3 formulation sweep")
    q3 = {}
    for variant in ("Q3a_bridge","Q3b_bridge_relation","Q3c_subject_bridge_relation",
                    "Q3d_relation_bridge","Q3e_two_query_union"):
        rows = []
        for t in raw:
            f = read_oracle_facts(t)
            if variant == "Q3e_two_query_union":
                b = f.bridge_surface or f.subject_surface
                merged = interleave([hits("bm25", b), hits("bge", b),
                                     hits("bm25", f"{b} {f.target_relation}"),
                                     hits("bge", f"{b} {f.target_relation}")], a.k)
            else:
                q = q3_variants(t, f)[variant]
                merged = rrf([hits("bm25", q), hits("bge", q)], 60, a.k)
            rows.append(score_coverage(truths[t["task_id"]], merged, retriever=variant))
        s = summarize_coverage(rows, truths, retriever=variant)["overall"]
        q3[variant] = s
        print(f"  {variant:30} CES@50={s['complete_set@50']:.3f} CES@10={s['complete_set@10']:.3f} PPC@50={s['partial_proof@50']:.3f}")
    report["phase3_q3_sweep"] = q3
    best_q3 = max(q3, key=lambda v: (q3[v]["complete_set@50"], q3[v]["partial_proof@50"]))
    report["frozen_q3_formulation"] = best_q3
    print(f"  -> frozen Q3 = {best_q3}")

    # ---- Phase 4: fusion arms + unique gold / overlap ---------------------
    print("\n[Phase 4] fusion arms (Q0 queries)")
    per = {n: {} for n in ("bm25","minilm","bge")}
    arms = {}
    for t in raw:
        for n in per: per[n][t["task_id"]] = hits(n, t["question"])
    def arm(name, build):
        rows = [score_coverage(truths[t["task_id"]], build(t["task_id"]), retriever=name) for t in raw]
        s = summarize_coverage(rows, truths, retriever=name)["overall"]
        arms[name] = s
        print(f"  {name:26} CES@50={s['complete_set@50']:.3f} PPC@50={s['partial_proof@50']:.3f}")
    arm("P0_bm25", lambda i: per["bm25"][i])
    arm("P1_minilm", lambda i: per["minilm"][i])
    arm("P2_bge", lambda i: per["bge"][i])
    arm("P3_rrf_bm25_minilm", lambda i: rrf([per["bm25"][i], per["minilm"][i]], 60, a.k))
    arm("P4_rrf_bm25_bge", lambda i: rrf([per["bm25"][i], per["bge"][i]], 60, a.k))
    arm("P5_rrf_three_way", lambda i: rrf([per["bm25"][i], per["minilm"][i], per["bge"][i]], 60, a.k))
    arm("P6_oracle_union_ceiling", lambda i: list(dict.fromkeys(
        per["bm25"][i] + per["minilm"][i] + per["bge"][i])))
    report["phase4_arms"] = arms

    # unique gold contribution + jaccard overlap
    uniq, jac = {}, {}
    for n in per:
        others = [o for o in per if o != n]
        total = 0
        for t in raw:
            g = set(truths[t["task_id"]].required_ids)
            mine = g & set(per[n][t["task_id"]])
            theirs = set().union(*[g & set(per[o][t["task_id"]]) for o in others])
            total += len(mine - theirs)
        uniq[n] = total
    for x in ("minilm","bge"):
        inter = union = 0
        for t in raw:
            A, Bs = set(per["bm25"][t["task_id"]]), set(per[x][t["task_id"]])
            inter += len(A & Bs); union += len(A | Bs)
        jac[f"bm25_vs_{x}"] = round(inter / max(1, union), 4)
    inter = union = 0
    for t in raw:
        A, Bs = set(per["minilm"][t["task_id"]]), set(per["bge"][t["task_id"]])
        inter += len(A & Bs); union += len(A | Bs)
    jac["minilm_vs_bge"] = round(inter / max(1, union), 4)
    report["unique_gold_contribution"] = uniq
    report["jaccard_overlap"] = jac
    print(f"  unique gold: {uniq}")
    print(f"  overlap: {jac}")

    # ---- Phase 5: RRF k sweep on the winning pair ------------------------
    print("\n[Phase 5] RRF k sweep (bm25+bge)")
    ksweep = {}
    for kk in (10, 30, 60, 100):
        rows = [score_coverage(truths[t["task_id"]],
                rrf([per["bm25"][t["task_id"]], per["bge"][t["task_id"]]], kk, a.k),
                retriever=f"rrf_k{kk}") for t in raw]
        s = summarize_coverage(rows, truths, retriever=f"k{kk}")["overall"]
        ksweep[kk] = s
        print(f"  k={kk:3} CES@50={s['complete_set@50']:.3f} CES@10={s['complete_set@10']:.3f} PPC@50={s['partial_proof@50']:.3f}")
    report["phase5_k_sweep"] = {str(k): v for k, v in ksweep.items()}
    best_k = max(ksweep, key=lambda k: (ksweep[k]["complete_set@50"], ksweep[k]["partial_proof@50"]))
    report["frozen_rrf_k"] = best_k
    print(f"  -> frozen k = {best_k}")

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / "dev_sweep.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")

if __name__ == "__main__":
    main()
