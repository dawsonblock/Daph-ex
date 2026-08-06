#!/usr/bin/env python3
"""Freeze the qualified candidate-generation policy and emit immutable pools.

Policy = P2 fusion (BM25 + pinned BGE, RRF k=10) + C4 bounded chain completion,
both now qualified. Pools are the boundary between Gate C2-R and C2-S: every
selector arm must consume these exact candidates so selection is isolated.

Proof labels stay evaluator-side; pool artifacts carry runtime-visible data only.
"""
from __future__ import annotations
import asyncio, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evaluation.retrieval_coverage import RetrievalGroundTruth
from hrm_adaptive_memory.retrieval.canonicalization import resolve_canonical
from hrm_adaptive_memory.retrieval.embedding import BGE_SMALL, EmbeddingSpec
from hrm_adaptive_memory.retrieval.information_state import (
    FOLLOWUP_FORMULATION, InformationState)

SRC = Path("data/hrm/controlled_gate_c2_description_valid_v4")
OUT = Path("evidence/gate_c2/candidate_pools/v1")
PARTS = ("descv4_id", "descv4_surface")
K, RRF_K = 50, 10


def h(x): return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()
def rrf(rs, k, lim):
    s = {}
    for r in rs:
        for i, v in enumerate(r, 1): s[v] = s.get(v, 0.0) + 1.0/(k+i)
    return sorted(s.items(), key=lambda x: (-x[1], x[0]))[:lim]
def inter(rs, lim):
    out, seen = [], set()
    for p in range(max(len(r) for r in rs)):
        for r in rs:
            if p < len(r) and r[p] not in seen:
                seen.add(r[p]); out.append(r[p])
                if len(out) >= lim: return out
    return out


def main() -> None:
    if OUT.exists(): raise FileExistsError(f"pools are immutable: {OUT}")
    policy = {
        "policy_id": "c2_candidate_generation_v1", "policy_version": 1,
        "components": ["P2_rrf_bm25_bge", "C4_bounded_chain_completion"],
        "p2_config": {"retrievers": ["bm25", "dense_bge"], "fusion": "rrf", "rrf_k": RRF_K, "k": K},
        "c4_config": {"max_followup_depth": 2, "formulation": FOLLOWUP_FORMULATION,
                      "canonicalization": "runtime_parse_of_candidate_text"},
        "embedding": dict(BGE_SMALL), "embedding_config_hash": EmbeddingSpec(**BGE_SMALL).digest(),
        "qualified_by": "gate_c2c_description_valid_v4 PASS all four regimes",
    }
    policy["p2_config_hash"] = h(json.dumps(policy["p2_config"], sort_keys=True))
    policy["c4_config_hash"] = h(json.dumps(policy["c4_config"], sort_keys=True))

    OUT.mkdir(parents=True)
    ceilings = {}
    for part in PARTS:
        raw = [json.loads(l) for l in (SRC/part/"oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
        ev = [json.loads(l) for l in (SRC/part/"evidence.jsonl").read_text().splitlines() if l.strip()]
        truths = {r["task_id"]: RetrievalGroundTruth.from_task(r) for r in raw}
        recs = [IndexRecord(evidence_id=r["evidence_id"], source_id=r["source_id"],
                content=r["content"], token_count=max(1, len(r["content"].split())),
                source_type=r["source_type"], metadata=r["metadata"]) for r in ev]
        idx = {r.evidence_id: r for r in recs}
        bm = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, recs)
        bg = CanonicalRetrievalBackend(CanonicalRetrievalMode.DENSE_BGE, recs)
        def scored(q):
            a = asyncio.run(bm.search(q, k=K)).evidence; b = asyncio.run(bg.search(q, k=K)).evidence
            lex = {e.evidence_id: e.lexical_score for e in a}
            den = {e.evidence_id: e.dense_score for e in b}
            fused = rrf([[e.evidence_id for e in a], [e.evidence_id for e in b]], RRF_K, K)
            return fused, lex, den
        rows, stats = [], {"complete": 0, "answer": 0, "bridge": 0, "identity": 0, "partial": 0.0}
        print(f"\n=== {part} ({len(raw)})", flush=True)
        for t in raw:
            m = t["_oracle_metadata"]
            fused, lex, den = scored(t["question"])
            pool = [i for i, _ in fused]
            st = InformationState(subject=m["surfaces"]["subject"], target_relation=m["target_relation"])
            link = resolve_canonical(st.subject, [idx[i] for i in pool if i in idx])
            depth, fq = 1, []
            if link:
                st = st.with_identity(link.surface, link.canonical, record_id=link.record_id)
                q1 = f"{st.subject} {st.canonical_subject} {st.target_relation}"
                f1, lex1, den1 = scored(q1); fq.append(q1); depth = 2
                lex.update(lex1); den.update(den1)
                pool = inter([pool, [i for i, _ in f1]], K)
                fused_scores = dict(fused); fused_scores.update(dict(f1))
            else:
                fused_scores = dict(fused)
            tr = truths[t["task_id"]]
            ps = set(pool)
            stats["complete"] += int(set(tr.required_ids) <= ps)
            stats["answer"] += int(bool(set(tr.answer_record_ids) & ps))
            if tr.bridge_ids: stats["bridge"] += int(set(tr.bridge_ids) <= ps)
            if tr.identity_record_ids: stats["identity"] += int(set(tr.identity_record_ids) <= ps)
            w = tr.weights(); tot = sum(w.values()) or 1
            stats["partial"] += sum(v for k2, v in w.items() if k2 in ps) / tot
            rows.append({
                "task_id": t["task_id"], "split": part, "policy_id": policy["policy_id"],
                "query_policy": "Q0_original", "query_hash": h(t["question"]),
                "candidate_budget": K,
                "followup_depth": depth, "followup_query_digests": [h(q) for q in fq],
                "runtime_canonicalization": ({"surface": link.surface, "canonical": link.canonical,
                                              "from_record": link.record_id} if link else None),
                "candidates": [{"document_id": i, "rank": r,
                                "lexical_score": lex.get(i), "dense_score": den.get(i),
                                "fusion_score": fused_scores.get(i)}
                               for r, i in enumerate(pool, 1)],
            })
        n = len(raw)
        nb = sum(1 for t in truths.values() if t.bridge_ids) or 1
        ni = sum(1 for t in truths.values() if t.identity_record_ids) or 1
        ceilings[part] = {
            "n": n, "CompleteSetAvailable@50": round(stats["complete"]/n, 4),
            "AnswerRecordAvailable@50": round(stats["answer"]/n, 4),
            "BridgeAvailable@50": round(stats["bridge"]/nb, 4),
            "IdentityAvailable@50": round(stats["identity"]/ni, 4),
            "PartialProofAvailable@50": round(stats["partial"]/n, 4)}
        for k2, v in ceilings[part].items(): print(f"  {k2:28} {v}")
        (OUT/f"{part}.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows))

    (OUT/"manifest.json").write_text(json.dumps({
        "policy": policy, "source_corpus": str(SRC), "partitions": list(PARTS),
        "availability_ceilings": ceilings,
        "corpus_hashes": {p: h((SRC/p/"evidence.jsonl").read_bytes()) for p in PARTS},
        "task_hashes": {p: h((SRC/p/"oracle_tasks.jsonl").read_bytes()) for p in PARTS},
        "immutable_after": "selector work begins; every S-arm consumes these exact candidates",
        "proof_labels_included": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, sort_keys=True, indent=2)+"\n")
    (OUT/"RESULTS.sha256").write_text("".join(
        f"{h((OUT/f).read_bytes())}  {f}\n" for f in sorted(x.name for x in OUT.iterdir() if x.is_file())))
    print(f"\nfrozen: {OUT}")


if __name__ == "__main__":
    main()
