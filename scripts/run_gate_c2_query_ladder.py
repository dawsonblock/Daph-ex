#!/usr/bin/env python3
"""Gate C2 — Q0-Q3 query ablation, retrieval only, no HRM.

Isolates query formulation from retrieval representation by feeding four query
policies to identical backends and scoring coverage against evaluator-only
proof labels.

  Q0  the original question
  Q1  oracle bridge surface alone
  Q2  oracle bridge + target relation
  Q3  fully rendered oracle retrieval request

If Q2 >> Q0 the opportunity is query formulation. If Q2 ~ Q0 the retriever's
representation is the binding constraint and a semantic gap reasoner would be
built on top of an index that cannot exploit a perfect query.
"""

from __future__ import annotations

import argparse, asyncio, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evaluation.retrieval_coverage import (
    RetrievalGroundTruth, score_coverage, summarize_coverage)
from hrm_adaptive_memory.experiments.oracle_ladder import read_oracle_facts


def rrf_fuse(rankings, k_constant, limit):
    scores = {}
    for ranking in rankings:
        for rank, value in enumerate(ranking, 1):
            scores[value] = scores.get(value, 0.0) + 1.0 / (k_constant + rank)
    return [v for v, _ in sorted(scores.items(), key=lambda i: (-i[1], i[0]))][:limit]


def queries_for(task, facts):
    """Q0-Q3. Oracle queries never contain the answer; a test asserts this."""
    subject = facts.subject_surface
    relation = facts.target_relation
    bridge = facts.bridge_surface
    return {
        "Q0_question": task["question"],
        "Q1_oracle_bridge": bridge or subject,
        "Q2_bridge_and_relation": f"{bridge} {relation}" if bridge else f"{subject} {relation}",
        "Q3_full_request": (f"{relation} of {bridge} which is the {relation} for {subject}"
                            if bridge else f"{relation} of {subject}"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="qualification")
    p.add_argument("--dataset-root", default="data/hrm/controlled_gate_a_v4")
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--rrf-k", type=int, default=60)
    args = p.parse_args()

    root = Path(args.dataset_root) / args.split
    raw = [json.loads(l) for l in (root / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    ev = [json.loads(l) for l in (root / "evidence.jsonl").read_text().splitlines() if l.strip()]
    truths = {r["task_id"]: RetrievalGroundTruth.from_task(r) for r in raw}
    records = [IndexRecord(evidence_id=r["evidence_id"], source_id=r["source_id"],
                           content=r["content"], token_count=max(1, len(r["content"].split())),
                           source_type=r["source_type"], metadata=r["metadata"]) for r in ev]

    print("building indexes...", flush=True)
    bm25 = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, records)
    dense = CanonicalRetrievalBackend(CanonicalRetrievalMode.DENSE, records)

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    matrix, leak = {}, 0
    for policy in ("Q0_question", "Q1_oracle_bridge", "Q2_bridge_and_relation", "Q3_full_request"):
        matrix[policy] = {}
        per_backend = {"bm25": [], "dense": [], "fusion": []}
        for task in raw:
            facts = read_oracle_facts(task)
            query = queries_for(task, facts)[policy]
            if task["answer"].lower() in query.lower():
                leak += 1
            a = [e.evidence_id for e in asyncio.run(bm25.search(query, k=args.k)).evidence]
            b = [e.evidence_id for e in asyncio.run(dense.search(query, k=args.k)).evidence]
            f = rrf_fuse([a, b], args.rrf_k, args.k)
            truth = truths[task["task_id"]]
            per_backend["bm25"].append(score_coverage(truth, a, retriever="bm25"))
            per_backend["dense"].append(score_coverage(truth, b, retriever="dense"))
            per_backend["fusion"].append(score_coverage(truth, f, retriever="fusion"))
        for backend, rows in per_backend.items():
            matrix[policy][backend] = summarize_coverage(rows, truths, retriever=backend)["overall"]
        row = matrix[policy]
        print(f"[{policy:24}] cs@10 bm25={row['bm25']['complete_set@10']:.3f} "
              f"dense={row['dense']['complete_set@10']:.3f} "
              f"fusion={row['fusion']['complete_set@10']:.3f} | "
              f"cs@50 fusion={row['fusion']['complete_set@50']:.3f} "
              f"ppc@50 fusion={row['fusion']['partial_proof@50']:.3f}", flush=True)

    assert leak == 0, f"{leak} oracle queries contained their own answer"
    (out / "query_ladder.json").write_text(json.dumps(
        {"split": args.split, "k": args.k, "answer_leaks": leak, "matrix": matrix},
        sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
