#!/usr/bin/env python3
"""Gate C2 — measure retrieval coverage on frozen V4. No HRM, no GPU.

C2-R0  bm25            the current lexical baseline
C2-R1  dense           one pinned dense encoder
C2-R2  union           BM25 top-k + dense top-k, deduplicated (coverage only)
C2-R3  rrf             reciprocal rank fusion over the same two rankings

Answer accuracy cannot distinguish "retrieval never found it" from "the reader
could not use it", so this measures coverage directly against evaluator-only
ground truth read from each task's proof graph.
"""

from __future__ import annotations

import argparse, asyncio, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evaluation.retrieval_coverage import (
    DEPTHS, RetrievalGroundTruth, score_coverage, summarize_coverage)


def interleave(rankings, limit):
    """Round-robin merge, deduplicated.

    Concatenating instead would place every result of the first retriever
    before any of the second, so at any depth <= len(first) the union is a
    copy of the first retriever and complementarity is invisible.
    """
    merged, seen = [], set()
    for position in range(max(len(r) for r in rankings)):
        for ranking in rankings:
            if position < len(ranking) and ranking[position] not in seen:
                seen.add(ranking[position])
                merged.append(ranking[position])
                if len(merged) >= limit:
                    return merged
    return merged


def rrf_fuse(rankings, k_constant, limit):
    scores = {}
    for ranking in rankings:
        for rank, value in enumerate(ranking, 1):
            scores[value] = scores.get(value, 0.0) + 1.0 / (k_constant + rank)
    return [v for v, _ in sorted(scores.items(), key=lambda i: (-i[1], i[0]))][:limit]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="qualification")
    p.add_argument("--dataset-root", default="data/hrm/controlled_gate_a_v4")
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--arms", default="bm25,dense,union,rrf")
    args = p.parse_args()

    root = Path(args.dataset_root) / args.split
    raw = [json.loads(l) for l in (root / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    ev = [json.loads(l) for l in (root / "evidence.jsonl").read_text().splitlines() if l.strip()]
    truths = {r["task_id"]: RetrievalGroundTruth.from_task(r) for r in raw}
    records = [IndexRecord(evidence_id=r["evidence_id"], source_id=r["source_id"],
                           content=r["content"], token_count=max(1, len(r["content"].split())),
                           source_type=r["source_type"], metadata=r["metadata"]) for r in ev]

    arms = args.arms.split(",")
    backends = {}
    if {"bm25", "union", "rrf"} & set(arms):
        backends["bm25"] = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, records)
    if {"dense", "union", "rrf"} & set(arms):
        print("building dense index (pinned MiniLM)...", flush=True)
        backends["dense"] = CanonicalRetrievalBackend(CanonicalRetrievalMode.DENSE, records)

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    report = {"split": args.split, "k": args.k, "rrf_k": args.rrf_k,
              "task_count": len(raw), "arms": {}}

    for arm in arms:
        rows, started = [], time.perf_counter()
        for index, task in enumerate(raw, 1):
            q = task["question"]
            t0 = time.perf_counter()
            if arm in ("bm25", "dense"):
                res = asyncio.run(backends[arm].search(q, k=args.k))
                ranked = [e.evidence_id for e in res.evidence]
            else:
                a = [e.evidence_id for e in asyncio.run(backends["bm25"].search(q, k=args.k)).evidence]
                b = [e.evidence_id for e in asyncio.run(backends["dense"].search(q, k=args.k)).evidence]
                # The union pool must be able to hold both rankings. Capping it
                # at k would let the first retriever fill it and silently make
                # the arm a copy of BM25.
                ranked = (interleave([a, b], args.k) if arm == "union"
                          else rrf_fuse([a, b], args.rrf_k, args.k))
            rows.append(score_coverage(truths[task["task_id"]], ranked, retriever=arm,
                                       latency_ms=(time.perf_counter() - t0) * 1000))
            if index % 100 == 0:
                print(f"  [{arm}] {index}/{len(raw)}", flush=True)
        (out / f"{arm}_coverage.jsonl").write_text(
            "".join(json.dumps(r.to_dict(), sort_keys=True) + "\n" for r in rows))
        summary = summarize_coverage(rows, truths, retriever=arm)
        summary["wall_seconds"] = round(time.perf_counter() - started, 1)
        report["arms"][arm] = summary
        o = summary["overall"]
        print(f"[{arm}] complete_set@10={o['complete_set@10']} @50={o['complete_set@50']} "
              f"recall@10={o['recall@10']} proof@10={o['proof_path@10']} "
              f"({summary['wall_seconds']}s)", flush=True)

    (out / "coverage_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
