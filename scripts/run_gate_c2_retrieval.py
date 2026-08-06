#!/usr/bin/env python3
"""Gate C2 retrieval through the BEIR-format corpus. No HRM, no GPU for BM25.

Phase 6 gate: BM25 run over the exported BEIR corpus must reproduce the
reference coverage measured directly on V4. If it does not, the adapter has
changed the benchmark and nothing downstream is interpretable.
"""

from __future__ import annotations

import argparse, asyncio, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evaluation.retrieval_coverage import (
    RetrievalGroundTruth, score_coverage, summarize_coverage)
from hrm_adaptive_memory.retrieval_bench.contracts import assert_runtime_clean

# Measured directly on V4 before the adapter existed.
REFERENCE_CES50 = {
    ("bm25", "qualification"): 0.350, ("bm25", "ood"): 0.048,
    ("dense_minilm", "qualification"): 0.364, ("dense_minilm", "ood"): 0.072,
    ("dense_bge", "qualification"): 0.370, ("dense_bge", "ood"): 0.152,
}

# "dense" is deliberately absent: it meant MiniLM in earlier runs and would
# mean BGE now, which is exactly how a provenance gap opens.
BACKEND_MODES = {
    "bm25": "BM25", "dense_minilm": "DENSE", "dense_bge": "DENSE_BGE",
}


def load_beir(root: Path, split: str):
    d = root / split
    corpus = [json.loads(l) for l in (d / "corpus.jsonl").read_text().splitlines() if l.strip()]
    queries = [json.loads(l) for l in (d / "queries.jsonl").read_text().splitlines() if l.strip()]
    proof = {json.loads(l)["task_id"]: json.loads(l)
             for l in (d / "proof_ground_truth.jsonl").read_text().splitlines() if l.strip()}
    return corpus, queries, proof


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--beir-root", default="data/hrm/controlled_gate_a_v4_beir")
    p.add_argument("--split", default="qualification")
    p.add_argument("--backend", default="bm25", choices=sorted(BACKEND_MODES))
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--tolerance", type=float, default=0.001)
    args = p.parse_args()

    root = Path(args.beir_root)
    corpus, queries, proof = load_beir(root, args.split)

    # Runtime sees corpus text and question text only.
    for row in corpus[:50] + queries[:50]:
        assert_runtime_clean(row, where="runtime input")

    records = [IndexRecord(evidence_id=r["_id"], source_id=r["_id"], content=r["text"],
                           token_count=max(1, len(r["text"].split())),
                           source_type="beir", metadata={}) for r in corpus]
    mode = getattr(CanonicalRetrievalMode, BACKEND_MODES[args.backend])
    backend = CanonicalRetrievalBackend(mode, records)

    truths = {}
    for task_id, row in proof.items():
        truths[task_id] = RetrievalGroundTruth(
            task_id=task_id, family=row["family"], entity_regime=row["entity_regime"],
            answer_kind=row["answer_kind"], source_style=row["source_style"],
            opportunity_group=row["opportunity_group"],
            required_ids=tuple(row["required_evidence_ids"]),
            proof_path_ids=tuple(row["proof_path_ids"]),
            bridge_ids=tuple(row["bridge_ids"]),
            answer_record_ids=tuple(row["answer_record_ids"]))

    rows, rankings, started = [], [], time.perf_counter()
    for index, query in enumerate(queries, 1):
        t0 = time.perf_counter()
        result = asyncio.run(backend.search(query["text"], k=args.k))
        ranked = [e.evidence_id for e in result.evidence]
        # Record every score channel explicitly. The previous `a or b or c`
        # chain silently fell through on a legitimate 0.0, attributing a
        # lexical zero to the dense channel and destroying provenance.
        rankings.append({"task_id": query["_id"], "candidate_ids": ranked,
                         "candidate_scores": [{
                             "document_id": e.evidence_id, "rank": e.rank,
                             "lexical_score": e.lexical_score,
                             "dense_score": e.dense_score,
                             "reranker_score": e.reranker_score,
                             "final_rank_score": next(
                                 (v for v in (e.reranker_score, e.lexical_score, e.dense_score)
                                  if v is not None), 0.0),
                         } for e in result.evidence]})
        rows.append(score_coverage(truths[query["_id"]], ranked, retriever=args.backend,
                                   latency_ms=(time.perf_counter() - t0) * 1000))
        if index % 200 == 0:
            print(f"  {index}/{len(queries)}", flush=True)

    summary = summarize_coverage(rows, truths, retriever=args.backend)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "per_task.jsonl").write_text(
        "".join(json.dumps(r.to_dict(), sort_keys=True) + "\n" for r in rows))
    (out / "rankings.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rankings))
    (out / "metrics.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")

    measured = summary["overall"]["complete_set@50"]
    reference = REFERENCE_CES50.get((args.backend, args.split))
    reproduced = None if reference is None else abs(measured - reference) <= args.tolerance
    import subprocess, torch, transformers
    spec = backend.embedding_spec
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True).stdout.strip())
    d = root / args.split
    manifest = {
        "gate": "C2", "stage": "retrieval", "split": args.split, "backend": args.backend,
        "model_id": None if spec is None else spec.model_id,
        "model_revision": None if spec is None else spec.revision,
        "pooling": None if spec is None else spec.pooling,
        "normalize": None if spec is None else spec.normalize,
        "dimension": None if spec is None else spec.dimension,
        "max_sequence_length": None if spec is None else spec.max_sequence_length,
        "query_prefix": None if spec is None else (spec.query_prefix or None),
        "document_prefix": None if spec is None else (spec.document_prefix or None),
        "dtype": None if spec is None else spec.dtype,
        "device": "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"),
        "embedding_config_hash": None if spec is None else spec.digest(),
        "backend_config_hash": backend.config_digest(),
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "corpus_hash": hashlib.sha256((d / "corpus.jsonl").read_bytes()).hexdigest(),
        "query_hash": hashlib.sha256((d / "queries.jsonl").read_bytes()).hexdigest(),
        "qrels_hash": hashlib.sha256((d / "qrels" / "test.tsv").read_bytes()).hexdigest(),
        "proof_truth_hash": hashlib.sha256((d / "proof_ground_truth.jsonl").read_bytes()).hexdigest(),
        "git_commit": git, "dirty_tree": dirty,
        "k_values": [1, 5, 10, 20, 50], "candidate_budget": args.k,
        "corpus_sha256": hashlib.sha256((root / args.split / "corpus.jsonl").read_bytes()).hexdigest(),
        "query_policy": "Q0_original",
        "reference_complete_set@50": reference, "measured_complete_set@50": measured,
        "reproduces_reference": reproduced,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (out / "RESULTS.sha256").write_text("".join(
        f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n"
        for n in ("per_task.jsonl", "rankings.jsonl", "metrics.json", "manifest.json")))

    print(f"[{args.backend}/{args.split}] CES@50={measured:.4f} "
          f"reference={reference} reproduced={reproduced}")
    if reproduced is False:
        raise SystemExit("ADAPTER CHANGED THE BENCHMARK — refusing to proceed")


if __name__ == "__main__":
    main()
