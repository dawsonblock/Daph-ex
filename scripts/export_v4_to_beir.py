#!/usr/bin/env python3
"""Export frozen V4 into BEIR format, with proof truth kept strictly adjacent.

Emits the standard corpus/queries/qrels layout so BEIR (or any other IR
harness) can consume it, without importing BEIR — the format is the contract.

Proof structure richer than qrels lives in a SEPARATE evaluator-only file that
must never be handed to a retriever, selector, or prompt.
"""

from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.evaluation.retrieval_coverage import RetrievalGroundTruth
from hrm_adaptive_memory.retrieval_bench.contracts import assert_runtime_clean

SPLITS = ("development", "qualification", "ood")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data/hrm/controlled_gate_a_v4")
    p.add_argument("--output", default="data/hrm/controlled_gate_a_v4_beir")
    args = p.parse_args()
    src, out = Path(args.source), Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")

    summary = {}
    for split in SPLITS:
        tasks = [json.loads(l) for l in (src / split / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
        evidence = [json.loads(l) for l in (src / split / "evidence.jsonl").read_text().splitlines() if l.strip()]
        d = out / split
        (d / "qrels").mkdir(parents=True)

        # corpus: runtime-visible text only
        corpus = [{"_id": r["evidence_id"], "title": "", "text": r["content"]} for r in evidence]
        for row in corpus:
            assert_runtime_clean(row, where=f"{split}/corpus")
        (d / "corpus.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in corpus))

        # queries: the question, nothing else
        queries = [{"_id": t["task_id"], "text": t["question"]} for t in tasks]
        for row in queries:
            assert_runtime_clean(row, where=f"{split}/queries")
        (d / "queries.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in queries))

        # qrels: required evidence, graded 1 — evaluator-side
        lines = ["query-id\tcorpus-id\tscore"]
        for task in tasks:
            for value in task["required_evidence_ids"]:
                lines.append(f"{task['task_id']}\t{value}\t1")
        (d / "qrels" / "test.tsv").write_text("\n".join(lines) + "\n")

        # proof truth: strictly adjacent, never a runtime input
        proof = [RetrievalGroundTruth.from_task(t) for t in tasks]
        (d / "proof_ground_truth.jsonl").write_text("".join(
            json.dumps({
                "task_id": g.task_id, "required_evidence_ids": list(g.required_ids),
                "proof_path_ids": list(g.proof_path_ids), "bridge_ids": list(g.bridge_ids),
                "answer_record_ids": list(g.answer_record_ids), "family": g.family,
                "entity_regime": g.entity_regime, "answer_kind": g.answer_kind,
                "source_style": g.source_style, "opportunity_group": g.opportunity_group,
            }, sort_keys=True) + "\n" for g in proof))

        summary[split] = {"documents": len(corpus), "queries": len(queries),
                          "qrel_rows": len(lines) - 1,
                          "corpus_sha256": hashlib.sha256(
                              (d / "corpus.jsonl").read_bytes()).hexdigest()}
        print(f"[{split}] {len(corpus)} docs, {len(queries)} queries, {len(lines)-1} qrels")

    (out / "EXPORT_MANIFEST.json").write_text(json.dumps({
        "source": str(src), "format": "beir-compatible",
        "beir_package_required": False,
        "evaluator_only_file": "proof_ground_truth.jsonl",
        "splits": summary,
    }, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
