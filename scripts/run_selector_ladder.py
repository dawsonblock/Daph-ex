#!/usr/bin/env python3
"""Gate C2-S selector ladder over the frozen candidate pools.

Every arm consumes the identical immutable pool from
evidence/gate_c2/candidate_pools/v1. Nothing upstream varies: retrieval, chain
completion, candidate count, query generation, canonicalization, the reader,
and the pool contents are fixed by construction here. Only selection differs,
which is what makes the comparison attributable to selection.

    S0  raw pool order (baseline packet)
    S1  pointwise lexical relevance
    S2  relation / connectivity chain scoring
    S3  lightweight pinned cross-encoder
    S4  stronger pinned cross-encoder
    S5  oracle selection restricted to what retrieval found (CEILING)

Two guards exist because of real failures:

  * Results are written after every (partition, budget, arm) cell and the run
    is resumable. An earlier foreground run died at a 10-minute timeout and
    wrote nothing at all, leaving measured numbers with no receipt.
  * Each arm records `differs_from_s0` — the fraction of tasks whose selection
    differs from S0. A reranker that silently fails to reorder shows up as
    0.0 here instead of masquerading as a genuine "no effect" finding.

The reserved holdout is never touched by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.evidence.packing import compose_evidence_prompt
from hrm_adaptive_memory.experiments.context_study import OracleTask, verify_answer
from hrm_adaptive_memory.retrieval_bench.selectors import (
    DegenerateRerankerError,
    make_cross_encoder_selector,
    s0_raw,
    s1_relevance,
    s2_connectivity,
    s5_oracle,
)

POOL_DIR = ROOT / "evidence/gate_c2/candidate_pools/v1"
CORPUS = ROOT / "data/hrm/controlled_gate_c2_description_valid_v4"
PARTITIONS = ("descv4_id", "descv4_surface")

# Pinned rerankers. ms-marco-MiniLM-L6-v2 is deliberately NOT used: its forward
# pass returns NaN for every pair under torch 2.10 / transformers 5.14.1, which
# is what made the first S3 arm duplicate S0 exactly.
S3_MODEL = ("cross-encoder/ms-marco-MiniLM-L-12-v2",
            "7b0235231ca2674cb8ca8f022859a6eba2b1c968")
S4_MODEL = ("Alibaba-NLP/gte-reranker-modernbert-base",
            "f7481e6055501a30fb19d090657df9ec1f79ab2c")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_records(meta: dict) -> dict[str, set[str]]:
    """Split the evaluator-only proof graph into record roles.

    identity  edges asserting one surface refers to a canonical entity
    answer    edges whose target is the answer node
    bridge    intermediate hops that are neither identity nor answer-bearing
    """
    answer_node = meta["answer_node"]
    identity, answer, bridge = set(), set(), set()
    for edge in meta["proof_edges"]:
        record = edge["record_id"]
        if edge["relation"] == "refers_to":
            identity.add(record)
        elif edge["target"] == answer_node:
            answer.add(record)
        else:
            bridge.add(record)
    return {"identity": identity, "answer": answer, "bridge": bridge}


def conditional(hits: int, eligible: int) -> float | None:
    """Rate over eligible tasks only; None when nothing was eligible.

    A retention metric is only meaningful for tasks whose target record was
    actually present in the pool. Reporting 0.0 for an empty denominator would
    invent a failure that the data does not contain.
    """
    return round(hits / eligible, 4) if eligible else None


def load_partition(part: str) -> tuple[list[dict], dict[str, dict], dict[str, str]]:
    pools = {}
    for line in (POOL_DIR / f"{part}.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            pools[row["task_id"]] = row
    tasks = [json.loads(l) for l in
             (CORPUS / part / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    texts = {}
    for line in (CORPUS / part / "evidence.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            texts[row["evidence_id"]] = row["content"]
    missing = [t["task_id"] for t in tasks if t["task_id"] not in pools]
    if missing:
        raise RuntimeError(f"{part}: {len(missing)} tasks absent from the frozen pool")
    return tasks, pools, texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evidence/gate_c2/selector_ladder")
    parser.add_argument("--budgets", default="6,2,4,8,10",
                        help="primary budget first; the sweep follows")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-s4", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers

    from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition

    frozen = json.loads((ROOT / "configs/gate_a/gate_a_v2_frozen.json").read_text())
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "cells.jsonl"

    done: set[tuple[str, int, str]] = set()
    if cells_path.exists():
        for line in cells_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["partition"], row["budget"], row["arm"]))
        print(f"resuming: {len(done)} cells already measured", flush=True)

    adapter = HRMAdapter.from_pretrained(
        spec=HRMModelSpec(), dtype=torch.bfloat16, device_map="auto")
    condition = PromptCondition(frozen["prompt_condition"])
    max_new = frozen["max_new_tokens"]

    print("loading rerankers...", flush=True)
    arms: list[tuple[str, object]] = [
        ("S0_raw", s0_raw), ("S1_relevance", s1_relevance),
        ("S2_connectivity", s2_connectivity),
        ("S3_cross_encoder", make_cross_encoder_selector(*S3_MODEL)),
    ]
    if not args.skip_s4:
        try:
            arms.append(("S4_cross_encoder_strong", make_cross_encoder_selector(*S4_MODEL)))
        except Exception as exc:  # noqa: BLE001 - record, do not fake the arm
            print(f"S4 unavailable ({type(exc).__name__}: {exc}); continuing without it",
                  flush=True)
    arms.append(("S5_oracle", s5_oracle))

    budgets = [int(b) for b in args.budgets.split(",")]
    data = {part: load_partition(part) for part in PARTITIONS}

    for part in PARTITIONS:
        tasks, pools, texts = data[part]
        if args.limit:
            tasks = tasks[:args.limit]
        for budget in budgets:
            print(f"\n=== {part} n={len(tasks)} budget={budget}", flush=True)
            # S0 selections are the reference for the no-op detector.
            s0_pick = {t["task_id"]: s0_raw(pools[t["task_id"]]["candidates"], budget=budget)
                       for t in tasks}
            for name, fn in arms:
                if (part, budget, name) in done:
                    print(f"  {name:24} (cached)", flush=True)
                    continue
                started = time.perf_counter()
                quality = 0.0
                csr_hit = csr_elig = 0
                role_hit = {"answer": 0, "bridge": 0, "identity": 0}
                role_elig = {"answer": 0, "bridge": 0, "identity": 0}
                gold_density = 0.0
                distractors = 0
                tokens = 0
                differs = 0
                degenerate: str | None = None
                for task_row in tasks:
                    meta = task_row["_oracle_metadata"]
                    candidates = pools[task_row["task_id"]]["candidates"]
                    pool_ids = {c["document_id"] for c in candidates}
                    required = set(task_row["required_evidence_ids"])
                    task = OracleTask.from_dict(
                        {k: v for k, v in task_row.items() if k != "_oracle_metadata"})
                    try:
                        selected = fn(candidates, budget=budget, question=task_row["question"],
                                      texts=texts, required=list(required),
                                      target_relation=meta["target_relation"])
                    except DegenerateRerankerError as exc:
                        degenerate = str(exc)
                        break
                    chosen = set(selected)
                    if selected != s0_pick[task_row["task_id"]]:
                        differs += 1
                    prompt = compose_evidence_prompt(
                        task_row["question"], [texts[i] for i in selected if i in texts])
                    generated = adapter.generate(prompt, condition=condition,
                                                 max_new_tokens=max_new)
                    score, _ = verify_answer(task, str(generated["text"]))
                    quality += score
                    if required <= pool_ids:
                        csr_elig += 1
                        csr_hit += int(required <= chosen)
                    for role, records in classify_records(meta).items():
                        available = records & pool_ids
                        if available:
                            role_elig[role] += 1
                            role_hit[role] += int(bool(available & chosen))
                    gold_density += len(required & chosen) / max(1, len(selected))
                    distractors += len(chosen - required)
                    tokens += sum(len(texts[i].split()) for i in selected if i in texts)

                n = len(tasks)
                if degenerate is not None:
                    cell = {"partition": part, "budget": budget, "arm": name,
                            "status": "DEGENERATE_ARM_NOT_MEASURED", "error": degenerate}
                    print(f"  {name:24} DEGENERATE: {degenerate[:60]}", flush=True)
                else:
                    cell = {
                        "partition": part, "budget": budget, "arm": name, "status": "MEASURED",
                        "tasks": n,
                        "quality": round(quality / n, 4),
                        "CSR_given_complete_set_available": conditional(csr_hit, csr_elig),
                        "csr_eligible_tasks": csr_elig,
                        "AnswerRetention": conditional(role_hit["answer"], role_elig["answer"]),
                        "BridgeRetention": conditional(role_hit["bridge"], role_elig["bridge"]),
                        "IdentityRetention": conditional(role_hit["identity"],
                                                         role_elig["identity"]),
                        "role_eligible_tasks": dict(role_elig),
                        "GoldDensity": round(gold_density / n, 4),
                        "DistractorCount": round(distractors / n, 2),
                        "SelectedTokens": round(tokens / n, 1),
                        "differs_from_s0": round(differs / n, 4),
                        "seconds": round(time.perf_counter() - started, 1),
                    }
                    print(f"  {name:24} Q={cell['quality']:.4f} "
                          f"CSR={cell['CSR_given_complete_set_available']} "
                          f"AnsRet={cell['AnswerRetention']} "
                          f"gold={cell['GoldDensity']:.3f} dist={cell['DistractorCount']:.1f} "
                          f"tok={cell['SelectedTokens']:.0f} "
                          f"differs_S0={cell['differs_from_s0']:.2f}", flush=True)
                with cells_path.open("a") as handle:
                    handle.write(json.dumps(cell, sort_keys=True) + "\n")
                done.add((part, budget, name))

    cells = [json.loads(l) for l in cells_path.read_text().splitlines() if l.strip()]
    manifest = {
        "gate": "C2_S_SELECTOR_LADDER",
        "pool_policy": "c2_candidate_generation_v1",
        "pool_dir": str(POOL_DIR.relative_to(ROOT)),
        "pool_digests": {p: sha256_file(POOL_DIR / f"{p}.jsonl") for p in PARTITIONS},
        "corpus_digests": {
            p: {"tasks": sha256_file(CORPUS / p / "oracle_tasks.jsonl"),
                "evidence": sha256_file(CORPUS / p / "evidence.jsonl")}
            for p in PARTITIONS},
        "upstream_unchanged": True,
        "only_selection_varies": True,
        "holdout_touched": False,
        "budgets": budgets,
        "arm_models": {"S3_cross_encoder": {"model_id": S3_MODEL[0], "revision": S3_MODEL[1]},
                       "S4_cross_encoder_strong": {"model_id": S4_MODEL[0],
                                                   "revision": S4_MODEL[1]}},
        "excluded_model": {
            "model_id": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "reason": ("forward pass returns NaN for every pair under this stack; "
                       "NaN sort keys left the pool unreordered so the arm reported "
                       "figures identical to S0")},
        "reader": {"model_id": adapter.spec.model_id, "revision": adapter.spec.revision,
                   "prompt_condition": frozen["prompt_condition"], "max_new_tokens": max_new},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__},
        "cells": cells,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report = out_dir / "selector_ladder.json"
    report.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (out_dir / "RESULTS.sha256").write_text(
        f"{sha256_file(report)}  selector_ladder.json\n")
    print(f"\nwrote {report} ({len(cells)} cells)", flush=True)


if __name__ == "__main__":
    main()
