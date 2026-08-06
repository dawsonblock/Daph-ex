#!/usr/bin/env python3
"""Diagnostic: why description-regime chain completion fails. No HRM, no promotion.

v3 measured description OGC = 0.025 against an oracle gap of 0.800 -- the largest
headroom of any regime, almost none realised -- while abbreviation closed 0.960
and alias 0.815. C5 reaching 1.000 proves the evidence is retrievable given the
right query, so the break is upstream of retrieval.

Runs a staged funnel over every description task in the frozen corpora and
classifies each failure. Measurement only: nothing here promotes or tunes.
"""
from __future__ import annotations
import asyncio, hashlib, json, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.retrieval.canonicalization import extract_identity_links, resolve_canonical
from hrm_adaptive_memory.retrieval.information_state import InformationState

CORPORA = [
    ("cal_v1", Path("data/hrm/controlled_gate_c2_calibration_v1/c2_cal_surface")),
    ("chain_v2", Path("data/hrm/controlled_gate_c2_chain_validation_v2/chain_v2_surface")),
    ("chain_v3", Path("data/hrm/controlled_gate_c2_chain_validation_v3/chain_v3_surface")),
]
K = 50
CLASSES = ("SUCCESS", "NO_DESCRIPTION_MATCH", "AMBIGUOUS_MATCH", "WRONG_CANONICAL_ENTITY",
           "CORRECT_CANONICAL_BAD_QUERY", "CORRECT_QUERY_RETRIEVAL_FAILURE")


def norm(t): return " ".join(re.findall(r"\w+", t.lower()))


def rrf(rs, k, lim):
    s = {}
    for r in rs:
        for i, v in enumerate(r, 1): s[v] = s.get(v, 0.0) + 1.0 / (k + i)
    return [v for v, _ in sorted(s.items(), key=lambda x: (-x[1], x[0]))][:lim]


def inter(rs, lim):
    out, seen = [], set()
    for p in range(max(len(r) for r in rs)):
        for r in rs:
            if p < len(r) and r[p] not in seen:
                seen.add(r[p]); out.append(r[p])
                if len(out) >= lim: return out
    return out


def main() -> None:
    report = {"diagnostic": "description_canonicalization", "corpora": {}, "examples": []}
    for label, d in CORPORA:
        raw = [json.loads(l) for l in (d / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
        ev = [json.loads(l) for l in (d / "evidence.jsonl").read_text().splitlines() if l.strip()]
        desc = [t for t in raw if t["metadata"]["entity_regime"] == "description"]
        recs = [IndexRecord(evidence_id=r["evidence_id"], source_id=r["source_id"],
                            content=r["content"], token_count=max(1, len(r["content"].split())),
                            source_type=r["source_type"], metadata=r["metadata"]) for r in ev]
        idx = {r.evidence_id: r for r in recs}
        bm = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, recs)
        bg = CanonicalRetrievalBackend(CanonicalRetrievalMode.DENSE_BGE, recs)
        def fuse(q):
            return rrf([[e.evidence_id for e in asyncio.run(bm.search(q, k=K)).evidence],
                        [e.evidence_id for e in asyncio.run(bg.search(q, k=K)).evidence]], 10, K)
        print(f"\n=== {label}: {len(desc)} description tasks", flush=True)

        stages = Counter(); classes = Counter()
        cond = {"target_given_correct": [0, 0], "target_given_failed": [0, 0]}
        for t in desc:
            m = t["_oracle_metadata"]
            subject, truth_canon = m["surfaces"]["subject"], m["surfaces"]["canonical"]
            required = set(t["required_evidence_ids"])
            identity_ids = {e["record_id"] for e in m["proof_edges"]
                            if str(e.get("source", "")).startswith("surface:")}
            pool = fuse(t["question"])
            pool_set = set(pool)
            visible = [idx[i] for i in pool if i in idx]

            id_retrieved = bool(identity_ids) and identity_ids <= pool_set
            stages["identity_record_retrieved"] += int(id_retrieved)
            links = extract_identity_links(visible)
            mention_ok = any(norm(subject) in norm(l.surface) or norm(l.surface) in norm(subject)
                             for l in links)
            stages["mention_recognized"] += int(mention_ok)
            link = resolve_canonical(subject, visible)
            stages["resolver_returned"] += int(link is not None)
            correct = bool(link) and norm(link.canonical) == norm(truth_canon)
            stages["canonical_correct"] += int(correct)

            # follow-up under the fixed C4 rule
            if link:
                st = InformationState(subject=subject, target_relation=m["target_relation"]) \
                    .with_identity(link.surface, link.canonical, record_id=link.record_id)
                q = f"{st.subject} {st.canonical_subject} {st.target_relation}"
                merged = set(inter([pool, fuse(q)], K))
            else:
                merged = pool_set
            query_ok = correct
            stages["correct_query_formed"] += int(query_ok)
            answer_ids = {e["record_id"] for e in m["proof_edges"]
                          if e["target"] == m["answer_node"]}
            target_ok = bool(answer_ids) and bool(answer_ids & merged)
            stages["target_evidence_retrieved"] += int(target_ok)
            complete = required <= merged
            stages["complete_proof_recovered"] += int(complete)

            bucket = "target_given_correct" if correct else "target_given_failed"
            cond[bucket][1] += 1; cond[bucket][0] += int(target_ok)

            if complete: cls = "SUCCESS"
            elif link is None: cls = "NO_DESCRIPTION_MATCH"
            elif not correct:
                candidates = [l for l in links
                              if norm(subject) in norm(l.surface) or norm(l.surface) in norm(subject)]
                cls = "AMBIGUOUS_MATCH" if len(candidates) > 1 else "WRONG_CANONICAL_ENTITY"
            elif not target_ok: cls = "CORRECT_CANONICAL_BAD_QUERY"
            else: cls = "CORRECT_QUERY_RETRIEVAL_FAILURE"
            classes[cls] += 1
            if len(report["examples"]) < 6 and cls != "SUCCESS":
                report["examples"].append({
                    "corpus": label, "task_id": t["task_id"], "class": cls,
                    "question_subject": subject, "truth_canonical": truth_canon,
                    "resolver_returned": link.canonical if link else None,
                    "identity_record_retrieved": id_retrieved,
                    "identity_record_text": (idx[sorted(identity_ids)[0]].content
                                             if identity_ids and sorted(identity_ids)[0] in idx else None),
                })

        n = len(desc)
        report["corpora"][label] = {
            "n": n,
            "funnel": {k: round(v / n, 4) for k, v in stages.items()},
            "failure_classes": {c: classes.get(c, 0) for c in CLASSES},
            "ResolutionRate": round(stages["resolver_returned"] / n, 4),
            "ResolutionPrecision": (round(stages["canonical_correct"] / stages["resolver_returned"], 4)
                                    if stages["resolver_returned"] else None),
            "TargetRecallGivenCorrectResolution": (round(cond["target_given_correct"][0] /
                                                         cond["target_given_correct"][1], 4)
                                                   if cond["target_given_correct"][1] else None),
            "TargetRecallGivenFailedResolution": (round(cond["target_given_failed"][0] /
                                                        cond["target_given_failed"][1], 4)
                                                  if cond["target_given_failed"][1] else None),
        }
        r = report["corpora"][label]
        for stage, rate in r["funnel"].items():
            print(f"  {stage:34} {rate:.3f}")
        print(f"  ResolutionRate={r['ResolutionRate']} precision={r['ResolutionPrecision']} "
              f"target|correct={r['TargetRecallGivenCorrectResolution']} "
              f"target|failed={r['TargetRecallGivenFailedResolution']}")
        print(f"  classes: {dict(classes)}")

    out = Path("evidence/gate_c2/description_diagnostic"); out.mkdir(parents=True, exist_ok=True)
    (out / "description_canonicalization.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n")
    (out / "RESULTS.sha256").write_text(
        f"{hashlib.sha256((out/'description_canonicalization.json').read_bytes()).hexdigest()}"
        f"  description_canonicalization.json\n")
    print("\n=== examples")
    for e in report["examples"][:4]:
        print(f"  [{e['class']}] subject={e['question_subject']!r}")
        print(f"     truth={e['truth_canonical']!r} returned={e['resolver_returned']!r}")
        print(f"     identity_text={str(e['identity_record_text'])[:88]!r}")


if __name__ == "__main__":
    main()
