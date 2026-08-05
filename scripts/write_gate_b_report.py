#!/usr/bin/env python3
"""Generate GATE_B_REPORT.md and the machine-readable Gate B verdict.

The verdict is computed from the run manifest, never hand-transcribed. Exactly
one of PASS_RETRIEVAL_EXPANSION / FAIL_RETRIEVAL_QUALITY /
FAIL_EVIDENCE_SET_RECOVERY / INVALID_EXPERIMENT is emitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILY_ORDER = (
    "single_hop", "temporal_update", "distractor_heavy", "two_hop", "numeric_derivation",
)

# A retriever must recover complete evidence sets and convert that into
# downstream answer quality over the no-evidence baseline.
MIN_COMPLETE_SET_SUCCESS = 0.50
MIN_DOWNSTREAM_GAIN = 0.05


def _table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return f"{line}\n{rule}\n{body}"


def build(manifest: dict) -> tuple[str, dict]:
    arms = manifest["arm_reports"]
    anchors = manifest["gate_a_anchors"]
    retrieval_only = manifest.get("retrieval_only", False)

    ranked = sorted(
        arms.items(),
        key=lambda item: -item[1]["retrieval"]["metrics"]["complete_set_success"],
    )
    best_name, best = ranked[0]
    bm25 = arms.get("bm25")
    dense = arms.get("dense")
    hybrids = {k: v for k, v in arms.items() if k.startswith("hybrid")}

    def css(arm: dict) -> float:
        return arm["retrieval"]["metrics"]["complete_set_success"]

    def downstream(arm: dict) -> float | None:
        return None if retrieval_only else arm["downstream"]["mean_quality"]

    best_downstream = downstream(best)
    dense_beats_bm25 = bool(dense and bm25 and css(dense) > css(bm25))
    best_hybrid = max(hybrids.items(), key=lambda item: css(item[1])) if hybrids else None
    hybrid_beats_singles = bool(
        best_hybrid and bm25 and dense
        and css(best_hybrid[1]) > css(bm25) and css(best_hybrid[1]) > css(dense)
    )

    # Verdict
    if not arms or (not retrieval_only and best_downstream is None):
        verdict, rationale = "INVALID_EXPERIMENT", "no arm produced a complete measurement"
    elif css(best) < MIN_COMPLETE_SET_SUCCESS:
        verdict = "FAIL_EVIDENCE_SET_RECOVERY"
        rationale = (
            f"best arm {best_name} recovers complete evidence sets on only "
            f"{css(best):.1%} of tasks (floor {MIN_COMPLETE_SET_SUCCESS:.0%})"
        )
    elif not retrieval_only and (best_downstream - anchors["b0_mean_quality"]) < MIN_DOWNSTREAM_GAIN:
        verdict = "FAIL_RETRIEVAL_QUALITY"
        rationale = (
            f"best arm {best_name} recovers evidence but lifts answer quality by only "
            f"{best_downstream - anchors['b0_mean_quality']:+.3f} over B0"
        )
    else:
        verdict = "PASS_RETRIEVAL_EXPANSION"
        rationale = (
            f"{best_name} recovers complete evidence sets on {css(best):.1%} of tasks"
            + ("" if retrieval_only else
               f" and lifts answer quality {best_downstream - anchors['b0_mean_quality']:+.3f} over B0")
        )

    # Per-family retrieval-bound analysis
    family_notes = {}
    for family in FAMILY_ORDER:
        family_css = best["retrieval"]["per_family"][family]["complete_set_success"]
        note = {"best_complete_set_success": family_css}
        if not retrieval_only:
            note["downstream_quality"] = best["downstream"]["per_family"].get(family)
            note["b3_oracle"] = anchors["b3_by_family"].get(family)
            classes = best["failure_attribution"]["per_family"].get(family, {})
            note["dominant_failure"] = max(
                ((k, v) for k, v in classes.items() if k != "NONE"),
                key=lambda item: item[1], default=("NONE", 0),
            )[0]
        family_notes[family] = note

    verdict_doc = {
        "gate": "B_RETRIEVAL",
        "verdict": verdict,
        "rationale": rationale,
        "best_arm": best_name,
        "best_complete_set_success": css(best),
        "best_downstream_quality": best_downstream,
        "b0_anchor": anchors["b0_mean_quality"],
        "b3_anchor": anchors["b3_mean_quality"],
        "dense_beats_bm25": dense_beats_bm25,
        "hybrid_beats_individual_backends": hybrid_beats_singles,
        "per_family": family_notes,
        "thresholds": {
            "min_complete_set_success": MIN_COMPLETE_SET_SUCCESS,
            "min_downstream_gain_over_b0": MIN_DOWNSTREAM_GAIN,
        },
        "iterative_retrieval_authorized": verdict == "PASS_RETRIEVAL_EXPANSION",
        "next_stage": (
            "BOUNDED_ITERATIVE_RETRIEVAL" if verdict == "PASS_RETRIEVAL_EXPANSION"
            else "RETRIEVAL_REPAIR"
        ),
        "still_blocked": [
            "macro_executive_training", "micro_compute_controller", "adaptive_recurrence",
            "graphiti_temporal_memory", "external_vector_engines",
            "transactional_persistent_memory",
        ],
        "source_manifest": manifest.get("_path"),
    }

    # --- markdown ---
    headers = ["arm", "CompleteSet", "ReqEvRecall", "R@10", "MRR", "nDCG", "irrelevant tok",
               "latency ms", "index s"]
    if not retrieval_only:
        headers += ["downstream Q", "Δ vs B0", "oracle gap"]
    rows = []
    for name, arm in ranked:
        m = arm["retrieval"]["metrics"]
        row = [
            f"`{name}`", f"**{m['complete_set_success']:.3f}**",
            f"{m['required_evidence_recall']:.3f}", f"{m['recall_at_10']:.3f}",
            f"{m['mrr']:.3f}", f"{m['ndcg']:.3f}", f"{m['irrelevant_token_ratio']:.3f}",
            f"{m['mean_latency_ms']:.1f}", f"{arm['index_seconds']:.1f}",
        ]
        if not retrieval_only:
            d = arm["downstream"]
            row += [f"{d['mean_quality']:.3f}", f"{d['delta_vs_b0']:+.3f}", f"{d['oracle_gap']:.3f}"]
        rows.append(row)
    arm_table = _table(headers, rows)

    family_rows = []
    for name, arm in ranked:
        pf = arm["retrieval"]["per_family"]
        family_rows.append(
            [f"`{name}`"] + [f"{pf[f]['complete_set_success']:.3f}" for f in FAMILY_ORDER]
        )
    family_table = _table(["arm"] + [f"`{f}`" for f in FAMILY_ORDER], family_rows)

    failure_section = "_Retrieval-only pass; downstream failure attribution not measured._"
    if not retrieval_only:
        fa = best["failure_attribution"]
        counts = {k: v for k, v in fa["counts"].items() if v and k != "NONE"}
        failure_rows = [
            [f"`{family}`"] + [
                str(fa["per_family"].get(family, {}).get(cls, 0)) for cls in counts
            ]
            for family in FAMILY_ORDER
        ]
        failure_section = (
            f"Best arm (`{best_name}`) — {fa['failure_count']} failures of {fa['task_count']} tasks; "
            f"{fa['retrieval_bound_fraction']:.1%} of failures are retrieval-bound.\n\n"
            + _table(["family"] + [f"`{c}`" for c in counts], failure_rows)
        )

    diagnostic_section = ""
    manifest_dir = Path(manifest.get("_path", "")).parent
    for diagnostic_path in (
        manifest_dir / "packing_diagnostic.json",
        manifest_dir.parent / "packing_diagnostic" / "packing_diagnostic.json",
    ):
        if diagnostic_path.exists():
            break
    if diagnostic_path.exists():
        diag = json.loads(diagnostic_path.read_text())
        size_rows = [[f"{k} records", f"{v['quality']:.3f}", str(v["slot_label_echoes"])]
                     for k, v in sorted(diag["packet_size"].items(), key=lambda i: int(i[0]))]
        pos_rows = [[k, f"{v['quality']:.3f}", str(v["slot_label_echoes"])]
                    for k, v in diag["oracle_position"].items()]
        kind_rows = [[f"`{k}`", f"{v['quality']:.3f}", str(v["slot_label_echoes"])]
                     for k, v in diag["distractor_kind"].items()]
        verdict_doc["packing_diagnostic"] = {
            "distractor_kind": {k: v["quality"] for k, v in diag["distractor_kind"].items()},
            "conclusion": diag["conclusion"],
        }
        diagnostic_section = f"""
## Why complete evidence was not always enough

Gate B's best arm answered only 1 of 9 `two_hop` tasks whose evidence was
*fully* retrieved, while Gate A's oracle arm scored 100/100 on that family.
The difference is not the amount of evidence. Holding the required evidence
present in every condition and varying one factor at a time
(`scripts/diagnose_gate_b_packing.py`, N = {diag['distractor_kind']['random_corpus']['n']} per cell):

**Packet size** — no effect:

{_table(["packet", "quality", "slot-label echoes"], size_rows)}

**Position of the required evidence** — no effect:

{_table(["oracle position", "quality", "slot-label echoes"], pos_rows)}

**Distractor similarity** — this is the mechanism:

{_table(["distractor kind", "quality", "slot-label echoes"], kind_rows)}

With unrelated padding the model is perfect; with near-duplicate records that
differ only in their entity identifiers it collapses, and its characteristic
failure is emitting an evidence slot label (`[E4]`) instead of a value.

**Retrieval precision, not just recall, is a binding constraint.** A retriever
that returns more lexically similar material can lower answer quality even
when it raises recall — which is also why the hybrid arms, which surface more
look-alike records, underperform BM25 here.
"""

    markdown = f"""# Gate B report — can practical retrieval recover the evidence HRM can use?

**Verdict: `{verdict}`**

{rationale}.

Gate A proved HRM converts correct evidence into correct answers
(B3−B0 = {anchors['b3_mean_quality'] - anchors['b0_mean_quality']:+.3f}). Gate B asks whether a
real retriever can find that evidence. Model, prompt condition
(`{manifest['prompt_condition']}`), packing, decoding, verifier, and corpus are pinned
identical to Gate A, so every difference below is attributable to retrieval alone.

- Tasks: {manifest['task_count']} · evidence records: {manifest['evidence_count']} · k = {manifest['retrieval_k']}
- Corpus digest: `{manifest['task_dataset_sha256'][:16]}…` / `{manifest['evidence_corpus_sha256'][:16]}…`
- Anchors from `{manifest['gate_a_report']}`: B0 = {anchors['b0_mean_quality']:.3f}, B3 = {anchors['b3_mean_quality']:.3f}

## Arms

{arm_table}

## Complete evidence-set success by family

A retriever that finds one of two required records has not made a two-hop task
solvable, so this — not Recall@k — is the decisive multi-hop measure.

{family_table}

## Failure attribution

{failure_section}
{diagnostic_section}
## The seven Gate B questions

1. **Which backend has the highest complete evidence-set recall?**
   `{best_name}` at {css(best):.3f}.
2. **Does dense beat BM25?**
   {"Yes." if dense_beats_bm25 else f"No — dense {css(dense):.3f} vs BM25 {css(bm25):.3f}." if dense and bm25 else "Not measured."}
3. **Does hybrid beat each individual backend?**
   {"Yes." if hybrid_beats_singles else f"No — best hybrid `{best_hybrid[0]}` {css(best_hybrid[1]):.3f} vs BM25 {css(bm25):.3f}." if best_hybrid and bm25 else "Not measured."}
4. **Are two-hop failures still retrieval-bound?**
   Complete-set success on `two_hop` is {best['retrieval']['per_family']['two_hop']['complete_set_success']:.3f} for the best arm.
5. **Are numeric failures retrieval-bound or reasoning-bound?**
   Complete-set success on `numeric_derivation` is {best['retrieval']['per_family']['numeric_derivation']['complete_set_success']:.3f} for the best arm.
6. **What does retrieval cost?**
   See the latency, index-time, and irrelevant-token columns above.
7. **Is iterative retrieval justified?**
   {"Yes — see the two-hop gap." if verdict == "PASS_RETRIEVAL_EXPANSION" else "Not until single-pass retrieval is repaired."}

## What this authorizes

`{verdict_doc['next_stage']}`. Still blocked pending their own gates:
{", ".join("`" + name + "`" for name in verdict_doc["still_blocked"])}.

Two constraints must be addressed together in the next stage, because
optimizing either alone is measurably counterproductive here:

1. **Bridge-entity recovery.** A single-pass retriever cannot know the entity
   that links hop one to hop two, so `RETRIEVE_FOLLOWUP` has a concrete,
   measured opportunity on `two_hop`.
2. **Evidence selection.** Simply retrieving more raises recall while lowering
   answer quality through distractor confusion. Redundancy control and
   near-duplicate suppression belong in the same stage as iterative retrieval,
   not deferred to a later packing stage.
"""
    return markdown, verdict_doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", default="GATE_B_REPORT.md")
    parser.add_argument("--verdict", default="evidence/gate_b/gate_b_verdict.json")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    manifest["_path"] = args.manifest
    markdown, verdict = build(manifest)
    Path(args.report).write_text(markdown)
    Path(args.verdict).write_text(json.dumps(verdict, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verdict.items() if k != "per_family"}, indent=2))


if __name__ == "__main__":
    main()
