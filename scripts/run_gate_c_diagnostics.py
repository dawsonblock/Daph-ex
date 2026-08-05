#!/usr/bin/env python3
"""Gate C diagnostics: decompose the remaining multi-hop gap.

Three independent decompositions, all on the family where the gap lives, all
with the frozen model/condition/composer:

  I-arms  where does iterative retrieval lose?
          I0 one-pass · I1 deterministic bridge · I2 oracle bridge entity
          · I3 oracle second-hop evidence
          I2−I1 = bridge/query-selection headroom
          I3−I2 = retrieval headroom once the query is correct

  P-arms  how much comes from selection rather than retrieval?
          identical candidate pool, varying only the selector:
          P0 raw · P1 id-dedupe · P2 entity-anchored · P3 MMR
          · P4 anchored+MMR · P5 oracle selection

  F-arms  is the slot-label echo a prompt-interface artefact?
          identical evidence, varying only the packet's label format:
          F0 [E1] · F1 unlabelled · F2 source names · F3 XML · F4 opaque
          · F5 entity-grouped
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import IndexRecord
from hrm_adaptive_memory.evidence.packing import (
    CAPABILITY_USE_REQUIREMENT,
    compose_evidence_prompt,
    select_evidence,
)
from hrm_adaptive_memory.evidence.state import EvidenceRecordView, build_evidence_state
from hrm_adaptive_memory.evidence.sufficiency import assess
from hrm_adaptive_memory.experiments.context_study import OracleTask, verify_answer
from hrm_adaptive_memory.retrieval.iterative import TwoPassRetriever

SLOT_ECHO = re.compile(r"^\s*\[E\d+\]")


def oracle_bridge(task: OracleTask, by_id: dict) -> str | None:
    """The entity shared by the required records — the true link between hops."""

    from hrm_adaptive_memory.evidence.state import extract_entities

    question_entities = set(extract_entities(task.question))
    seen: dict[str, int] = {}
    for value in task.required_evidence_ids:
        for entity in set(extract_entities(by_id[value]["content"])):
            seen[entity] = seen.get(entity, 0) + 1
    shared = [
        entity for entity, count in seen.items()
        if count >= 2 and entity not in question_entities
    ]
    return shared[0] if shared else None


def compose_labeled(question: str, records, style: str) -> str:
    contents = [row.content for row in records]
    if style == "F0_slot_labels":
        return compose_evidence_prompt(question, contents)
    parts = ["[OBJECTIVE]", question, "[EVIDENCE]"]
    if style == "F1_unlabeled":
        parts.extend(contents)
    elif style == "F2_source_names":
        for row in records:
            parts.extend([f"[{row.source_id}]", row.content])
    elif style == "F3_xml":
        for row in records:
            parts.append(f"<evidence>{row.content}</evidence>")
    elif style == "F4_opaque":
        for index, row in enumerate(records):
            token = hashlib.sha256(f"{row.evidence_id}".encode()).hexdigest()[:6]
            parts.extend([f"[{token}]", row.content])
    elif style == "F5_entity_grouped":
        from hrm_adaptive_memory.evidence.state import extract_entities
        grouped: dict[str, list[str]] = {}
        for row in records:
            key = extract_entities(row.content)
            grouped.setdefault(key[0] if key else "other", []).append(row.content)
        for entity, items in grouped.items():
            parts.append(f"[{entity}]")
            parts.extend(items)
    else:
        raise ValueError(style)
    parts.extend(["[RESPONSE REQUIREMENT]", CAPABILITY_USE_REQUIREMENT])
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/hrm/controlled_gate_a_v2/oracle_tasks.jsonl")
    parser.add_argument("--evidence", default="data/hrm/controlled_gate_a_v2/evidence.jsonl")
    parser.add_argument("--frozen-config", default="configs/gate_a/gate_a_v2_frozen.json")
    parser.add_argument("--family", default="two_hop")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="evidence/gate_c/diagnostics")
    parser.add_argument("--phases", default="I,P,F")
    args = parser.parse_args()

    import torch
    from hrm_adaptive_memory.hrm.model import HRMAdapter, HRMModelSpec, PromptCondition

    frozen = json.loads(Path(args.frozen_config).read_text())
    tasks = [OracleTask.from_dict(json.loads(line))
             for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    evidence = [json.loads(line)
                for line in Path(args.evidence).read_text().splitlines() if line.strip()]
    by_id = {row["evidence_id"]: row for row in evidence}
    subject = [task for task in tasks if task.family == args.family]

    adapter = HRMAdapter.from_pretrained(
        spec=HRMModelSpec(), dtype=torch.bfloat16, device_map="auto",
    )
    condition = PromptCondition(frozen["prompt_condition"])

    def token_count(text: str) -> int:
        values = adapter.tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(values[0] if values and isinstance(values[0], list) else values)

    records = [IndexRecord(
        evidence_id=row["evidence_id"], source_id=row["source_id"], content=row["content"],
        token_count=token_count(row["content"]), source_type=row["source_type"],
        metadata=row["metadata"],
    ) for row in evidence]
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, records)
    index = {row.evidence_id: row for row in records}

    def as_view(evidence_id: str, rank: int) -> EvidenceRecordView:
        return EvidenceRecordView.from_retrieved(
            type("R", (), {
                "evidence_id": evidence_id, "source_id": index[evidence_id].source_id,
                "content": index[evidence_id].content,
                "token_count": index[evidence_id].token_count, "rank": rank,
            })()
        )

    def answer(task: OracleTask, views, style: str = "F0_slot_labels") -> dict:
        prompt = compose_labeled(task.question, views, style)
        generated = adapter.generate(
            prompt, condition=condition, max_new_tokens=frozen["max_new_tokens"],
        )
        text = str(generated["text"])
        quality, _ = verify_answer(task, text)
        return {
            "quality": quality, "output": text,
            "echo": bool(SLOT_ECHO.match(text)),
            "records": len(views),
            "complete": float(set(task.required_evidence_ids) <= {v.evidence_id for v in views}),
        }

    report: dict = {
        "diagnostic": "gate_c_decomposition", "family": args.family,
        "n": len(subject), "retrieval_k": args.k,
        "model_revision": adapter.spec.revision,
        "prompt_condition": frozen["prompt_condition"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def aggregate(name: str, rows: list[dict]) -> dict:
        block = {
            "quality": round(sum(r["quality"] for r in rows) / len(rows), 4),
            "complete_set_success": round(sum(r["complete"] for r in rows) / len(rows), 4),
            "slot_label_echoes": sum(1 for r in rows if r["echo"]),
            "mean_records": round(sum(r["records"] for r in rows) / len(rows), 2),
        }
        print(f"  {name:26} quality={block['quality']:.3f} "
              f"css={block['complete_set_success']:.3f} echoes={block['slot_label_echoes']}")
        return block

    # ---- I-arms -----------------------------------------------------------
    if "I" in args.phases:
        print("I-arms (iterative retrieval decomposition):")
        i_rows: dict[str, list[dict]] = {k: [] for k in
                                         ("I0_one_pass", "I1_deterministic_bridge",
                                          "I2_oracle_bridge", "I3_oracle_second_hop")}
        for task in subject:
            one = asyncio.run(TwoPassRetriever(
                backend, k=args.k, max_passes=1, enforce_anchoring=False,
            ).retrieve(task.question, select=False))
            i_rows["I0_one_pass"].append(answer(task, one.records))

            det = asyncio.run(TwoPassRetriever(
                backend, k=args.k, followup_k=args.k,
            ).retrieve(task.question))
            i_rows["I1_deterministic_bridge"].append(answer(task, det.records))

            bridge = oracle_bridge(task, by_id)
            if bridge is None:
                i_rows["I2_oracle_bridge"].append(i_rows["I1_deterministic_bridge"][-1])
            else:
                second = asyncio.run(backend.search(bridge, k=args.k))
                merged = list(one.records) + [
                    EvidenceRecordView.from_retrieved(row) for row in second.evidence
                ]
                unique: dict[str, EvidenceRecordView] = {}
                for row in merged:
                    unique.setdefault(row.evidence_id, row)
                state = build_evidence_state(question=task.question, records=tuple(unique.values()))
                selected, _ = select_evidence(
                    tuple(unique.values()),
                    anchor_entities=tuple(set(state.required_entities) | {bridge}),
                )
                i_rows["I2_oracle_bridge"].append(answer(task, selected))

            oracle_views = [as_view(value, rank)
                            for rank, value in enumerate(task.required_evidence_ids, 1)]
            i_rows["I3_oracle_second_hop"].append(answer(task, oracle_views))
        report["I_arms"] = {name: aggregate(name, rows) for name, rows in i_rows.items()}
        report["I_headroom"] = {
            "bridge_selection (I2-I1)": round(
                report["I_arms"]["I2_oracle_bridge"]["quality"]
                - report["I_arms"]["I1_deterministic_bridge"]["quality"], 4),
            "retrieval (I3-I2)": round(
                report["I_arms"]["I3_oracle_second_hop"]["quality"]
                - report["I_arms"]["I2_oracle_bridge"]["quality"], 4),
            "total (I3-I0)": round(
                report["I_arms"]["I3_oracle_second_hop"]["quality"]
                - report["I_arms"]["I0_one_pass"]["quality"], 4),
        }

    # ---- P-arms -----------------------------------------------------------
    if "P" in args.phases:
        print("P-arms (selection ablation on an identical candidate pool):")
        p_rows: dict[str, list[dict]] = {}
        for task in subject:
            result = asyncio.run(TwoPassRetriever(
                backend, k=args.k, followup_k=args.k,
            ).retrieve(task.question, select=False))
            pool = list(result.records)
            state = build_evidence_state(question=task.question, records=pool)
            anchors = tuple(set(state.required_entities) | set(state.bridge_entities))
            variants = {
                "P0_raw_pool": pool,
                "P1_id_dedupe": list({row.evidence_id: row for row in pool}.values()),
                "P2_entity_anchored": select_evidence(
                    pool, anchor_entities=anchors, lambda_redundancy=0.0)[0],
                "P3_mmr_only": select_evidence(
                    pool, anchor_entities=(), enforce_anchoring=False,
                    lambda_redundancy=0.5)[0],
                "P4_anchored_mmr": select_evidence(
                    pool, anchor_entities=anchors, lambda_redundancy=0.5)[0],
                "P5_oracle_selection": [
                    as_view(value, rank)
                    for rank, value in enumerate(task.required_evidence_ids, 1)
                    if value in index
                ],
            }
            for name, views in variants.items():
                p_rows.setdefault(name, []).append(answer(task, list(views)))
        report["P_arms"] = {name: aggregate(name, rows) for name, rows in sorted(p_rows.items())}

    # ---- F-arms -----------------------------------------------------------
    if "F" in args.phases:
        print("F-arms (packet label format, identical evidence):")
        styles = ("F0_slot_labels", "F1_unlabeled", "F2_source_names",
                  "F3_xml", "F4_opaque", "F5_entity_grouped")
        f_rows: dict[str, list[dict]] = {name: [] for name in styles}
        for task in subject:
            result = asyncio.run(backend.search(task.question, k=args.k))
            pool = [EvidenceRecordView.from_retrieved(row) for row in result.evidence]
            required = [value for value in task.required_evidence_ids
                        if value not in {row.evidence_id for row in pool}]
            views = [as_view(value, 0) for value in required] + pool
            views = views[:args.k]
            for style in styles:
                f_rows[style].append(answer(task, views, style))
        report["F_arms"] = {name: aggregate(name, rows) for name, rows in f_rows.items()}

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate_c_diagnostics.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k.endswith(("_arms", "_headroom"))}, indent=2))


if __name__ == "__main__":
    main()
