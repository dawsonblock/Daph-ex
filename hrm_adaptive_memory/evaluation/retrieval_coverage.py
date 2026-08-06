"""Gate C2 — retrieval coverage measured without the reader.

The V4 ladder showed complete-evidence-set success of only 0.134–0.168 across
every real arm: retrieval finds the whole required set for roughly one task in
six. Answer accuracy cannot separate "retrieval never found it" from "the
reader could not use it", so coverage is measured here directly, with no HRM in
the loop and therefore no GPU.

`CompleteEvidenceSet@K` is the primary metric. A two-hop task stays unsolvable
when only one of its required records is retrieved, so mean Recall@K flatters a
retriever that reliably finds half of each proof path.

`AllRequiredPresentRate` is the ceiling an oracle selector could reach from the
candidate pool — it bounds what any downstream selection can recover.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Mapping, Sequence

DEPTHS = (1, 5, 10, 20, 50)


@dataclass(frozen=True)
class RetrievalGroundTruth:
    """Evaluator-only truth. Never reaches a query, index, selector, or prompt."""

    task_id: str
    family: str
    entity_regime: str
    answer_kind: str
    source_style: str
    opportunity_group: str
    required_ids: tuple[str, ...]
    proof_path_ids: tuple[str, ...]
    bridge_ids: tuple[str, ...]
    answer_record_ids: tuple[str, ...]
    identity_record_ids: tuple[str, ...] = ()

    def weights(self, *, answer_weight: float = 3.0, bridge_weight: float = 2.0,
                base_weight: float = 1.0) -> dict[str, float]:
        """Evaluator-only record weights for partial proof coverage.

        R4 answered 46.2% of tasks while holding the complete evidence set for
        only 16.8%, so complete-set success is a necessary metric but not a
        sufficient proxy for answerability: a partial packet containing the
        answer-bearing record is often enough. Weighting lets partial coverage
        reflect that without pretending it is a causal model of the reader.
        """

        out: dict[str, float] = {}
        for value in self.required_ids:
            weight = base_weight
            if value in self.bridge_ids:
                weight = max(weight, bridge_weight)
            if value in self.answer_record_ids:
                weight = max(weight, answer_weight)
            out[value] = weight
        return out

    @classmethod
    def from_task(cls, task: Mapping[str, Any]) -> "RetrievalGroundTruth":
        meta = task["_oracle_metadata"]
        edges = meta["proof_edges"]
        required = tuple(task["required_evidence_ids"])
        proof = tuple(dict.fromkeys(edge["record_id"] for edge in edges))
        bridge = tuple(dict.fromkeys(
            edge["record_id"] for edge in edges
            if meta.get("latent_bridge") and edge["target"] == meta["latent_bridge"]))
        answers = tuple(dict.fromkeys(
            edge["record_id"] for edge in edges if edge["target"] == meta["answer_node"]))
        # Identity records are what an alias/description question must traverse
        # before the target relation is even addressable.
        identity = tuple(dict.fromkeys(
            edge["record_id"] for edge in edges
            if str(edge.get("source", "")).startswith("surface:")))
        return cls(
            identity_record_ids=identity,
            task_id=task["task_id"], family=task["family"],
            entity_regime=task["metadata"]["entity_regime"],
            answer_kind=task["metadata"]["answer_kind"],
            source_style=task["metadata"]["source_style"],
            opportunity_group=task["metadata"]["opportunity_group"],
            required_ids=required, proof_path_ids=proof,
            bridge_ids=bridge, answer_record_ids=answers,
        )


@dataclass(frozen=True)
class CoverageResult:
    task_id: str
    retriever: str
    retrieved: tuple[str, ...]
    recall_at: Mapping[int, float]
    complete_set_at: Mapping[int, float]
    proof_path_coverage_at: Mapping[int, float]
    partial_proof_coverage_at: Mapping[int, float]
    # None for tasks that have no bridge at all. Reporting False there would
    # let single-hop tasks dilute a bridge-recall statistic.
    bridge_found: float | None
    # Alias decomposition. An alias score of 0.08 vs 0.12 is uninformative;
    # "identity found 72% / canonicalized 61% / target recovered 19%" names the
    # failing mechanism.
    identity_record_found: float | None
    target_relation_record_found: float | None
    answer_record_found: float
    mrr: float
    ndcg: float
    precision_at_k: float
    candidate_pool_size: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key in ("recall_at", "complete_set_at", "proof_path_coverage_at",
                    "partial_proof_coverage_at"):
            row[key] = {str(k): v for k, v in getattr(self, key).items()}
        return row


def score_coverage(
    truth: RetrievalGroundTruth, retrieved: Sequence[str], *,
    retriever: str, latency_ms: float = 0.0, depths: Sequence[int] = DEPTHS,
) -> CoverageResult:
    ranked = list(retrieved)
    required = set(truth.required_ids)
    proof = set(truth.proof_path_ids)
    hits = [value in required for value in ranked]

    weights = truth.weights()
    total_weight = sum(weights.values()) or 1.0
    recall_at, complete_at, proof_at, partial_at = {}, {}, {}, {}
    for depth in depths:
        window = set(ranked[:depth])
        recall_at[depth] = len(window & required) / max(1, len(required))
        complete_at[depth] = float(required <= window)
        proof_at[depth] = len(window & proof) / max(1, len(proof))
        partial_at[depth] = sum(
            weight for value, weight in weights.items() if value in window) / total_weight

    first = next((index for index, hit in enumerate(hits, 1) if hit), None)
    ideal = min(len(ranked), len(required))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    dcg = sum((1.0 if hit else 0.0) / math.log2(i + 1) for i, hit in enumerate(hits, 1))

    return CoverageResult(
        task_id=truth.task_id, retriever=retriever, retrieved=tuple(ranked),
        recall_at=recall_at, complete_set_at=complete_at, proof_path_coverage_at=proof_at,
        partial_proof_coverage_at=partial_at,
        bridge_found=(None if not truth.bridge_ids
                      else float(set(truth.bridge_ids) <= set(ranked))),
        identity_record_found=(None if not truth.identity_record_ids
                               else float(set(truth.identity_record_ids) <= set(ranked))),
        # The answer-bearing record is the target-relation record; it is only
        # reachable once identity is resolved, so both are reported.
        target_relation_record_found=(None if not truth.answer_record_ids
                                      else float(bool(set(truth.answer_record_ids) & set(ranked)))),
        answer_record_found=float(
            bool(truth.answer_record_ids) and bool(set(truth.answer_record_ids) & set(ranked))),
        mrr=0.0 if first is None else 1.0 / first,
        ndcg=dcg / idcg if idcg else 0.0,
        precision_at_k=len(set(ranked) & required) / max(1, len(ranked)),
        candidate_pool_size=len(ranked), latency_ms=latency_ms,
    )


AXES = ("family", "entity_regime", "answer_kind", "source_style", "opportunity_group")


def summarize_coverage(
    results: Sequence[CoverageResult], truths: Mapping[str, RetrievalGroundTruth],
    *, retriever: str, depths: Sequence[int] = DEPTHS,
) -> dict[str, Any]:
    if not results:
        return {"retriever": retriever, "task_count": 0}

    def block(rows: Sequence[CoverageResult]) -> dict[str, Any]:
        return {
            "n": len(rows),
            **{f"recall@{d}": round(mean(r.recall_at[d] for r in rows), 4) for d in depths},
            **{f"complete_set@{d}": round(mean(r.complete_set_at[d] for r in rows), 4)
               for d in depths},
            **{f"proof_path@{d}": round(mean(r.proof_path_coverage_at[d] for r in rows), 4)
               for d in depths},
            **{f"partial_proof@{d}": round(mean(r.partial_proof_coverage_at[d] for r in rows), 4)
               for d in depths},
            "mrr": round(mean(r.mrr for r in rows), 4),
            "ndcg": round(mean(r.ndcg for r in rows), 4),
            "precision_at_k": round(mean(r.precision_at_k for r in rows), 4),
            # Conditioned on tasks that actually have a bridge.
            "bridge_recall_among_bridge_tasks": (
                round(mean(r.bridge_found for r in rows if r.bridge_found is not None), 4)
                if any(r.bridge_found is not None for r in rows) else None),
            "bridge_task_count": sum(1 for r in rows if r.bridge_found is not None),
            "identity_record_recall_among_identity_tasks": (
                round(mean(r.identity_record_found for r in rows
                           if r.identity_record_found is not None), 4)
                if any(r.identity_record_found is not None for r in rows) else None),
            "identity_task_count": sum(1 for r in rows if r.identity_record_found is not None),
            "target_relation_record_recall": (
                round(mean(r.target_relation_record_found for r in rows
                           if r.target_relation_record_found is not None), 4)
                if any(r.target_relation_record_found is not None for r in rows) else None),
            # Conditional: the target relation is only addressable after identity
            # resolves, so mixing the two would hide which mechanism failed.
            "target_recall_given_identity_found": (
                round(mean(r.target_relation_record_found for r in rows
                           if r.identity_record_found == 1.0
                           and r.target_relation_record_found is not None), 4)
                if any(r.identity_record_found == 1.0 for r in rows) else None),
            "answer_record_found": round(mean(r.answer_record_found for r in rows), 4),
            "mean_pool_size": round(mean(r.candidate_pool_size for r in rows), 2),
            "mean_latency_ms": round(mean(r.latency_ms for r in rows), 3),
        }

    overall = block(results)
    # The ceiling an oracle selector could reach from these pools.
    overall["all_required_present_rate"] = overall[f"complete_set@{max(depths)}"]

    by_axis: dict[str, dict[str, Any]] = {}
    for axis in AXES:
        buckets: dict[str, list[CoverageResult]] = {}
        for row in results:
            buckets.setdefault(getattr(truths[row.task_id], axis), []).append(row)
        by_axis[axis] = {name: block(rows) for name, rows in sorted(buckets.items())}

    return {"retriever": retriever, "task_count": len(results),
            "overall": overall, "by_axis": by_axis}
