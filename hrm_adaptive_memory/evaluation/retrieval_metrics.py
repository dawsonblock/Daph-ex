"""Gate B retrieval metrics, centered on complete evidence-set recovery.

A retriever that finds one of two required facts has not made a two-hop task
solvable, so ``complete_set_success`` — not Recall@k — is the decisive
multi-hop measure.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

RECALL_DEPTHS = (1, 3, 5, 10)


@dataclass(frozen=True)
class TaskRetrievalMetrics:
    task_id: str
    family: str
    backend_id: str
    requested_k: int
    retrieved_ids: tuple[str, ...]
    required_ids: tuple[str, ...]
    recall_at: Mapping[int, float]
    precision_at_k: float
    mrr: float
    ndcg: float
    required_evidence_recall: float
    complete_set_success: float
    redundancy: float
    irrelevant_token_ratio: float
    evidence_tokens: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["recall_at"] = {str(depth): value for depth, value in self.recall_at.items()}
        return row


def _dcg(hits: Sequence[bool]) -> float:
    return sum((1.0 if hit else 0.0) / math.log2(index + 1) for index, hit in enumerate(hits, 1))


def score_task(
    *, task_id: str, family: str, backend_id: str, requested_k: int,
    retrieved_ids: Sequence[str], required_ids: Iterable[str],
    token_counts: Mapping[str, int] | None = None, latency_ms: float = 0.0,
) -> TaskRetrievalMetrics:
    required = set(required_ids)
    ranked = list(retrieved_ids)
    unique_ranked = list(dict.fromkeys(ranked))
    hits = [value in required for value in ranked]

    recall_at = {}
    for depth in RECALL_DEPTHS:
        found = set(ranked[:depth]) & required
        recall_at[depth] = len(found) / max(1, len(required))

    first_hit = next((index for index, hit in enumerate(hits, 1) if hit), None)
    ideal = min(len(ranked), len(required))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal + 1))
    retrieved_required = set(ranked) & required

    tokens = token_counts or {}
    evidence_tokens = sum(int(tokens.get(value, 0)) for value in unique_ranked)
    irrelevant_tokens = sum(
        int(tokens.get(value, 0)) for value in unique_ranked if value not in required
    )

    return TaskRetrievalMetrics(
        task_id=task_id,
        family=family,
        backend_id=backend_id,
        requested_k=requested_k,
        retrieved_ids=tuple(ranked),
        required_ids=tuple(sorted(required)),
        recall_at=recall_at,
        precision_at_k=len(retrieved_required) / max(1, len(ranked)),
        mrr=0.0 if first_hit is None else 1.0 / first_hit,
        ndcg=_dcg(hits) / idcg if idcg else 0.0,
        required_evidence_recall=len(retrieved_required) / max(1, len(required)),
        complete_set_success=float(required <= set(ranked)),
        redundancy=1.0 - len(unique_ranked) / max(1, len(ranked)),
        irrelevant_token_ratio=irrelevant_tokens / evidence_tokens if evidence_tokens else 0.0,
        evidence_tokens=evidence_tokens,
        latency_ms=latency_ms,
    )


@dataclass
class RetrievalSummary:
    backend_id: str
    task_count: int
    metrics: dict[str, float]
    per_family: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _aggregate(rows: Sequence[TaskRetrievalMetrics]) -> dict[str, float]:
    if not rows:
        return {}
    summary = {
        f"recall_at_{depth}": round(mean(row.recall_at[depth] for row in rows), 4)
        for depth in RECALL_DEPTHS
    }
    summary.update({
        "precision_at_k": round(mean(row.precision_at_k for row in rows), 4),
        "mrr": round(mean(row.mrr for row in rows), 4),
        "ndcg": round(mean(row.ndcg for row in rows), 4),
        "required_evidence_recall": round(mean(row.required_evidence_recall for row in rows), 4),
        "complete_set_success": round(mean(row.complete_set_success for row in rows), 4),
        "redundancy": round(mean(row.redundancy for row in rows), 4),
        "irrelevant_token_ratio": round(mean(row.irrelevant_token_ratio for row in rows), 4),
        "mean_evidence_tokens": round(mean(row.evidence_tokens for row in rows), 2),
        "mean_latency_ms": round(mean(row.latency_ms for row in rows), 3),
        "p95_latency_ms": round(sorted(row.latency_ms for row in rows)[int(0.95 * (len(rows) - 1))], 3),
    })
    return summary


def summarize(rows: Sequence[TaskRetrievalMetrics], *, backend_id: str) -> RetrievalSummary:
    """Aggregate overall and per family — easy families must never hide hard ones."""

    families: dict[str, list[TaskRetrievalMetrics]] = {}
    for row in rows:
        families.setdefault(row.family, []).append(row)
    return RetrievalSummary(
        backend_id=backend_id,
        task_count=len(rows),
        metrics=_aggregate(rows),
        per_family={family: _aggregate(items) for family, items in sorted(families.items())},
    )
