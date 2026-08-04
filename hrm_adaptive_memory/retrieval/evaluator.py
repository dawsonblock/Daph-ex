from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    ndcg: float
    oracle_evidence_recall: float


def evaluate_retrieval(ranked_ids: Sequence[str], relevant_ids: Iterable[str], *, k: int,
                       required_evidence_ids: Iterable[str] | None = None) -> RetrievalMetrics:
    relevant = set(relevant_ids); ranked = list(ranked_ids[:k]); hits = [identifier in relevant for identifier in ranked]
    recall = len(set(ranked) & relevant) / max(1, len(relevant))
    first = next((index for index, hit in enumerate(hits, 1) if hit), None)
    dcg = sum((1.0 if hit else 0.0) / math.log2(index + 1) for index, hit in enumerate(hits, 1))
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    required = set(required_evidence_ids if required_evidence_ids is not None else relevant)
    return RetrievalMetrics(
        recall_at_k=recall, mrr=0.0 if first is None else 1.0 / first,
        ndcg=dcg / idcg if idcg else 0.0,
        oracle_evidence_recall=float(required <= set(ranked)),
    )
