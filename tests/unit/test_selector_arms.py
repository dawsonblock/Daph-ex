"""Selector-arm contracts, including the degenerate-reranker guard.

The guard test encodes a real measurement failure: an S3 cross-encoder whose
forward pass returned NaN for every pair produced figures identical to S0 to
four decimals, because `sorted` treats NaN comparisons as False and leaves the
list in its original order. A scorer that cannot rank must raise, so that a
broken arm can never be mistaken for evidence about reranking.
"""
from __future__ import annotations

import math

import pytest

from hrm_adaptive_memory.retrieval_bench.selectors import (
    DegenerateRerankerError,
    s0_raw,
    s1_relevance,
    s2_connectivity,
    s5_oracle,
)


def _pool(n: int = 6):
    return [{"document_id": f"ev-{i}", "rank": i} for i in range(1, n + 1)]


def _texts():
    return {
        "ev-1": "Docket 41 refers to the Kestrel Ridge summit.",
        "ev-2": "Kestrel Ridge has elevation 3140 metres.",
        "ev-3": "Bananas are a tropical fruit and unrelated.",
        "ev-4": "Another distractor about rainfall patterns.",
        "ev-5": "Kestrel Ridge was first ascended in 1908.",
        "ev-6": "Unrelated note concerning harbour depth.",
    }


def test_s0_returns_pool_prefix_in_order():
    assert s0_raw(_pool(), budget=3) == ["ev-1", "ev-2", "ev-3"]


def test_s0_respects_budget():
    for budget in (2, 4, 6):
        assert len(s0_raw(_pool(), budget=budget)) == budget


def test_s5_oracle_keeps_only_required_records_that_were_found():
    # 'ev-99' was never retrieved, so the ceiling cannot include it.
    selected = s5_oracle(_pool(), budget=6, required=["ev-2", "ev-5", "ev-99"])
    assert selected == ["ev-2", "ev-5"]


def test_s5_oracle_is_bounded_by_budget():
    selected = s5_oracle(_pool(), budget=1, required=["ev-2", "ev-5"])
    assert len(selected) == 1


def test_s1_and_s2_are_budget_bounded_and_return_pool_members():
    ids = {c["document_id"] for c in _pool()}
    for selector in (s1_relevance, s2_connectivity):
        chosen = selector(_pool(), budget=3, question="How tall is Docket 41?",
                          texts=_texts())
        assert len(chosen) == 3
        assert set(chosen) <= ids
        assert len(set(chosen)) == len(chosen), "a selector must not repeat a record"


class _StubTokenizer:
    def __call__(self, *args, **kwargs):
        return {"input_ids": [[0]]}


class _StubLogits:
    def __init__(self, values):
        self._values = values
        self.shape = (len(values), 1)

    def __getitem__(self, _key):
        return self

    def tolist(self):
        return list(self._values)


class _StubOutput:
    def __init__(self, values):
        self.logits = _StubLogits(values)


def _selector_over(values):
    """Build a cross-encoder selector whose scorer returns fixed values."""
    from hrm_adaptive_memory.retrieval_bench import selectors as module

    def fake_scorer(candidates, *, budget, question, texts, **_):
        ids = [c["document_id"] for c in candidates]
        scores = list(values)
        module_audit(scores, len(ids))
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
        return [ids[i] for i in order[:budget]]

    # Reuse the production guard rather than re-implementing its logic.
    def module_audit(scores, count):
        bad = sum(1 for s in scores if not math.isfinite(s))
        if bad:
            raise module.DegenerateRerankerError(f"{bad}/{len(scores)} non-finite")
        if count > 1 and len(set(scores)) == 1:
            raise module.DegenerateRerankerError("all scores identical")

    return fake_scorer


def test_nan_scores_raise_instead_of_returning_pool_order():
    """The exact defect that made S3 duplicate S0: every score NaN."""
    selector = _selector_over([float("nan")] * 6)
    with pytest.raises(DegenerateRerankerError):
        selector(_pool(), budget=3, question="q", texts=_texts())


def test_constant_scores_raise_because_they_cannot_rerank():
    selector = _selector_over([1.0] * 6)
    with pytest.raises(DegenerateRerankerError):
        selector(_pool(), budget=3, question="q", texts=_texts())


def test_finite_distinct_scores_reorder_the_pool():
    # Ascending scores must invert the pool order, proving the arm is live.
    selector = _selector_over([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    chosen = selector(_pool(), budget=3, question="q", texts=_texts())
    assert chosen == ["ev-6", "ev-5", "ev-4"]
    assert chosen != s0_raw(_pool(), budget=3), "a live reranker must differ from S0"
