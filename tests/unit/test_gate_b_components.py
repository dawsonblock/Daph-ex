"""Gate B building blocks: retrieval arms, metrics, failure attribution."""

from __future__ import annotations

import asyncio

import pytest

from hrm_adaptive_memory.backends import CanonicalRetrievalBackend, CanonicalRetrievalMode
from hrm_adaptive_memory.contracts import EvidenceFilter, IndexRecord
from hrm_adaptive_memory.evaluation.failure_analysis import FailureClass, classify, summarize
from hrm_adaptive_memory.evaluation.resources import ResourceLedger
from hrm_adaptive_memory.evaluation.retrieval_metrics import score_task
from hrm_adaptive_memory.evaluation.retrieval_metrics import summarize as summarize_retrieval
from hrm_adaptive_memory.retrieval.embedding import EmbeddingSpec


def run(value):
    return asyncio.run(value)


class StubEmbedder:
    """Deterministic 3-dim embedder; no network, no model download."""

    spec = EmbeddingSpec(dimension=3)

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "adapter" in lowered else 0.0,
            1.0 if "category" in lowered else 0.0,
            1.0 if "trial" in lowered else 0.0,
        ]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]


def records():
    return [
        IndexRecord(evidence_id="hop-1", source_id="s1", token_count=12,
                    content="The deployment record for Trial-000-867 lists Adapter-78103 as its adapter."),
        IndexRecord(evidence_id="hop-2", source_id="s2", token_count=10,
                    content="The classification registry maps Adapter-78103 to category code 840."),
        IndexRecord(evidence_id="noise-1", source_id="s3", token_count=9,
                    content="Unrelated maintenance note about scheduling and staffing."),
    ]


@pytest.mark.parametrize("mode", list(CanonicalRetrievalMode))
def test_every_canonical_arm_returns_receipted_results(mode):
    backend = CanonicalRetrievalBackend(mode, records(), embedder=StubEmbedder()
                                        if mode != CanonicalRetrievalMode.HASH else None)
    result = run(backend.search("Which category code applies to Trial-000-867?", k=3))
    assert result.evidence, f"{mode.value} returned nothing"
    assert result.receipt.returned_ids == tuple(row.evidence_id for row in result.evidence)
    assert result.receipt.backend_id == f"canonical:{mode.value}"
    assert result.receipt.latency_ms >= 0
    assert len({row.rank for row in result.evidence}) == len(result.evidence)


def test_canonical_arms_reject_filters_instead_of_silently_ignoring_them():
    backend = CanonicalRetrievalBackend(CanonicalRetrievalMode.BM25, records())
    with pytest.raises(RuntimeError, match="filters"):
        run(backend.search("query", k=1, filters=EvidenceFilter(tags=("x",))))


def test_config_digest_changes_with_fusion_parameters():
    left = CanonicalRetrievalBackend(CanonicalRetrievalMode.HYBRID_RRF, records(), embedder=StubEmbedder())
    right = CanonicalRetrievalBackend(CanonicalRetrievalMode.HYBRID_RRF, records(),
                                      embedder=StubEmbedder(), rrf_k=17)
    assert left.config_digest() != right.config_digest()


def test_embedding_spec_digest_is_pinned_and_sensitive():
    base = EmbeddingSpec()
    assert base.digest() == EmbeddingSpec().digest()
    for changed in (
        EmbeddingSpec(revision="deadbeef"),
        EmbeddingSpec(pooling="cls"),
        EmbeddingSpec(normalize=False),
        EmbeddingSpec(max_sequence_length=512),
    ):
        assert changed.digest() != base.digest()


def test_complete_set_success_requires_every_required_record():
    partial = score_task(
        task_id="t1", family="two_hop", backend_id="b", requested_k=10,
        retrieved_ids=["hop-1", "noise-1"], required_ids=["hop-1", "hop-2"],
    )
    assert partial.required_evidence_recall == 0.5
    assert partial.complete_set_success == 0.0, "half the evidence does not solve a two-hop task"

    full = score_task(
        task_id="t1", family="two_hop", backend_id="b", requested_k=10,
        retrieved_ids=["hop-1", "noise-1", "hop-2"], required_ids=["hop-1", "hop-2"],
    )
    assert full.complete_set_success == 1.0
    assert full.recall_at[1] == 0.5 and full.recall_at[3] == 1.0


def test_irrelevant_token_ratio_and_redundancy():
    row = score_task(
        task_id="t1", family="single_hop", backend_id="b", requested_k=3,
        retrieved_ids=["hop-1", "noise-1", "noise-1"], required_ids=["hop-1"],
        token_counts={"hop-1": 10, "noise-1": 30},
    )
    assert row.evidence_tokens == 40
    assert row.irrelevant_token_ratio == pytest.approx(0.75)
    assert row.redundancy == pytest.approx(1 / 3)


def test_family_summary_does_not_hide_hard_families():
    rows = [
        score_task(task_id=f"s{i}", family="single_hop", backend_id="b", requested_k=10,
                   retrieved_ids=["a"], required_ids=["a"]) for i in range(4)
    ] + [
        score_task(task_id=f"h{i}", family="two_hop", backend_id="b", requested_k=10,
                   retrieved_ids=["a"], required_ids=["a", "b"]) for i in range(4)
    ]
    summary = summarize_retrieval(rows, backend_id="b")
    assert summary.metrics["complete_set_success"] == 0.5
    assert summary.per_family["single_hop"]["complete_set_success"] == 1.0
    assert summary.per_family["two_hop"]["complete_set_success"] == 0.0


def test_failure_attribution_separates_retrieval_from_reasoning():
    retrieval_bound = classify(
        task_id="t1", family="two_hop", quality=0.0, answer="840", output="176",
        required_ids=["hop-1", "hop-2"], retrieved_ids=["hop-1", "noise-1"],
    )
    assert retrieval_bound.failure_class == FailureClass.RETRIEVAL_FAILURE
    assert retrieval_bound.missing_required_ids == ("hop-2",)

    reasoning_bound = classify(
        task_id="t2", family="two_hop", quality=0.0, answer="840", output="176",
        required_ids=["hop-1", "hop-2"], retrieved_ids=["hop-1", "hop-2"],
    )
    assert reasoning_bound.failure_class == FailureClass.REASONING_FAILURE


def test_failure_attribution_detects_packing_and_calculation():
    packing = classify(
        task_id="t3", family="single_hop", quality=0.0, answer="63287", output="1401",
        required_ids=["e1"], retrieved_ids=["e1"], prompt_evidence_ids=["other"],
    )
    assert packing.failure_class == FailureClass.PACKING_FAILURE
    assert packing.dropped_in_packing_ids == ("e1",)

    calculation = classify(
        task_id="t4", family="numeric_derivation", quality=0.0, answer="114", output="12",
        required_ids=["u", "m"], retrieved_ids=["u", "m"],
        evidence_contents={"u": "records 19 units", "m": "multiplies by 6"},
    )
    assert calculation.failure_class == FailureClass.CALCULATION_FAILURE


def test_verification_artifact_is_not_a_model_failure():
    row = classify(
        task_id="t5", family="single_hop", quality=0.0, answer="840",
        output="The code is 840, as recorded in registry 12.",
        required_ids=["e1"], retrieved_ids=["e1"],
    )
    assert row.failure_class == FailureClass.VERIFICATION_FAILURE


def test_failure_summary_counts_and_retrieval_bound_fraction():
    rows = [
        classify(task_id="a", family="two_hop", quality=1.0, answer="1", output="1",
                 required_ids=["x"], retrieved_ids=["x"]),
        classify(task_id="b", family="two_hop", quality=0.0, answer="1", output="2",
                 required_ids=["x", "y"], retrieved_ids=["x"]),
        classify(task_id="c", family="two_hop", quality=0.0, answer="1", output="2",
                 required_ids=["x"], retrieved_ids=["x"]),
    ]
    report = summarize(rows)
    assert report["failure_count"] == 2
    assert report["counts"]["RETRIEVAL_FAILURE"] == 1
    assert report["counts"]["REASONING_FAILURE"] == 1
    assert report["retrieval_bound_fraction"] == 0.5


def test_resource_ledger_attributes_phases_and_never_fakes_memory():
    ledger = ResourceLedger()
    with ledger.wall_clock():
        with ledger.phase("retrieval"):
            sum(range(1000))
        with ledger.phase("model"):
            sum(range(1000))
    row = ledger.to_dict()
    assert row["latency_ms"]["retrieval_ms"] > 0
    assert row["latency_ms"]["model_ms"] > 0
    assert row["latency_ms"]["verification_ms"] == 0
    assert row["latency_ms"]["unattributed_ms"] is not None
    # tracemalloc is never reported as physical model memory
    assert row["memory"]["python_allocator_peak_bytes"] is None
    with pytest.raises(ValueError, match="Unknown resource phase"):
        with ledger.phase("teleportation"):
            pass


def test_tokenizer_does_not_glue_trailing_punctuation_to_entities():
    """A sentence-final entity must tokenize identically to a mid-sentence one.

    Regression: the original class-based pattern included '.' and '-' as
    ordinary members, so "Plan-000-965." became a distinct token from
    "Plan-000-965" and lexical retrieval silently missed the evidence.
    """

    from hrm_adaptive_memory.retrieval.lexical import tokenize

    final = tokenize("The base ledger records 32 units for Plan-000-965.")
    middle = tokenize("The operating rule for Plan-000-965 multiplies its units by 7.")
    query = tokenize("Plan-000-965")
    assert query == ["plan-000-965"]
    assert "plan-000-965" in final and "plan-000-965" in middle
    assert "plan-000-965." not in final
    assert "7" in middle and "7." not in middle
    # Internal separators are still preserved.
    assert tokenize("Config v1.2 in src/main.py, done.") == [
        "config", "v1.2", "in", "src/main.py", "done",
    ]


def test_lexical_retrieval_finds_sentence_final_entities():
    from hrm_adaptive_memory.retrieval.lexical import BM25Retriever
    from hrm_adaptive_memory.memory.chunking import Chunk

    chunks = [
        Chunk(chunk_id="units", source_id="s1", source_type="source", title="t", section="",
              content="The base ledger records 32 units for Plan-000-965.", token_count=9, metadata={}),
        Chunk(chunk_id="mult", source_id="s2", source_type="source", title="t", section="",
              content="The operating rule for Plan-000-965 multiplies its units by 7.", token_count=11, metadata={}),
        Chunk(chunk_id="other", source_id="s3", source_type="source", title="t", section="",
              content="The operating rule for Plan-777-111 multiplies its units by 3.", token_count=11, metadata={}),
    ]
    hits = {chunk.chunk_id for chunk, _ in BM25Retriever(chunks).search("Plan-000-965", 10)}
    assert hits == {"units", "mult"}, "entity query must recover both records and exclude other plans"
